from typing import Dict
import torch
import torch.nn as nn
import sys
from omegaconf import OmegaConf

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
sys.path.append('/home/siyi/project/mm/mul_foundation/Dynamic_Missing')
from models.utils.MVAE.BaseMMVae import BaseMMVae
from models.utils.MVAE.Modality import Modality
from itertools import chain, combinations
from models.utils.MVAE.divergence_measures.kl_div import calc_kl_divergence
from models.utils.MVAE import utils

class EncoderImg(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, flags):
        super(EncoderImg, self).__init__()

        self.flags = flags
        self.shared_encoder = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),     # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # -> (128, 4, 4)
            nn.ReLU(),
            utils.Flatten(),                                                # -> (2048)
            nn.Linear(2048, flags.style_dim + flags.class_dim),       # -> (ndim_private + ndim_shared)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Linear(flags.style_dim + flags.class_dim, flags.class_dim)
        self.class_logvar = nn.Linear(flags.style_dim + flags.class_dim, flags.class_dim)
        # optional style branch
        if flags.factorized_representation:
            self.style_mu = nn.Linear(flags.style_dim + flags.class_dim, flags.style_dim)
            self.style_logvar = nn.Linear(flags.style_dim + flags.class_dim, flags.style_dim)

    def forward(self, x):
        h = self.shared_encoder(x)
        if self.flags.factorized_representation:
            return self.style_mu(h), self.style_logvar(h), self.class_mu(h), \
                   self.class_logvar(h)
        else:
            return None, None, self.class_mu(h), self.class_logvar(h)


class DecoderImg(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, flags):
        super(DecoderImg, self).__init__()
        self.flags = flags
        self.decoder = nn.Sequential(
            nn.Linear(flags.style_dim + flags.class_dim, 2048),                                # -> (2048)
            nn.ReLU(),
            utils.Unflatten((128, 4, 4)),                                                            # -> (128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),                   # -> (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (3, 28, 28)
        )

    def forward(self, style_latent_space, class_latent_space):
        if self.flags.factorized_representation:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        x_hat = self.decoder(z)
        # x_hat = torch.sigmoid(x_hat)
        return x_hat, torch.tensor(0.75).to(z.device)  # NOTE: consider learning scale param, too


def calc_log_probs(modalities, result, batch, rec_weights):
    mods = modalities;
    log_probs = dict()
    weighted_log_prob = 0.0;
    for m, m_key in enumerate(mods.keys()):
        mod = mods[m_key]
        log_probs[mod.name] = -mod.calc_log_prob(result['rec'][mod.name],
                                                 batch[mod.name],
                                                 batch[mod.name].shape[0]);
        weighted_log_prob += rec_weights[mod.name]*log_probs[mod.name];
    return log_probs, weighted_log_prob;

def calc_klds(flags, result):
    latents = result['latents']['subsets'];
    klds = dict();
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key];
        klds[key] = calc_kl_divergence(mu, logvar,
                                       norm_value=flags.batch_size)
    return klds;

def calc_klds_style(flags, result):
    latents = result['latents']['modalities'];
    klds = dict();
    for m, key in enumerate(latents.keys()):
        if key.endswith('style'):
            mu, logvar = latents[key];
            klds[key] = calc_kl_divergence(mu, logvar,
                                           norm_value=flags.batch_size)
    return klds;


def calc_style_kld(modalities, style_weights, klds):
    mods = modalities;
    style_weights = style_weights;
    weighted_klds = 0.0;
    for m, m_key in enumerate(mods.keys()):
        weighted_klds += style_weights[m_key]*klds[m_key+'_style'];
    return weighted_klds;



class MoPoE(nn.Module):
    '''
    Multimodal VAE
    Use PoE within each subset and MoE among all subsets
    '''
    def __init__(self, args):
        super(MoPoE, self).__init__()
        self.num_modalities = args.num_modalities
        flags = {'modality_moe': False, 'modality_jsd': False, 'joint_elbo': True, 'modality_poe': False,
                 'alpha_modalities': [1/(self.num_modalities+1)]*(self.num_modalities+1), 'batch_size': args.batch_size,
                 'device': None, 'class_dim': 512, 'style_dim': 0, 'factorized_representation': False,
                 'dir_checkpoints':args.logdir, 'beta_style': 1.0, 'beta_content': 1.0, 'beta': 5.0}
        self.flags = OmegaConf.create(flags)
        self.modalities = self.set_modalities()
        self.subsets = self.set_subsets()
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()
        self.model = BaseMMVae(self.flags, self.modalities, self.subsets)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            self.model.load_state_dict(ckpt)
            print(f"Load model from {args.checkpoint}")

    def set_modalities(self,):
        mods = [Modality("m%d" % m, EncoderImg(self.flags), DecoderImg(self.flags),
                       10, 0, 'laplace') for m in range(self.num_modalities)]
        mods_dict = {m.name: m for m in mods}
        return mods_dict

    def set_subsets(self):
        """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
        (1,2,3)
        """
        num_mods = len(list(self.modalities.keys()));
        xs = list(self.modalities)
        # note we return an iterator rather than a list
        subsets_list = chain.from_iterable(combinations(xs, n) for n in
                                          range(len(xs)+1))
        subsets = dict();
        mod_names_dict = {'m0': 0, 'm1': 1, 'm2': 2, 'm3': 3, 'm4': 4}
        self.subset2name = {}
        for k, mod_names in enumerate(subsets_list):
            mods = [];
            for l, mod_name in enumerate(sorted(mod_names)):
                mods.append(self.modalities[mod_name])
            key = '_'.join(sorted(mod_names));
            subsets[key] = mods;
            # create mask2name
            mask = torch.ones(len(self.modalities), dtype=torch.bool)
            mod_idx = [mod_names_dict[m] for m in mod_names]
            mask[mod_idx] = False
            self.subset2name[tuple(mask.tolist())] = key
        return subsets;

    def set_rec_weights(self):
        rec_weights = dict()
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            # numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = 1.0
        return rec_weights
    
    def set_style_weights(self):
        weights = {"m%d" % m: self.flags.beta_style for m in range(self.num_modalities)}
        return weights

    def forward_loss(self, target, results, mask: torch.Tensor):
        beta_style = self.flags.beta_style
        beta_content = self.flags.beta_content
        beta = self.flags.beta
        rec_weight = 1.0
        mods = self.modalities

        log_probs, weighted_log_prob = calc_log_probs(self.modalities, results, target, self.rec_weights)
        group_divergence = results['joint_divergence']

        klds = calc_klds(self.flags, results);
        if self.flags.factorized_representation:
            klds_style = calc_klds_style(self.flags, results);

        if (self.flags.modality_jsd or self.flags.modality_moe
            or self.flags.joint_elbo):
            if self.flags.factorized_representation:
                kld_style = calc_style_kld(self.modalities, self.style_weights, klds_style);
            else:
                kld_style = 0.0;
            kld_content = group_divergence;
            kld_weighted = beta_style * kld_style + beta_content * kld_content;
            total_loss = rec_weight * weighted_log_prob + beta * kld_weighted;
        elif self.flags.modality_poe:
            klds_joint = {'content': group_divergence,
                        'style': dict()};
            elbos = dict();
            for m, m_key in enumerate(mods.keys()):
                mod = mods[m_key];
                if self.flags.factorized_representation:
                    kld_style_m = klds_style[m_key + '_style'];
                else:
                    kld_style_m = 0.0;
                klds_joint['style'][m_key] = kld_style_m;
                if self.flags.poe_unimodal_elbos:
                    i_batch_mod = {m_key: target[m_key]};
                    r_mod = self.model(i_batch_mod);
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                    target[m_key],
                                                    self.flags.batch_size);
                    log_prob = {m_key: log_prob_mod};
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}};
                    elbo_mod = utils.calc_elbo(self.flags, self.modalities, self.rec_weights, self.style_weights, m_key, log_prob, klds_mod);
                    elbos[m_key] = elbo_mod;
            elbo_joint = utils.calc_elbo(self.flags, self.modalities, self.rec_weights, 'joint', log_probs, klds_joint);
            elbos['joint'] = elbo_joint;
            total_loss = sum(elbos.values())
        return total_loss

        
    def forward_recon(self, x: torch.Tensor, mask: torch.Tensor):
        '''
        x: dict(), name: modality name, value: modality data
        mask: (B, M), 1: missing
        '''
        assert self.num_modalities == len(x)
        # currently only support all subjects miss the same modality
        mods = dict()
        for i in range(self.num_modalities):
            if mask[0, i] == 0:
                assert mask[:, i].sum() == 0
                mods[f'm{i}'] = x[f'm{i}']
        results = self.model(mods)   
        loss = self.forward_loss(x, results, mask)  
        return results, loss

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        '''
        x: dict(), name: modality name, value: modality data
        mask: (B, M), 1: missing
        generate reconstructions for all modalities
        '''
        assert self.num_modalities == len(x) 
        # support randomly missing modalities, for loop on all samples
        inferred = self.model.inference(x)   
        latents = inferred['subsets']
        x_recon_all = self.model.cond_generation(latents)
        
        output = dict()
        for i in range(self.num_modalities):    
            output[f'm{i}'] = torch.zeros_like(x[f'm{i}'], device=mask.device)

        for k in range(mask.shape[0]):
            mods_name = []
            for i in range(self.num_modalities):
                if mask[k, i] == 0:
                    # find all non-missing modalities
                    mods_name.append(f'm{i}')
            key = '_'.join(mods_name)
            for i in range(self.num_modalities):
                if mask[k, i] == 0:
                    output[f'm{i}'][k,:] = x[f'm{i}'][k,:]
                else:
                    output[f'm{i}'][k,:] = x_recon_all[key][f'm{i}'][k,:]

        return output
    
    def forward_feature(self, x: torch.Tensor, mask: torch.Tensor):
        assert self.num_modalities == len(x)
        # support randomly missing modalities, for loop on all samples
        inferred = self.model.inference(x)   
        latents = inferred['subsets']
        out = []
        for k in range(mask.shape[0]):
            key = self.subset2name[tuple(mask[k,:].tolist())]
            out.append(latents[key][0][k,:])
        out = torch.stack(out, dim=0)

        return out




if __name__ == "__main__":
    args = {'PolyMNIST':{"transformer_dim": 128, "transformer_heads": 4, "transformer_layers": 2, "transformer_drop": 0.0, "num_patches_list": [1, 1, 1, 1, 1], 
                         "recon_encoder_layers":2, "recon_decoder_layers": 2, "num_masks": 1}, 'dataset_name': 'PolyMNIST',
                "num_modalities": 5, "checkpoint": "/home/siyi/project/mm/result/Dynamic_project/PM23/MoPoE_MMNIST_2025_04_01_17_18_11_421525/checkpoints/0299/mm_vae", 'logdir': None, 'batch_size': 2,
                "downstream_strategy": "frozen", "num_classes": 10}
    args = OmegaConf.create(args)
    model = MoPoE(args)
    x = {'m0': torch.randn(2, 3, 28, 28), 'm1': torch.randn(2, 3, 28, 28), 'm2': torch.randn(2, 3, 28, 28),
         'm3': torch.randn(2, 3, 28, 28), 'm4': torch.randn(2, 3, 28, 28)}
    mask = torch.tensor([[True, False, False, False, False], [False, True, True, False, False]])
    # output = model.forward_recon(x, mask)
    # for mod, dist in output[0]['rec'].items():
    #     sample = dist.sample()
    #     print(mod, sample.shape)
    # print(output[1])
    
    output = model.forward(x, mask)
    for mod, sample in output.items():
        print(mod, sample.shape)
    
    output_feature = model.forward_feature(x, mask)
    print(output_feature.shape)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

