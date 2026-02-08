import torch
import torch.nn as nn
import sys
from omegaconf import OmegaConf
from PIL import ImageFont
import json

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../'))
sys.path.append(project_path)
from models.utils.MVAE.BaseMMVae import BaseMMVae
from models.utils.MVAE.Modality import CelebAImg, CelebAText
from itertools import chain, combinations
from models.utils.MVAE.divergence_measures.kl_div import calc_kl_divergence
from models.utils.MVAE import utils
from models.CelebA.networks.MoPoE_ConvNetworksImgCelebA import EncoderImg, DecoderImg
from models.CelebA.networks.MoPoE_ConvNetworksTextCelebA import EncoderText, DecoderText



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
                 'alpha_modalities': [0.3, 0.35, 0.35], 'batch_size': args.batch_size,
                 'device': None, 'class_dim': 32, 'factorized_representation': True,
                 'dir_checkpoints':args.logdir, 'beta_style': 2.0, 'beta_content': 1.0, 'beta': 2.5, 
                 'beta_m1_style': 1.0, 'beta_m2_style': 5.0,  'num_features': 71,
                 # CelebA
                 'num_layers_img': 5, 'num_layers_text': 7, 'DIM_img': 128, 'DIM_text': 128, 'image_channels': 3,
                 'style_img_dim': 32, 'style_text_dim': 32}
        self.alphabet = args.alphabet
        self.plot_img_size = torch.Size((3, 64, 64))
        self.font = ImageFont.truetype('FreeSerif.ttf', 38)
        self.flags = OmegaConf.create(flags)
        self.len_sequence = args.len_sequence
        self.modalities = self.set_modalities()
        self.subsets = self.set_subsets()
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()
        self.model = BaseMMVae(self.flags, self.modalities, self.subsets)
        self.alphabet = args.alphabet
        self.modality_names = args.modality_names   
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            self.model.load_state_dict(ckpt)
            print(f"Load model from {args.checkpoint}")

    def set_modalities(self,):
        mod1 = CelebAImg(EncoderImg(self.flags), DecoderImg(self.flags), self.plot_img_size, style_dim=32)
        mod2 = CelebAText(EncoderText(self.flags), DecoderText(self.flags), self.len_sequence, self.alphabet, self.plot_img_size, self.font, style_dim=32)
        mods_dict = {m.name: m for m in [mod1, mod2]}
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
        mod_names_dict = {'img': 0, 'text': 1}
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
        rec_weights = dict();
        ref_mod_d_size = self.modalities['img'].data_size.numel()/3;
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key];
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size/numel_mod)
        return rec_weights;
    
    def set_style_weights(self):
        weights = dict();
        weights['img'] = self.flags.beta_m1_style;
        weights['text'] = self.flags.beta_m2_style;
        return weights;

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
        for i, name in enumerate(self.modality_names):
            if mask[0, i] == 0:
                assert mask[:, i].sum() == 0
                mods[name] = x[name]
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
        num_samples = mask.shape[0]
        inferred = self.model.inference(x)   
        latents = inferred['subsets']
        x_recon_all = self.model.cond_generation(latents, num_samples)
        
        output = dict()
        for i, name in enumerate(self.modality_names):    
            output[name] = torch.zeros_like(x[name], device=mask.device)

        for k in range(mask.shape[0]):
            mods_name = []
            for i, name in enumerate(self.modality_names):
                if mask[k, i] == 0:
                    # find all non-missing modalities
                    mods_name.append(name)
            key = '_'.join(mods_name)
            for i, name in enumerate(self.modality_names):
                if mask[k, i] == 0:
                    output[name][k,:] = x[name][k,:]
                else:
                    output[name][k,:] = x_recon_all[key][name][k,:]

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
    alphabet_path = join(project_path, 'datasets/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = {'CelebA':{"transformer_dim": 64, "transformer_heads": 2, "transformer_layers": 2, "transformer_drop": 0.0, "num_patches_list": [1, 1], 
                         "num_masks": 0}, 'dataset_name': 'CelebA',
                "num_modalities": 2, "checkpoint": '/home/siyi/project/mm/result/Dynamic_project/CA11/CelebA/joint_elbo/factorized/laplace_categorical/CelebA_2025_05_15_18_15_43_805671/checkpoints/0199/mm_vae',
                  'logdir': None, 'batch_size': 2,
                "downstream_strategy": "frozen", 'alphabet': alphabet, 'modality_names': ['img', 'text'], "len_sequence": 256, }

    args = OmegaConf.create(args)
    model = MoPoE(args)

    indices = torch.randint(0, 71, (2, 256))
    one_hot_tensor = torch.nn.functional.one_hot(indices, num_classes=71).float()
    x = {'img': torch.rand(2, 3, 64, 64), 'text': one_hot_tensor,}
    mask = torch.tensor([[True, False], [False, True]])
    
    output = model.forward(x, mask)
    for mod, sample in output.items():
        print(mod, sample.shape)

    output_feature = model.forward_feature(x, mask)
    print(output_feature.shape)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

