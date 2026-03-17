from typing import Dict
import torch
import torch.nn as nn
import sys
from omegaconf import OmegaConf

sys.path.append('/home/siyi/project/mm/mul_foundation/Dynamic_Missing')
from multivae.models.cmvae import CMVAEConfig
from models.PolyMNIST.Comparison.MVAE_lib.CMVAE_architectures import Enc, Dec
from models.PolyMNIST.Comparison.MVAE_lib.cmvae_model import CMVAE_Batch
from pythae.data.datasets import DatasetOutput




class ReconCMVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.modalities = ["m0", "m1", "m2", "m3", "m4"]
        self.num_modalities = len(self.modalities)

        model_config = CMVAEConfig(
            n_modalities=5,
            K=1,
            decoders_dist={m: "laplace" for m in self.modalities},
            decoder_dist_params={m: dict(scale=0.75) for m in self.modalities},
            prior_and_posterior_dist="laplace_with_softmax",
            beta=2.5,
            modalities_specific_dim=32,
            latent_dim=32,
            input_dims={m: (3, 28, 28) for m in self.modalities},
            learn_modality_prior=True,
            number_of_clusters=40,
            loss="iwae_looser",
        )
        encoders = {
            m: Enc(model_config.modalities_specific_dim, ndim_u=model_config.latent_dim)
            for m in  self.modalities
        }
        decoders = {
            m: Dec(model_config.latent_dim + model_config.modalities_specific_dim)
            for m in self.modalities
        }

        self.model = CMVAE_Batch(model_config, encoders, decoders)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            self.model.load_state_dict(ckpt['model_state_dict'])
            print(f"Load model from {args.checkpoint}")
    
    def forward_feature(self, x: Dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        out = self.model.encode_batch(x=x, mask=mask)['z']
        return out
    

    def forward(self, x: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        '''
        mask = True means missing
        '''
        output = dict()
        for i in range(self.num_modalities):    
            output[f'm{i}'] = torch.zeros_like(x[f'm{i}'], device=mask.device)

        for k in range(mask.shape[0]):
            cond_mods_name = []
            gen_mods_name = []
            cond_mods = {}
            for i, m in enumerate(self.modalities):
                if mask[k, i] == True:
                    gen_mods_name.append(m)
                else:
                    cond_mods_name.append(m)
                    cond_mods[m] = x[m][k].unsqueeze(0)
            gen_mods = self.model.predict(inputs=DatasetOutput(data=cond_mods), cond_mod=cond_mods_name, gen_mod=gen_mods_name)
            for m in gen_mods_name:
                output[m][k] = gen_mods[m].squeeze(0)
            for m in cond_mods_name:
                output[m][k] = cond_mods[m].squeeze(0)

        return output


if __name__ == "__main__":
    args = {'PolyMNIST':{"transformer_dim": 128, "transformer_heads": 4, "transformer_layers": 2, "transformer_drop": 0.0, "num_patches_list": [1, 1, 1, 1, 1], 
                         "recon_encoder_layers":2, "recon_decoder_layers": 2, "num_masks": 1}, 'dataset_name': 'PolyMNIST',
                "num_modalities": 5, "checkpoint": "/home/siyi/project/mm/result/Dynamic_project/PM51/reproduce_cmvae/K__1/CMVAE_training_2025-09-01_16-26-14/final_model/model.pt", 'logdir': None, 'batch_size': 2,
                "downstream_strategy": "frozen", "num_classes": 10}
    args = OmegaConf.create(args)
    model = ReconCMVAE(args)
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
    
    output = model.forward_feature(x, mask)
    print(output.shape)
    
    # output_feature = model.forward_feature(x, mask)
    # print(output_feature.shape)

    # # calculate the number of parameters
    # num_params = sum(p.numel() for p in model.parameters())
    # print(num_params/1e6)
