from typing import Dict
import copy
import torch.nn as nn
import torch
from omegaconf import OmegaConf
import sys
from einops import rearrange,repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os
import json
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.utils.quality_prototype import check_update_effect

class DyMo(nn.Module):
    '''
    Have a frozen imputation network to generate the missing modalities.
    Input is a dictionary of modalities and a mask indication matrix.
    Encode all modalities through modality-specific CNN encoders.
    Use the mask to select existing modalities and pass them through a transformer.
    Still input masked tokens into the transformer, use attention mask to avoid attending to masked tokens.
    Special embeddings: cls token (1,dim), modality embeddings (num_modalities, dim), intra-modality positional embeddings (1+sum_num_patches, dim)
    '''
    def __init__(self, args):
        super(DyMo, self).__init__()
        print('DyMo.py')
        assert args[args.dataset_name].transformer_checkpoint is not None, 'Please provide the transformer checkpoint'
        from models.MST.Dynamic.DynamicTransformer import DynamicTransformer
        self.model = DynamicTransformer(args)
        self.quality_metric_name = args[args.dataset_name].quality_metric_name
        self.num_modalities = args.num_modalities
        transformer_folder = os.path.dirname(os.path.dirname(args[args.dataset_name].transformer_checkpoint))

        self.distance_metric = args[args.dataset_name].distance_metric
        if self.distance_metric == 'cosine_similarity':
            from models.utils.quality_prototype import calculate_probabilities_COS, PrototypeDistance_COS
            self.quality_metric = PrototypeDistance_COS(temperature=0.1)
            self.subset_gaussian = torch.load(join(transformer_folder, 'gaussian/subset_gaussian.pt'))
            self.calculate_probabilities = calculate_probabilities_COS
        elif self.distance_metric == 'squared_euclidean':
            from models.utils.quality_prototype import calculate_probabilities_EU, PrototypeDistance_EU
            self.quality_metric = PrototypeDistance_EU(temperature=0.1)
            self.subset_gaussian = torch.load(join(transformer_folder, 'gaussian/subset_gaussian_EU.pt'))
            self.calculate_probabilities = calculate_probabilities_EU
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        print(f"Using distance metric for DyMo: {self.distance_metric}")

        self.create_imputation_network(args)
    
    
    def create_imputation_network(self, args):
        self.imputer_name = args[args.dataset_name].imputer_name
        if self.imputer_name == 'MoPoE':
            args_imputer = copy.deepcopy(args)
            args_imputer.checkpoint = args[args.dataset_name].imputer_checkpoint
            from models.MST.MoPoE import MoPoE
            self.imputer = MoPoE(args_imputer)
            for param in self.imputer.parameters():
                param.requires_grad = False
        elif self.imputer_name == 'SimpleImputer':
            from models.utils.simple_imputer import SimpleImputer
            self.imputer_p = args[args.dataset_name].imputer_p
            self.imputer = SimpleImputer(modality_names=args.modality_names, p=self.imputer_p, fill_value=0.0)
            print(f'Using Simple Imputer with fill_value 0.0 and p={self.imputer_p}')
        else:
            raise ValueError(f"Unsupported imputer name: {self.imputer_name}")
        print(f"Imputer {args[args.dataset_name].imputer_name} frozen")


    def calculate_reward(self, u: int, x: Dict, mask: torch.Tensor, loc: torch.Tensor, y: torch.Tensor):
        '''
        Calculate the reward for combining i-th candidate recovered modality
        reward = Metric(w/o i-th modality) - Metric(w/ i-th modality)
        This metric is called multimodal quality score, we can use different algorithms like prototype distance
        u: id of candidate modality
        x: input data
        mask: input mask   (B,M).   True means masked
        loc: the locations of masked and not bad modalities  (B).   True means need to process
        '''
        # prototypes = self.models.prototypes.clone().detach()  # (num_subsets, num_classes, dim)
        prototypes = self.subset_gaussian['overall_prototypes'].to(mask.device)  # (num_subsets, num_classes, dim)
        # repeat to have (len(loc), num_subsets, num_classes, dim)
        # prototypes_all = prototypes_all.unsqueeze(0).repeat(loc.sum(), 1, 1, 1)  # (B, num_subsets, num_classes, dim)
        # get the input data that need to process using loc
        x_temp = {k: v[loc] for k, v in x.items()}
        y = y[loc]
        B = len(y)
        mask_temp = mask[loc]
        if self.quality_metric_name == 'prototype_distance':
            mask_before = mask_temp.clone().detach()
            y_hat, feat, subsets_ids = self.model.forward_train(x_temp, mask=mask_temp, return_subsets_ids=True)
            pred = torch.softmax(y_hat.detach(), dim=-1)
            max_prob, max_id = torch.max(pred, dim=1)
            quality, max_id_prot, pred_prot = self.quality_metric(feat, prototypes)
            gaussian_probs, _ = self.calculate_probabilities(feat, mask_temp, max_id, self.subset_gaussian)

            # make u-th modality non-masked
            mask_temp[:, u] = False
            mask_update = mask_temp.clone().detach()
            y_hat_u, feat_u, subsets_ids_u = self.model.forward_train(x_temp, mask=mask_temp, return_subsets_ids=True)
            pred_u = torch.softmax(y_hat_u.detach(), dim=-1)
            max_prob_u, max_id_u = torch.max(pred_u, dim=1)
            quality_u, max_id_prot_u, pred_prot_u = self.quality_metric(feat_u, prototypes)
            gaussian_probs_u, _ = self.calculate_probabilities(feat_u, mask_temp, max_id_u, self.subset_gaussian)
            # calibration matrix  C (B), C=conf_u/conf if conf>conf_u else 1
            G = torch.ones_like(gaussian_probs).float().to(mask.device)
            # TODO different distance metric may need different G calculation
            if self.distance_metric == 'cosine_similarity':
                G = torch.where(gaussian_probs_u < gaussian_probs, (torch.log(gaussian_probs)/torch.log(gaussian_probs_u+1e-9)), G)
            elif self.distance_metric == 'squared_euclidean':
                G = torch.where(gaussian_probs_u < gaussian_probs, (gaussian_probs_u / (gaussian_probs + 1e-9)), G)
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
            R = (quality_u*G - quality)
        
        return R
            

    def forward(self, x: Dict, mask: torch.Tensor, y: torch.Tensor, visualize: bool = False):
        mask = mask.bool()
        with torch.no_grad():
            x = self.imputer(x, mask)
            recon = copy.deepcopy(x)  # save the original imputation result
        
        origin_mask = copy.deepcopy(mask)
        B, M = mask.shape
        # get the iteration steps T: max(mask.sum(dim=1))
        T = mask.sum(dim=1).max().int().item()
        # history recording
        R_history = torch.zeros(T, B, M).to(mask.device)
        mask_history = torch.zeros((T+1), B, M).bool().to(mask.device)   # True means masked
        action_history = torch.full((T, B), 100).to(mask.device)  # 100 means no action
        # avoid updating some modalities
        finished_sample_indicator = torch.zeros(B, dtype=bool).to(mask.device)  # True means finished
        bad_modality_indicator = torch.zeros(B, M, dtype=bool).to(mask.device)  # True means bad modality
        mask_history[0, ...] = mask.clone().detach()  # record the initial mask
        
        for t in range(T):
            # initialize the reward matrix
            R = torch.zeros(B, M).to(mask.device)
            for u in range(self.num_modalities):
                # print(f'Processing modality {u} at step {t} ==========================')
                # get the locations of masked and not bad modalities
                loc = (mask[:, u] == True) & (bad_modality_indicator[:, u] == False)   # (B)
                if loc.sum() == 0:
                    continue
                R[loc, u] = self.calculate_reward(u, x, mask, loc, y)
            
            # update bad modality indicator, R<0 locations are bad
            bad_modality_indicator[R < 0] = True
            # update finished_sample_indicator, a sample finished means all modalities are either bad (R<0) or no-use (R=0) or non-missing (R=0)
            finished_t = (R <= 0).all(dim=1)
            finished_sample_indicator[finished_t] = True
            # get optimal action
            i_optimal = R.argmax(dim=1)
            # update the action history, only for unfinished samples
            action_history[t, ~finished_sample_indicator] = i_optimal[~finished_sample_indicator].clone().detach()
            # update the mask, only for unfinished samples
            mask[~finished_sample_indicator, i_optimal[~finished_sample_indicator]] = False
            # record reward history
            R_history[t, ...] = R.clone().detach()
            mask_history[t+1, ...] = mask.clone().detach()


            if finished_sample_indicator.all():
                break
        
        # record the final optimal mask
        out = self.model(x, mask=mask)

        if visualize:
            return out, {'R_history': R_history,  'mask_history': mask_history, 'action_history': action_history, 'mask_final': mask, 'recon': recon}
        else:
            return out


if __name__ == "__main__":
    alphabet_path = join(project_path, 'datasets/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = {'MST':{"transformer_dim": 32, "transformer_heads": 2, "transformer_layers": 2, "transformer_drop": 0.0, 
                         "num_patches_list": [1,1,1], "num_masks": 0, 'imputer_name': 'MoPoE', "distance_metric": "cosine_similarity",
                         'imputer_checkpoint': '/home/siyi/project/mm/result/Dynamic_project/MS6/MNIST_SVHN_Text/joint_elbo/non_factorized/laplace_laplace_categorical/SVHN_MNIST_text_2025_05_05_14_13_16_648881/checkpoints/0199/mm_vae',
                         'transformer_name': 'DynamicTransformer_singleCLS',
                         'transformer_checkpoint': '/home/siyi/project/mm/result/Dynamic_project/MS4/none_MST_DynamicTransformer_singleCLS_0507_145754/downstream/checkpoint_best_acc.ckpt',
                         'quality_metric_name': 'prototype_distance', 'projection_dim': 16}, 
                         'checkpoint': None, 'num_classes': 10,
                         'dataset_name': 'MST', 'batch_size':2, 'logdir':None,
                "num_modalities": 3, 'alphabet': alphabet, 'modality_names': ['mnist', 'svhn', 'text'],'len_sequence': 8}
    args = OmegaConf.create(args)
    model = DyMo(args)
    indices = torch.randint(0, 71, (2, 8))
    one_hot_tensor = torch.nn.functional.one_hot(indices, num_classes=71).float()
    x = {'mnist': torch.rand(2, 1, 28, 28), 'svhn': torch.rand(2, 3, 32, 32), 'text': one_hot_tensor,}
    mask = torch.tensor([[True, True, False], [True, True, False]])
    y = torch.tensor([1, 2])
    output, records = model.forward(x, mask, y, visualize=True)
    print(torch.max(torch.softmax(output, dim=-1), dim=1))

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)


        

        
