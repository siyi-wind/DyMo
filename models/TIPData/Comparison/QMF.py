from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
import torch
from omegaconf import OmegaConf
import sys
import json
from einops import rearrange,repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.utils.QMF_utils.utils import rank_loss, create_history
from models.utils.TIP_utils.Transformer import TabularTransformerEncoder
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder


class QMF(nn.Module):
    '''
    Quality-aware Multimodal Fusion (QMF).   https://github.com/QingyangZhang/QMF/blob/main/text-image-classification/src/models/late_fusion.py
    '''
    def __init__(self, args):
        super(QMF, self).__init__()
        self.modality_names = args.modality_names
        self.field_lengths_tabular = torch.load(args.DATA_field_lengths_tabular)
        self.cat_lengths_tabular = []
        self.con_lengths_tabular = []
        self.embedding_size = args[args.dataset_name].embedding_size
        for x in self.field_lengths_tabular:
            if x == 1:
                self.con_lengths_tabular.append(x) 
            else:
                self.cat_lengths_tabular.append(x)
        flags = OmegaConf.create({'tabular_embedding_dim': 512, 'embedding_dropout': 0.0,
                                    'tabular_transformer_num_layers': 4, 'multimodal_transformer_num_layers': 4,        
                                'multimodal_embedding_dim': 512, 'drop_rate': 0.0, 'checkpoint': None})
        self.m_encoders = nn.ModuleDict({
                'img': torchvision_ssl_encoder('resnet50', pretrained=True, return_all_feature_maps=True),
                'tabular': TabularTransformerEncoder(flags, self.cat_lengths_tabular, self.con_lengths_tabular)
                })
        self.m_mappers = nn.ModuleDict({
                'img': nn.Linear(2048, self.embedding_size),
                'tabular': nn.Linear(512, self.embedding_size)
                }) 
        self.m_classifiers = nn.ModuleDict()
        for i, name in enumerate(self.modality_names):
            self.m_classifiers[name] = nn.Linear(self.embedding_size, args.num_classes)
        

        self.use_imputer = args[args.dataset_name].use_imputer
        print(f"Use imputer: {self.use_imputer}")
        if self.use_imputer:
            self.create_imputation_network(args)
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss_no_reduce = nn.CrossEntropyLoss(reduction='none')
        self.num_classes = args.num_classes

    def create_imputation_network(self, args):
        self.imputer_name = args[args.dataset_name].imputer_name
        if self.imputer_name  == 'TIP':
            args_imputer = args
            args_imputer.checkpoint = args[args.dataset_name].imputer_checkpoint
            args_imputer.field_lengths_tabular = args.DATA_field_lengths_tabular
            from models.utils.TIP_utils.BaseReconTIP import ReconTIP
            self.imputer = ReconTIP(args_imputer)
            for param in self.imputer.parameters():
                param.requires_grad = False
        elif self.imputer_name  == 'ML':
            # Input has already been imputed through machine learning methods
            self.imputer = None  
        else:
            raise ValueError(f"Unsupported imputer name: {self.imputer_name}")
        print(f"Imputer {self.imputer_name } frozen")


    def forward_encoder_classifier(self, x: Dict, mask: torch.Tensor):
        mask = mask.bool()  
        original_mask = mask.clone()  # save the original mask for imputation
        if self.use_imputer:
            assert mask[0].sum() > 0, 'Mask should have at least one missing modality'
            assert not torch.any(mask[:, 0]), f'TIP does not support missing imaging features, but got {mask[:, 0].sum()} missing values in imaging features.'
            with torch.no_grad():
                if 'tabular_missing_mask' in x:
                    tabular_missing_mask = x['tabular_missing_mask']
                else:
                    tabular_missing_mask = torch.full_like(x['tabular'], fill_value=mask[0, 1], dtype=torch.bool)
                if self.imputer_name == 'TIP':
                    x = self.imputer(x, tabular_missing_mask)
                elif self.imputer_name == 'ML':
                    pass
            mask = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)  # after imputation, no missing modalities

        # encode all modalities
        feat = []
        out = []
        for i, name in enumerate(self.modality_names):
            tmp = self.m_encoders[name](x[name]) 
            if name == 'img':
                tmp = F.adaptive_avg_pool2d(tmp[-1], (1, 1)).flatten(1)
            elif name == 'tabular':
                tmp = tmp[:, 0]
            tmp = self.m_mappers[name](tmp)  # (B, C)
            mask_i = mask[:, i].unsqueeze(1)  # (B, 1)
            tmp = tmp.masked_fill(mask_i, 1.0)  
            feat.append(tmp)  
            tmp = self.m_classifiers[name](tmp)  # (B, num_classes)
            out.append(tmp)  # (B, num_classes)
        out = torch.stack(out, dim=1)  # (B, M, num_classes)
        feat = torch.stack(feat, dim=1)  # (B, M, C)
        return out, feat


    def forward_train(self, x: Dict, mask: torch.Tensor, y: torch.Tensor, m_history: Dict = None):
        idx = x['index'].squeeze(1)  # (B)
        # remove 'index' from x
        x = {k: v for k, v in x.items() if k != 'index'}
        m_out, _ = self.forward_encoder_classifier(x, mask)
        conf = []
        for i, name in enumerate(self.modality_names):
            out_i = m_out[:, i, :]  # (B, num_classes)
            energy_i = torch.log(torch.sum(torch.exp(out_i), dim=1))
            conf_i = energy_i / 10
            conf_i = torch.reshape(conf_i, (-1, 1))  
            conf.append(conf_i)
        conf = torch.stack(conf, dim=1)  # (B, M, 1)
        out = m_out * conf.detach()  # (B, M, num_classes)
        out = torch.sum(out, dim=1)

        ###### calculate loss ######
        # CE loss
        clf_loss = []
        crl_loss = []
        for i, name in enumerate(self.modality_names):
            conf_i = conf[:, i]  # (B, 1)
            clf_loss.append(self.ce_loss(m_out[:, i, :], y))
            loss_detach = self.ce_loss_no_reduce(m_out[:, i, :], y).detach()
            m_history[name].correctness_update(idx, loss_detach, conf_i.squeeze())  # update correctness for each modality
            crl_loss_i =  rank_loss(conf_i, idx, m_history[name])
            crl_loss.append(crl_loss_i)

        clf_loss = sum(clf_loss)
        crl_loss = sum(crl_loss)
        joint_clf_loss = self.ce_loss(out, y)
        loss = clf_loss + crl_loss + joint_clf_loss

        return (loss, out)


    def forward(self, x: Dict, mask: torch.Tensor):
        if 'index' in x:
            idx = x['index']
            # remove 'index' from x
            x = {k: v for k, v in x.items() if k != 'index'}
        m_out, _ = self.forward_encoder_classifier(x, mask)
        conf = []
        for i, name in enumerate(self.modality_names):
            out_i = m_out[:, i, :]  # (B, num_classes)
            energy_i = torch.log(torch.sum(torch.exp(out_i), dim=1))
            conf_i = energy_i / 10
            conf_i = torch.reshape(conf_i, (-1, 1))  
            conf.append(conf_i)
        conf = torch.stack(conf, dim=1)  # (B, M, 1)
        out = m_out * conf.detach()  # (B, M, num_classes)
        out = torch.sum(out, dim=1)
        return out


if __name__ == "__main__":
    args = {'DVM':{"embedding_size": 32, "use_imputer": False,
                         'imputer_name': 'TIP', 'imputer_checkpoint': '/vol/biomedic3/sd1523/project/mm/result/TIP_results/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt'}, 
                         'dataset_name': 'DVM', 'modality_names': ['img', 'tabular'], 'DATA_field_lengths_tabular': '/bigdata/siyi/data/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                "num_modalities": 2, "checkpoint": None, "num_classes": 283, 'batch_size':2, 'missing_tabular': False}
    args = OmegaConf.create(args)
    model = QMF(args)
    x = {'img': torch.randn(2, 3, 128, 128), 'tabular': torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                    [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32), 
                    'index': torch.tensor([[0], [1]])}
    mask = torch.tensor([[False, False], [False, False]])
    m_history = create_history(args.modality_names, 40)
    loss, output = model.forward_train(x, mask, y=torch.tensor([0, 1]), m_history=m_history)
    print(loss)
    print(output.shape)
    args[args.dataset_name].use_imputer = True
    model = QMF(args)
    mask = torch.tensor([[False, True], [False, False]])
    output = model.forward(x, mask)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

        

        
