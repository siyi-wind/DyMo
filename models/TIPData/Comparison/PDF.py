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
from models.utils.transformer import Block
from models.utils.TIP_utils.Transformer import TabularTransformerEncoder
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder


class PDF(nn.Module):
    '''
    Predictive Dynamic Fusion (PDF) model.   https://github.com/Yinan-Xia/PDF/tree/main
    '''
    def __init__(self, args):
        super(PDF, self).__init__()
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
        self.m_ConfidNets = nn.ModuleDict()
        self.m_classifiers = nn.ModuleDict()
        
        for i, name in enumerate(self.modality_names):
            self.m_ConfidNets[name] = nn.Sequential(
                    nn.Linear(self.embedding_size, self.embedding_size*2),
                    nn.Linear(self.embedding_size*2, self.embedding_size),
                    nn.Linear(self.embedding_size, 1),
                    nn.Sigmoid())
            self.m_classifiers[name] = nn.Linear(self.embedding_size, args.num_classes)
        
        self.use_imputer = args[args.dataset_name].use_imputer
        print(f"Use imputer: {self.use_imputer}")
        self.create_imputation_network(args)
        
        self.maeloss = nn.L1Loss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss()
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
                    # tabular_missing_mask = torch.fill_(torch.empty(x['tabular'].shape[0], x['tabular'].shape[1]), mask[0,1]).bool().to(mask.device)
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


    def forward_train(self, x: Dict, mask: torch.Tensor, y: torch.Tensor):
        m_out, m_f = self.forward_encoder_classifier(x, mask)
        tcp = []
        for i, name in enumerate(self.modality_names):
            f_cp_i = m_f[:,i,:].clone().detach()
            tcp_i = self.m_ConfidNets[name](f_cp_i)  # (B, 1)
            tcp.append(tcp_i)
        tcp = torch.cat(tcp, dim=1)  # (B, M)
        # calculate holo
        B, M = tcp.shape
        tcp_mat = repeat(tcp, 'b m -> b k m', k=M)  # (B, M, M)
        # holo numerator is the sum of all log conf expect the current modality,
        # denominator is the sum of all log conf
        eye_mask = torch.eye(M, device=tcp.device).bool().unsqueeze(0)  # (1, M, M)
        holo_numerator = torch.log(tcp_mat + 1e-8) * (~eye_mask).float()  # (B, M, M)
        holo_numerator = holo_numerator.sum(dim=2)  # (B, M)
        holo_denominator = torch.log(tcp_mat + 1e-8).sum(dim=2)  # (B, M)
        holo = holo_numerator / (holo_denominator + 1e-8)  # (B, M)
        w_all = tcp.detach() + holo.detach()  # (B, M)
        w_all = torch.softmax(w_all, dim=1)  # (B, M)
        out = m_out * (w_all.unsqueeze(2).detach())  # (B, M, num_classes)
        out = out.sum(dim=1)  # (B, num_classes)

        # calculate the loss
        label = F.one_hot(y.long(), num_classes=self.num_classes)
        tcp_pred_loss = []
        clf_loss = []
        for i, name in enumerate(self.modality_names):
            tmp = m_out[:, i, :]
            tmp = torch.softmax(tmp, dim=1)
            tcp_gt_i, _ = torch.max(tmp * label, dim=1, keepdim=True)
            tcp_pred_loss.append(self.maeloss(tcp[:, i].unsqueeze(1), tcp_gt_i.detach()))
            clf_loss.append(self.ce_loss(m_out[:, i, :], y))

        tcp_pred_loss = sum(tcp_pred_loss) 
        clf_loss = sum(clf_loss)
        joint_clf_loss = self.ce_loss(out, y)
        loss = tcp_pred_loss + clf_loss + joint_clf_loss  # (B)
        return (loss, out)


    def forward(self, x: Dict, mask: torch.Tensor):
        m_out, m_f = self.forward_encoder_classifier(x, mask)
        tcp = []
        for i, name in enumerate(self.modality_names):
            f_cp_i = m_f[:,i,:].clone().detach()
            tcp_i = self.m_ConfidNets[name](f_cp_i)  # (B, 1)
            tcp.append(tcp_i)
        tcp = torch.cat(tcp, dim=1)  # (B, M)
        # calculate holo
        B, M = tcp.shape
        tcp_mat = repeat(tcp, 'b m -> b k m', k=M)  # (B, M, M)
        # holo numerator is the sum of all log conf expect the current modality,
        # denominator is the sum of all log conf
        eye_mask = torch.eye(M, device=tcp.device).bool().unsqueeze(0)  # (1, M, M)
        holo_numerator = torch.log(tcp_mat + 1e-8) * (~eye_mask).float()  # (B, M, M)
        holo_numerator = holo_numerator.sum(dim=2)  # (B, M)
        holo_denominator = torch.log(tcp_mat + 1e-8).sum(dim=2)  # (B, M)
        holo = holo_numerator / (holo_denominator + 1e-8)  # (B, M)
        cb = tcp.detach() + holo.detach() # (B, M)

        # calculate distribution uniformity
        du = []
        for i, name in enumerate(self.modality_names):
            pred_i = torch.softmax(m_out[:, i, :], dim=1)  # (B, num_classes)
            du_i = torch.mean(torch.abs(pred_i - 1 / pred_i.shape[1]), dim=1, keepdim=True)  # (B, 1)
            du.append(du_i)
        du = torch.cat(du, dim=1) # (B, M)
        du_mat = repeat(du, 'b m -> b k m', k=M)  # (B, M, M)
        rc_numerator = du * (M-1)   # (B, M)
        rc_denominator = du_mat * (~eye_mask).float()  # (B, M, M)
        rc_denominator = rc_denominator.sum(dim=2)  # (B, M)
        rc = rc_numerator / (rc_denominator + 1e-8)  # (B, M)
        # replace rc>1 with 1
        rc = torch.where(rc > 1, torch.ones_like(rc), rc)  # (B, M)
        assert rc.max() <= 1, 'rc should be less than or equal to 1'

        ccb = cb * rc  # (B, M)
        w_all = torch.softmax(ccb, dim=1)
        out = m_out * (w_all.unsqueeze(2).detach())
        out = out.sum(dim=1)
        return out



if __name__ == "__main__":
    alphabet_path = '/home/siyi/project/mm/mul_foundation/MoPoE/alphabet.json'
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = {'DVM':{"embedding_size": 256, "use_imputer": False,
                   'imputer_name': 'TIP', 'imputer_checkpoint': '/vol/biomedic3/sd1523/project/mm/result/TIP_results/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt',}, 
                         'dataset_name': 'DVM', 'alphabet': alphabet, 'modality_names': ['img', 'tabular'], 'DATA_field_lengths_tabular': '/bigdata/siyi/data/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                "num_modalities": 2, "checkpoint": None, "num_classes": 286, 'batch_size':2,'missing_tabular': False}
    args = OmegaConf.create(args)
    model = PDF(args)
    x = {'img': torch.randn(2, 3, 128, 128), 'tabular': torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                    [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)}
    mask = torch.tensor([[False, False], [False, False]])
    loss, output = model.forward_train(x, mask, y=torch.tensor([0, 1]))
    print(loss)
    print(output.shape)

    args['DVM'].use_imputer = True
    model = PDF(args)
    mask = torch.tensor([[False, True], [False, True]])
    output = model.forward(x, mask)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

        

        
