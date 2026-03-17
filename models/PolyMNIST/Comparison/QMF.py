from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
import torch
from omegaconf import OmegaConf
import sys
import time
import copy
from einops import rearrange,repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.utils.QMF_utils.utils import rank_loss, create_history


class UnimodalEncoder(nn.Module):
    def __init__(self,):
        super(UnimodalEncoder, self).__init__()
        self.network  = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),     # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # -> (128, 4, 4)
            nn.ReLU(),     
            nn.Flatten(),   
            nn.Linear(2048, 512),       # modality-shared space                 
        )
    
    def forward(self, x: torch.Tensor):
        x = self.network(x)
        return x


class QMF(nn.Module):
    '''
    Quality-aware Multimodal Fusion (QMF).   https://github.com/QingyangZhang/QMF/blob/main/text-image-classification/src/models/late_fusion.py
    '''
    def __init__(self, args):
        super(QMF, self).__init__()
        self.m_encoders = nn.ModuleDict()
        self.m_classifiers = nn.ModuleDict()
        self.embedding_size = args[args.dataset_name].embedding_size
        for i in range(args.num_modalities):
            self.m_encoders[f'm{i}'] = UnimodalEncoder()
            self.m_classifiers[f'm{i}'] = nn.Linear(self.embedding_size, args.num_classes)
        
        self.modality_names = args.modality_names
        self.use_imputer = args[args.dataset_name].use_imputer
        print(f"Use imputer: {self.use_imputer}")
        if self.use_imputer:
            self.create_imputation_network(args)
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss_no_reduce = nn.CrossEntropyLoss(reduction='none')
        self.num_classes = args.num_classes

    def create_imputation_network(self, args):
        assert args[args.dataset_name].imputer_name == 'MoPoE', 'Only MoPoE is supported'
        args_imputer = copy.deepcopy(args)
        args_imputer.checkpoint = args[args.dataset_name].imputer_checkpoint
        from models.PolyMNIST.MoPoE import MoPoE
        self.imputer = MoPoE(args_imputer)
        for param in self.imputer.parameters():
            param.requires_grad = False
        print(f"Imputer {args[args.dataset_name].imputer_name} frozen")


    def forward_encoder_classifier(self, x: Dict, mask: torch.Tensor):
        original_mask = mask.clone()  # save the original mask for imputation
        if self.use_imputer:
            assert mask[0].sum() > 0, 'Mask should have at least one missing modality'
            with torch.no_grad():
                x = self.imputer(x, mask)
            mask = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)  # after imputation, no missing modalities

        start_time = time.time()
        # encode all modalities
        feat = []
        out = []
        for i, name in enumerate(self.modality_names):
            tmp = self.m_encoders[name](x[name]) # (B, C)
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


    def forward(self, x: Dict, mask: torch.Tensor, visualize=False):
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
        if visualize: 
            return out, conf.squeeze(-1)
        else:
            return out


if __name__ == "__main__":
    args = {'PolyMNIST':{"embedding_size": 512, "use_imputer": False,
                         'imputer_name': 'MoPoE', 'imputer_checkpoint': '/home/siyi/project/mm/result/Dynamic_project/PM23/MoPoE_MMNIST_2025_04_01_17_18_11_421525/checkpoints/0299/mm_vae'}, 
                         'dataset_name': 'PolyMNIST', 'modality_names': ['m0', 'm1', 'm2', 'm3', 'm4'],
                "num_modalities": 5, "checkpoint": None, "num_classes": 10, 'batch_size':2, 'logdir':None,}
    args = OmegaConf.create(args)
    model = QMF(args)
    x = {'m0': torch.randn(2, 3, 28, 28), 'm1': torch.randn(2, 3, 28, 28), 'm2': torch.randn(2, 3, 28, 28),
         'm3': torch.randn(2, 3, 28, 28), 'm4': torch.randn(2, 3, 28, 28),'index': torch.tensor([[0], [2]], dtype=torch.long)}
    mask = torch.tensor([[False, False, False, False, False], [False, False, False, False, False]])
    m_history = create_history(args.modality_names, 40)
    loss, output = model.forward_train(x, mask, y=torch.tensor([0, 1]), m_history=m_history)
    print(loss)
    print(output.shape)
    args[args.dataset_name].use_imputer = True
    model = QMF(args)
    mask = torch.tensor([[True, True, False, False, False], [True, True, False, False, False]])
    output, weight = model.forward(x, mask, visualize=True)
    print(output.shape)
    print(weight.shape)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

        

        
