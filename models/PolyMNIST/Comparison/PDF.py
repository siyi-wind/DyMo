from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
import torch
from omegaconf import OmegaConf
import sys
import copy
from einops import rearrange,repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import time
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)


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


class PDF(nn.Module):
    '''
    Predictive Dynamic Fusion (PDF) model.   https://github.com/Yinan-Xia/PDF/tree/main
    '''
    def __init__(self, args):
        super(PDF, self).__init__()
        self.m_encoders = nn.ModuleDict()
        self.m_ConfidNets = nn.ModuleDict()
        self.m_classifiers = nn.ModuleDict()
        self.embedding_size = args[args.dataset_name].embedding_size
        for i in range(args.num_modalities):
            self.m_encoders[f'm{i}'] = UnimodalEncoder()
            self.m_ConfidNets[f'm{i}'] = nn.Sequential(
                    nn.Linear(self.embedding_size, self.embedding_size*2),
                    nn.Linear(self.embedding_size*2, self.embedding_size),
                    nn.Linear(self.embedding_size, 1),
                    nn.Sigmoid())
            self.m_classifiers[f'm{i}'] = nn.Linear(self.embedding_size, args.num_classes)
        
        self.modality_names = args.modality_names
        self.use_imputer = args[args.dataset_name].use_imputer
        print(f"Use imputer: {self.use_imputer}")
        if self.use_imputer:
            self.create_imputation_network(args)
        
        self.maeloss = nn.L1Loss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss()
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
        return out, feat, start_time


    def forward_train(self, x: Dict, mask: torch.Tensor, y: torch.Tensor):
        m_out, m_f, _ = self.forward_encoder_classifier(x, mask)
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


    def forward(self, x: Dict, mask: torch.Tensor, visualize=False):
        m_out, m_f, start_time = self.forward_encoder_classifier(x, mask)
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
        end_time = time.time()
        # report inference latency using ms
        inference_time = (end_time - start_time)*1000
        print(f'Inference time for batch size {B}: {inference_time:.4f} ms, per sample: {inference_time/B:.4f} ms')
        if visualize:
            return out, w_all
        else:
            return out



if __name__ == "__main__":
    args = {'PolyMNIST':{"embedding_size": 512, "use_imputer": False,
                         'imputer_name': 'MoPoE', 'imputer_checkpoint': '/home/siyi/project/mm/result/Dynamic_project/PM23/MoPoE_MMNIST_2025_04_01_17_18_11_421525/checkpoints/0299/mm_vae'}, 
                         'dataset_name': 'PolyMNIST', 'modality_names': ['m0', 'm1', 'm2', 'm3', 'm4'],
                "num_modalities": 5, "checkpoint": None, "num_classes": 10, 'batch_size':2, 'logdir':None,}
    args = OmegaConf.create(args)
    model = PDF(args)
    x = {'m0': torch.randn(2, 3, 28, 28), 'm1': torch.randn(2, 3, 28, 28), 'm2': torch.randn(2, 3, 28, 28),
         'm3': torch.randn(2, 3, 28, 28), 'm4': torch.randn(2, 3, 28, 28)}
    mask = torch.tensor([[False, False, False, False, False], [False, False, False, False, False]])
    loss, output = model.forward_train(x, mask, y=torch.tensor([0, 1]))
    print(loss)
    print(output.shape)

    args[args.dataset_name].use_imputer = True
    model = PDF(args)
    mask = torch.tensor([[True, True, False, False, False], [True, True, False, False, False]])
    output, weight = model.forward(x, mask, visualize=True)
    print(output.shape)
    print(weight.shape)

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)

        

        
