import torch
import torch.nn as nn
import sys
from omegaconf import OmegaConf
from itertools import chain, combinations
import json

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.CelebA.networks.ConvNetworksImgCelebA import EncoderImg, DecoderImg
from models.CelebA.networks.ConvNetworksTextCelebA import EncoderText, DecoderText



def powerset(iterable):
    "powerset([1,2,3]) --> (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class MultiAE(nn.Module):
    '''
    Multimodal AutoEncoder'''
    def __init__(self, args):
        super(MultiAE, self).__init__()
        self.num_modalities = args.num_modalities
        num_text_features = len(args.alphabet)
        self.m_encoders = nn.ModuleDict({
                'img': EncoderImg(None),
                'text': EncoderText(None),
                })
        self.m_decoders = nn.ModuleDict({
                'img': DecoderImg(None),
                'text': DecoderText(None),
                })

        self.create_subset2id()
        self.embedding_size = args[args.dataset_name].embedding_size
        self.multimodal_layer = nn.Sequential(
                nn.Linear(self.embedding_size * args.num_modalities, self.embedding_size),  # concatenate all modalities
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size),)
        self.classifier = nn.Linear(self.embedding_size, args.num_classes)
        self.modality_names = args.modality_names
        self.K = args[args.dataset_name].augmentation_K
        print(f'Using {self.K} augmentations')

        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'])
            if args.downstream_strategy == 'frozen':
                # freeze the encoder
                for i, name in enumerate(self.modality_names):
                    for param in self.m_encoders[name].parameters():
                        param.requires_grad = False
                print("Encoder frozen")
            elif args.downstream_strategy == 'trainable':
                print("Encoder trainable")
            else:
                raise NotImplementedError
        self.criterion_recon_continuous = torch.nn.MSELoss()
        self.criterion = torch.nn.CrossEntropyLoss()

    def create_subset2id(self):
        # create a dictionary to map each combination of modalities to an id
        modalities = list(range(self.num_modalities))
        self.subsets = list(powerset(modalities))
        self.subsets = [comb for comb in self.subsets if len(comb) > 0]
        self.subset2id = {}
        self.id2subset = {}
        for i, comb in enumerate(self.subsets):
            mask = torch.ones(len(modalities), dtype=torch.bool)
            mask[list(comb)] = False
            self.subset2id[tuple(mask.tolist())] = i
            self.id2subset[i] = tuple(mask.tolist())
        self.num_subsets = len(self.subsets)

    def calculate_recon_loss(self, x: torch.Tensor, x_recon: torch.Tensor, mask: torch.Tensor):
        loss = []
        for i, name in enumerate(self.modality_names):
            x_i, x_recon_i = x[name], x_recon[name]
            if name in set(['img']):
                loss.append(self.criterion_recon_continuous(x_recon_i, x_i))
            elif name == 'text':
                # x_i one-hot encoded (B,L,D), x_recon_i is logsoftmax (B,L,D)
                loss_categorical = -torch.sum(x_i * x_recon_i, dim=-1)
                loss_categorical = torch.mean(loss_categorical)  # (B,)
                loss.append(loss_categorical)

        loss = sum(loss)
        return loss

    def forward_one_recon(self, x: torch.Tensor, mask: torch.Tensor):
        assert self.num_modalities == len(x)
        # encode all modalities
        out  = []
        for i, name in enumerate(self.modality_names):
            tmp = self.m_encoders[name](x[name]) 
            mask_i = mask[:, i].unsqueeze(1)  # (B, 1)
            tmp = tmp * (~mask_i).float()
            out.append(tmp)
        out = torch.cat(out, dim=1)  # (B, M*C)
        out = self.multimodal_layer(out)  # (B, C)

        # decode
        x= {}
        for i, name in enumerate(self.modality_names):
            x[name] = self.m_decoders[name](out)
        return x
    
    def forward_recon(self, x: torch.Tensor, mask: torch.Tensor):
        if self.K is None:
            out = self.forward_one_recon(x, mask)
            loss = self.calculate_recon_loss(x, out, mask)
            return (loss, out)
        
        # if K is not None, we need to reconstruct K times
        loss_list = []
        device = mask.device
        sample_idx = torch.randperm(len(self.id2subset), device=device)[:self.K]

        B = mask.shape[0]
        for id in sample_idx:
            mask = torch.tensor(self.id2subset[int(id)], device=device)
            mask = mask.unsqueeze(0).expand(B, -1)
            out = self.forward_one_recon(x, mask)
            loss = self.calculate_recon_loss(x, out, mask)
            loss_list.append(loss)
        loss = sum(loss_list) / len(loss_list)
        # return loss and output of the last subset
        return (loss, out)

    def forward_one_train(self, x: torch.Tensor, mask: torch.Tensor):
        assert self.num_modalities == len(x)
        # encode all modalities
        out  = []
        for i, name in enumerate(self.modality_names):
            tmp = self.m_encoders[name](x[name])   # (B, C)
            mask_i = mask[:, i].unsqueeze(1)  # (B, 1)
            tmp = tmp * (~mask_i).float()
            out.append(tmp)
        out = torch.cat(out, dim=1)  # (B, M*C)
        out = self.multimodal_layer(out)  # (B, C)
        out = self.classifier(out)
        return out
    
    def forward_train(self, x: torch.Tensor, mask: torch.Tensor, y: torch.Tensor):
        '''Calculate CE loss over all subsets'''
        if self.K is None: 
            out = self.forward_one_train(x, mask)
            loss = self.criterion(out, y)
            return (loss, out)
        
        loss_list = []
        sample_idx = torch.randperm(len(self.id2subset), device=mask.device)[:self.K]

        for id in sample_idx:
            mask = torch.tensor(self.id2subset[int(id)], device=mask.device)
            mask = mask.unsqueeze(0).expand(len(y), -1)
            out = self.forward_one_train(x, mask)
            loss = self.criterion(out, y)
            loss_list.append(loss)
        loss = sum(loss_list) / len(loss_list)
        # return loss and output of the last subset
        return (loss, out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        out = self.forward_one_train(x, mask)
        return out
    



if __name__ == "__main__":
    alphabet_path = join(project_path, 'datasets/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = {'CelebA':{'embedding_size':32, 'augmentation_K': None}, 
                         'dataset_name': 'CelebA', 'alphabet': alphabet, 'modality_names': ['img', 'text'],
                "num_modalities": 2, "checkpoint": None, "num_classes": 2}
    args = OmegaConf.create(args)
    model = MultiAE(args)
    indices = torch.randint(0, 71, (2, 256))
    one_hot_tensor = torch.nn.functional.one_hot(indices, num_classes=71).float()
    x = {'img': torch.randn(2, 3, 64, 64),  'text': one_hot_tensor}
    mask = torch.tensor([[True, False], [True, False]])
    loss, out = model.forward_recon(x, mask)
    for name in model.modality_names:
        print(f"{name} output shape: {out[name].shape}")
    print(loss)
    out = model.forward_train(x, mask, y=torch.tensor([0, 1]))

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)
