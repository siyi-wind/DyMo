import torch
import torch.nn as nn
import sys
from omegaconf import OmegaConf
from itertools import chain, combinations

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
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


class UnimodalDecoder(nn.Module):
    def __init__(self,):
        super(UnimodalDecoder, self).__init__()
        self.network  = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor):
        x = self.network(x)
        return x

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
        self.m_encoders = nn.ModuleDict()
        self.m_decoders = nn.ModuleDict()
        for i in range(args.num_modalities):
            self.m_encoders[f'm{i}'] = UnimodalEncoder()
            self.m_decoders[f'm{i}'] = UnimodalDecoder()
        self.embedding_size = args[args.dataset_name].embedding_size
        self.multimodal_layer = nn.Sequential(
                nn.Linear(self.embedding_size * args.num_modalities, self.embedding_size),  # concatenate all modalities
                nn.ReLU(),
                nn.Linear(self.embedding_size, self.embedding_size),)
        self.create_subset2id()
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
        self.criterion_recon = torch.nn.MSELoss()
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
            loss.append(self.criterion_recon(x_recon_i, x_i))
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
    args = {'PolyMNIST':{'embedding_size': 512, 'augmentation_K': None}, 
                         'dataset_name': 'PolyMNIST', 'modality_names': ['m0', 'm1', 'm2'],
                "num_modalities": 3, "checkpoint": None, "num_classes": 10}
    args = OmegaConf.create(args)
    model = MultiAE(args)
    x = {'m0': torch.randn(2, 3, 28, 28), 'm1': torch.randn(2, 3, 28, 28), 'm2': torch.randn(2, 3, 28, 28)}
    mask = torch.tensor([[True, True, False], [False, True, True]])
    loss, out = model.forward_recon(x, mask)
    for name in model.modality_names:
        print(f"{name} output shape: {out[name].shape}")
    print(loss)
    out = model.forward_train(x, mask, y=torch.tensor([0, 1]))

    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(num_params/1e6)
