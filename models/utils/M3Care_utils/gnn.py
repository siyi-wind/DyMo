from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
import math
from omegaconf import OmegaConf


def euclidean_dist(x, y):
    b = x.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(b, b)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(b, b).t()
    dist = xx+yy-2*torch.mm(x, y.t())
    dist = torch.clamp(dist, min=0.0) 
    return dist 

def guassian_kernel(source, kernel_mul=2.0, kernel_num=1, fix_sigma=None):
    n = source.size(0)
    L2_distance = euclidean_dist(source, source)

        
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n**2-n)
    
    if bandwidth < 1e-3:
        bandwidth = 1
    
    bandwidth /= kernel_mul ** (kernel_num//2)
    bandwidth_list = [bandwidth*(kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance/bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)/len(kernel_val)


# Our MAPLE framework
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
#         print(self.dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha# 4 leakyrelu
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.attention = None

    def forward(self, input, adj):# V N, V V
        h = torch.mm(input, self.W)# V O
        N = h.size()[0]# NUM OF V

        # V*V O ->123412341234, V*V O -> 111222333444, V V 2O
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))# V V

        zero_vec = -9e15*torch.ones_like(e)# V V
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        self.attention = attention
        attention = F.dropout(attention, self.dropout, training=self.training)# V V
        h_prime = torch.matmul(attention, h)# V N

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features).float())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).float())
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj, x):
        y = torch.mm(x.float(), self.weight.float())
        output = torch.mm(adj.float(), y.float())
        if self.bias is not None:
            return output + self.bias.float()
        else:
            return output



class GNNImputation(nn.Module):
    '''
    Impute missing modality features of one subject using other non-missing subjects.
    1. Input: x dict including modality features, mask
    2. Calculate the similarity matrix using guassian kernel
    3. Refine the similarity matrix (dissimilar subjects) 
    4. Use the similarity matrix to aggregate features from other subjects through GNN
    5. For non-missing modalities, weighted sum of the original and the aggregated features
    5. Output: the imputed modality features of the subject, a dict
    '''
    def __init__(self, args):
        super(GNNImputation, self).__init__()
        self.num_modalities = args.num_modalities
        self.modality_names = args.modality_names
        self.hidden_dim = args[args.dataset_name].embedding_size
        self.num_gnn_layers = args[args.dataset_name].num_gnn_layers

        self.m_SimProjections = nn.ModuleDict()
        self.m_GNNs = nn.ModuleDict()
        self.m_weights1 = nn.ModuleDict()
        self.m_weights2 = nn.ModuleDict()
        self.m_epsilons = nn.ParameterDict()
        for name in self.modality_names:
            self.m_SimProjections[name] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
                nn.ReLU(inplace=True),
                torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),)
            self.m_GNNs[name] = nn.ModuleList([
                GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True) for _ in range(self.num_gnn_layers)
                ])
            self.m_epsilons[name] = nn.Parameter(torch.ones(size=(1,))+1)
            self.m_weights1[name] = nn.Linear(self.hidden_dim, 1)
            self.m_weights2[name] = nn.Linear(self.hidden_dim, 1)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.threshold = nn.Parameter(torch.zeros(size=(1,)))
    
    def forward(self, x: Dict, mask: torch.Tensor):
        # notice 1 means non-missing modality, which is different from the original data input
        mask = mask.float()
        original_mask = mask.clone() # (B, num_modalities)
        mask_matrices = {}
        for i, name in enumerate(self.modality_names):
            mask_i = mask[:, i].unsqueeze(1) # (B, 1)
            # create mask matrix (B, B), only two non-missing modalities position 
            mask_matrices[name] = mask_i @ mask_i.t()
        
        # calculate the mat matrix
        mat_list = []
        diff_list = []
        for i, name in enumerate(self.modality_names):
            x_i, eps, mask_matrix = x[name], self.m_epsilons[name], mask_matrices[name]
            mat1 = guassian_kernel(self.bn(self.m_SimProjections[name](x_i)), kernel_mul=2.0, kernel_num=3)
            mat2 = guassian_kernel(self.bn(x_i), kernel_mul=2.0, kernel_num=3)
            mat = (1-torch.sigmoid(eps)) * mat1 + torch.sigmoid(eps) * mat2
            mat = mat * mask_matrix
            mat_list.append(mat)
            diff = torch.abs(torch.norm(self.m_SimProjections[name](x_i)) - torch.norm(x_i))
            diff_list.append(diff)
        
        # calculate the similarity scores and diffs
        sum_of_diff = sum(diff_list)
        similar_score = sum(mat_list) / (sum(mask_matrices.values())+1e-9)
        similar_score = F.relu(similar_score - torch.sigmoid(self.threshold)[0])  
        temp_thresh = torch.sigmoid(self.threshold)[0]
        bin_mask = similar_score > 0
        similar_score = similar_score + bin_mask * temp_thresh.detach()

        # aggregate features using GNN
        out = {}
        for i, name in enumerate(self.modality_names):
            x_i, mask_matrix = x[name], mask_matrices[name]
            mask_i = mask[:, i].unsqueeze(1)  # (B, 1)
            tmp = x_i
            for _, gnn_layer in enumerate(self.m_GNNs[name]):
                tmp = gnn_layer(similar_score, tmp)
                tmp = F.selu(tmp)

            # weighted sum of the original and the aggregated features
            weight1 = torch.sigmoid(self.m_weights1[name](tmp))
            weight2 = torch.sigmoid(self.m_weights2[name](x_i))
            weight1 = weight1/(weight1+weight2+1e-9)
            weight2 = 1 - weight1
            tmp_weighted = weight1 * tmp + weight2 * x_i

            # use mask_i to decide whether to use tmp or tmp_weighted, 1 use tmp_weighted, 0 use tmp
            tmp = tmp_weighted * mask_i + tmp * (1 - mask_i)
            out[name] = tmp
        
        return out, sum_of_diff


            
if __name__ == "__main__":
    args = {'PolyMNIST':{"embedding_size": 128, "num_gnn_layers": 2}, 
            'dataset_name': 'PolyMNIST',
            "num_modalities": 3, "modality_names": ['m0', 'm1', 'm2']}
    args = OmegaConf.create(args)
    model = GNNImputation(args)
    x = {'m0': torch.randn(2, 128), 'm1': torch.randn(2, 128), 'm2': torch.randn(2, 128)}
    mask = torch.tensor([[True, False, False], [False, False, True]])  # 2 samples, 3 modalities
    output = model.forward(x, ~mask)
            
