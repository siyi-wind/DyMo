from typing import Tuple, List

import torch
from torch import nn
import sys
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../'))
sys.path.append(project_path)
import torch.nn.functional as F
import time


class PrototypeDistance_COS(nn.Module):
    '''
    This is used to measure the quality of the multimodal features
    - Calculate the distance between the feature and the predicted class prototype
    - Calculate L1 uniformity as the confidence of the prediction
    '''
    def __init__(self, temperature) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self, feat: torch.Tensor, prototypes: torch.Tensor):
        # calculate similarity between prototypes and features
        sim = torch.mm(feat, prototypes.t())/self.temperature  # (B, C)
        prediction = F.softmax(sim, dim=-1)
        quality, max_id = torch.max(prediction, dim=1)
        # calculate confidence using L1 uniformity
        # uniform = 1/prediction.size(1)
        # conf = torch.sum(torch.abs(prediction-uniform),dim=1)/(prediction.size(1))
        return quality, max_id, prediction
    

class PrototypeDistance_EU(nn.Module):
    '''
    This is used to measure the quality of the multimodal features
    - Calculate the distance between the feature and the predicted class prototype
    - Caculate euclidean distance
    '''
    def __init__(self, temperature) -> None:
        super().__init__()
        self.temperature = temperature
    
    def forward(self, feat: torch.Tensor, prototypes: torch.Tensor, subsets_ids: torch.Tensor=None):
        # calculate EU distance between prototypes and features
        feat = feat.unsqueeze(1)  # (B, 1, D)
        prototypes = prototypes.unsqueeze(0)  # (1, C, D)
        out = -((feat - prototypes).pow(2).sum(dim=2)) /self.temperature  # (B, C)
        prediction = F.softmax(out, dim=-1)
        quality, max_id = torch.max(prediction, dim=1)
        return quality, max_id, prediction
    

def check_update_effect(origin_pred, update_pred, y):
    correct_origin = (origin_pred == y)
    correct_update = (update_pred == y)
    improvement = correct_update.sum().item() - correct_origin.sum().item() 
    print(f"Total: {len(y)}. Correctness before update: {correct_origin.sum().item()}, after update: {correct_update.sum().item()}, improvement: {improvement}")
    return correct_origin, correct_update, improvement


def calculate_probabilities_EU(feat, mask, max_id, subset_gaussian):
    subset2id = subset_gaussian['subset2id']
    prototypes_all = subset_gaussian['prototypes'].to(feat.device)  # (num_subsets, num_classes, D)
    dist_std_all = subset_gaussian['dist_std'].to(feat.device)  # (num_subsets, num_classes)
    probs = []
    dist_list = []
    for i in range(len(feat)):
        feat_i = feat[i]  #(D)
        mask_i = mask[i]
        label = int(max_id[i])
        subset_id = subset2id[tuple(mask_i.tolist())]  # (1)
        prototypes = prototypes_all[subset_id]  # (C, D)
        dist_std = dist_std_all[subset_id, label]  # (1)

        distribution = torch.distributions.Normal(0, dist_std)
        dist_classes = torch.norm(prototypes - feat_i, dim=-1) # (C)
        # dist_classes = torch.sum((prototypes - feat_i) ** 2, dim=-1)  # (C)
        dist_score = dist_classes[label]
        prob = 2 * (1 - distribution.cdf(dist_score))
        # print('i: ', i, 'dist all: ', dist_classes, 'Gaussian prob: ', prob)
        probs.append(prob)
        dist_list.append(dist_score)
    probs = torch.stack(probs).to(feat.device)  # (N)
    dist_list = torch.stack(dist_list).to(feat.device)  # (N)
    return probs, dist_list


def calculate_probabilities_COS(feat, mask, max_id, subset_gaussian):
    start_time = time.time()
    subset2id = subset_gaussian['subset2id']
    prototypes_all = subset_gaussian['prototypes'].to(feat.device)  # (num_subsets, num_classes, D)
    dist_std_all = subset_gaussian['dist_std'].to(feat.device)  # (num_subsets, num_classes)
    probs = []
    dist_list = []

    for i in range(len(feat)):
        feat_i = feat[i]  #(D)
        mask_i = mask[i]
        label = int(max_id[i])
        subset_id = subset2id[tuple(mask_i.tolist())]  # (1)
        prototypes = prototypes_all[subset_id]  # (C, D)
        dist_std = dist_std_all[subset_id, label]  # (1)

        distribution = torch.distributions.Normal(0, dist_std)
        dist_classes = torch.matmul(prototypes, feat_i.unsqueeze(1)).squeeze(1)  # (1)
        dist_score = 1 - dist_classes[label]
        prob = 2 * (1 - distribution.cdf(dist_score))

        probs.append(prob)
        dist_list.append(dist_score)
    probs = torch.stack(probs).to(feat.device)  # (N)
    dist_list = torch.stack(dist_list).to(feat.device)  # (N)
    end_time = time.time()
    inference_time = (end_time - start_time)*1000
    # print(f'Probability calculation time for batch size {len(feat)}: {inference_time:.4f} ms, per sample: {inference_time/len(feat):.4f} ms')
    return probs, dist_list





