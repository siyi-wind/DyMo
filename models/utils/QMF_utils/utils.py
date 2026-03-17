import torch.nn as nn
import torch
import numpy as np

def rank_loss(confidence, idx, history):
    # make input pair
    rank_input1 = confidence
    rank_input2 = torch.roll(confidence, -1)
    idx2 = torch.roll(idx, -1)

    # calc target, margin
    rank_target, rank_margin = history.get_target_margin(idx, idx2)
    rank_target_nonzero = rank_target.clone()
    rank_target_nonzero[rank_target_nonzero == 0] = 1
    rank_input2 = rank_input2 + (rank_margin / rank_target_nonzero).reshape((-1,1))

    # ranking loss
    ranking_loss = nn.MarginRankingLoss(margin=0.0)(rank_input1,
                                        rank_input2,
                                        -rank_target.reshape(-1,1))

    return ranking_loss


def create_history(modality_names, n_data):
    history = {}
    for name in modality_names:
        history[name] = History(n_data)
    return history


class History(object):
    def __init__(self, n_data):
        self.correctness = np.zeros((n_data))
        self.confidence = np.zeros((n_data))
        self.max_correctness = 1

    # correctness update
    def correctness_update(self, data_idx, correctness, confidence):
        #probs = torch.nn.functional.softmax(output, dim=1)
        #confidence, _ = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()

        self.correctness[data_idx] += correctness.cpu().numpy()
        self.confidence[data_idx] = confidence.cpu().detach().numpy()

    # max correctness update
    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, data):
        data_min = self.correctness.min()
        #data_max = float(self.max_correctness)
        data_max = float(self.correctness.max())

        return (data - data_min) / (data_max - data_min)

    # get target & margin
    def get_target_margin(self, data_idx1, data_idx2):
        device = data_idx1.device
        data_idx1 = data_idx1.cpu().numpy()
        data_idx2 = data_idx2.cpu().numpy()
        cum_correctness1 = self.correctness[data_idx1]
        cum_correctness2 = self.correctness[data_idx2]
        # normalize correctness values
        cum_correctness1 = self.correctness_normalize(cum_correctness1)
        cum_correctness2 = self.correctness_normalize(cum_correctness2)
        # make target pair
        n_pair = len(data_idx1)
        target1 = cum_correctness1[:n_pair]
        target2 = cum_correctness2[:n_pair]
        # calc target
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)

        target = greater + less
        target = torch.from_numpy(target).float().to(device)
        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().to(device)

        return target, margin
    

