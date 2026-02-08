from typing import Dict
import torch

def generate_modality_dropout_mask(x: Dict[str, torch.Tensor], missing_mask: torch.Tensor=None, p: float=0.5): 
    '''
    Randomly drops modalities in the input dictionary x with probability p.
    Output new final mask and dropout mask
    Noted designs:
    1. 1 means the modality is dropped. The final dropout mask: 1 means the modality is dropped or missing 
    2. at leas one modality should be present in the output
    '''
    modality_names = list(x.keys())
    B, M = len(x[modality_names[0]]), len(modality_names)
    device = x[modality_names[0]].device
    if missing_mask is None:
        missing_mask = torch.zeros(B, M, dtype=torch.bool, device=device)  # 1 means missing
    
    dropout_mask = torch.rand(B, M, device=device) < p   # 1 means dropped
    final_mask = missing_mask | dropout_mask  # 1 means missing or dropped
    # ensure at least one modality is present
    if final_mask.all(dim=1).any():
        # find the indices where all modalities are missing
        missing_row_idx = final_mask.all(dim=1).nonzero(as_tuple=True)[0]
        # randomly select one non-missing modality for these rows
        noise = torch.rand(B, M, device=device) * (~missing_mask)
        ids_shuffle = torch.argsort(noise, dim=1, descending=True)
        ids_second_mask = ids_shuffle[:, 0]
        final_mask[missing_row_idx, ids_second_mask[missing_row_idx]] = False
        dropout_mask[missing_row_idx, ids_second_mask[missing_row_idx]] = False
    assert torch.equal(final_mask & missing_mask, missing_mask), "Final mask must include all original missing modalities"
    assert torch.equal(final_mask & dropout_mask, dropout_mask), "Final mask must include all dropped modalities"
    assert (final_mask.sum(dim=1) < M).all(), "Each sample must have at least one modality present"


    return final_mask, dropout_mask


if __name__ == "__main__":
    x = {
        'm1': torch.randn(4, 3, 32, 32),
        'm2': torch.randn(4, 3, 32, 32),
        'm3': torch.randn(4, 3, 32, 32)
    }
    missing_mask = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1]], dtype=torch.bool)
    final_mask, dropout_mask = generate_modality_dropout_mask(x, missing_mask=missing_mask, p=0.9)
    print(missing_mask)
    print(dropout_mask)
    print(final_mask)