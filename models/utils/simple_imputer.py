import torch
import torch.nn as nn


class SimpleImputer(nn.Module):
    """
    A simple imputer that fills missing values with a specified constant.
    p is a probability, each sample has p chance to use the imputer
    """

    def __init__(self, modality_names, p=1, fill_value=0.0):
        super(SimpleImputer, self).__init__()
        self.fill_value = fill_value
        self.modality_names = modality_names
        self.p = p

    def forward(self, x, mask):
        """
        Impute missing values in the input tensor.

        Args:
            x dict(key: tensorTensor): Input dict, with each key a modality.
            mask (torch.Tensor): Binary mask tensor with the same shape as x,
                                 where 1 indicates missing values and 0 indicates observed values.
        Returns:
            torch.Tensor: Imputed tensor with missing values filled.
        """
        # Create a copy of the input tensor
        imputed = {}
        # create use_imputer mask, 1 means use imputer, (B,)
        use_imputer = (torch.rand(x[list(x.keys())[0]].shape[0]) < self.p).float().to(mask.device)
        for i, name in enumerate(self.modality_names):
            tmp_original = x[name].clone()
            tmp = x[name].clone()
            mask_tmp = mask[:, i]
            while mask_tmp.dim() < tmp.dim():
                mask_tmp = mask_tmp.unsqueeze(-1)
            imputed[name] = torch.where(mask_tmp == 1,
                        torch.tensor(self.fill_value, dtype=tmp.dtype, device=tmp.device), tmp)
            # only impute those samples that use imputer
            use_imputer_tmp = use_imputer.clone()
            while use_imputer_tmp.dim() < tmp.dim():
                use_imputer_tmp = use_imputer_tmp.unsqueeze(-1)
            imputed[name] = torch.where(use_imputer_tmp == 1, imputed[name], tmp_original)
        return imputed


if __name__ == "__main__":
    # Example usage
    modality_names = ['modality1', 'modality2']
    imputer = SimpleImputer(modality_names, p=0.5, fill_value=-1.0)

    # Create a sample input tensor with missing values
    x = {
        'modality1': torch.randn(3,1,4,4),
        'modality2': torch.randn(3,12),
    }
    mask = torch.tensor([[1, 0], [1, 0], [1, 0]])  # 1 indicates missing

    # Impute missing values
    imputed_x = imputer(x, mask)
    print("Original x:", x)
    print("Mask:", mask)
    print("Imputed x:", imputed_x)