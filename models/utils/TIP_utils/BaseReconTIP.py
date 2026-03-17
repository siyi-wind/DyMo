'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
'''
import torch
import torch.nn as nn

import sys
import copy
from omegaconf import OmegaConf, open_dict
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)

from models.utils.TIP_utils.Transformer import TabularTransformerEncoder, MultimodalTransformerEncoder, TabularPredictor
from models.utils.TIP_utils.build_ssl_encoder import torchvision_ssl_encoder

class ReconTIP(nn.Module):
  """
  Evaluation model for TIP.
  """
  def __init__(self, args) -> None:
    super(ReconTIP, self).__init__()
    self.missing_tabular = args.missing_tabular
    print(f'Current intra-missing tabular for TIPBackbone: {self.missing_tabular}')
    if args.checkpoint:
      print(f'TIP checkpoint name: {args.checkpoint}')
      # Load weights
      checkpoint = torch.load(args.checkpoint)
      original_args = OmegaConf.create(checkpoint['hyper_parameters'])
      # original_args.field_lengths_tabular = args.DATA_field_lengths_tabular
      state_dict = checkpoint['state_dict']
      if 'algorithm_name' not in original_args:
        with open_dict(original_args):
          # original_args.algorithm_name = args.algorithm_name
          original_args.algorithm_name = 'TIP'
      self.hidden_dim = original_args.multimodal_embedding_dim

      # load image encoder
      if 'encoder_imaging.0.weight' in state_dict:
        self.encoder_name_imaging = 'encoder_imaging.'
      else:
        encoder_name_dict = {'clip' : 'encoder_imaging.', 'remove_fn' : 'encoder_imaging.', 'supcon' : 'encoder_imaging.', 'byol': 'online_network.encoder.', 'simsiam': 'online_network.encoder.', 'swav': 'model.', 'barlowtwins': 'network.encoder.'}
        self.encoder_name_imaging = encoder_name_dict[original_args.loss]

      if original_args.model.startswith('resnet'):
        self.encoder_imaging = torchvision_ssl_encoder(original_args.model, return_all_feature_maps=True)
      
      # load tabular encoder
      self.create_tabular_model(original_args, args)
      self.encoder_name_tabular = 'encoder_tabular.'
      assert len(self.cat_lengths_tabular) == original_args.num_cat
      assert len(self.con_lengths_tabular) == original_args.num_con
      # load multimodal encoder
      self.create_multimodal_model(original_args)
      self.encoder_name_multimodal = 'encoder_multimodal.'
      # load tabular predictor
      self.create_predictor_tabular(original_args)
      self.name_predictor_tabular = 'predictor_tabular.'
      finetune_strategy = 'frozen'

      for module, module_name in zip([self.encoder_imaging, self.encoder_tabular, self.encoder_multimodal, self.predictor_tabular], 
                                     [self.encoder_name_imaging, self.encoder_name_tabular, self.encoder_name_multimodal, self.name_predictor_tabular]):
        self.load_weights(module, module_name, state_dict)
        if finetune_strategy == 'frozen':
          for _, param in module.named_parameters():
            param.requires_grad = False
          parameters = list(filter(lambda p: p.requires_grad, module.parameters()))
          assert len(parameters)==0
          print(f'Freeze {module_name}')
        elif finetune_strategy == 'trainable':
          print(f'Full finetune {module_name}')
        else:
          assert False, f'Unknown finetune strategy {args.finetune_strategy}'

    else:
      self.create_imaging_model(args)
      self.create_tabular_model(args)
      self.create_multimodal_model(args)
      self.create_tabular_predictor(args)
      self.hidden_dim = args.multimodal_embedding_dim

    self.classifier = nn.Linear(self.hidden_dim, args.num_classes)


  def create_imaging_model(self, args):
    if args.model.startswith('resnet'):
      self.encoder_imaging = torchvision_ssl_encoder(args.model, return_all_feature_maps=True)
  
  def create_tabular_model(self, args, new_args):
    try:
      self.field_lengths_tabular = torch.load(new_args.field_lengths_tabular)
    except:
      self.field_lengths_tabular = torch.load(new_args.DATA_field_lengths_tabular)
    self.cat_lengths_tabular = []
    self.con_lengths_tabular = []
    for x in self.field_lengths_tabular:
      if x == 1:
        self.con_lengths_tabular.append(x) 
      else:
        self.cat_lengths_tabular.append(x)
    self.encoder_tabular = TabularTransformerEncoder(args, self.cat_lengths_tabular, self.con_lengths_tabular)

  def create_multimodal_model(self, args):
    self.encoder_multimodal = MultimodalTransformerEncoder(args)
  
  def create_predictor_tabular(self, args):
    self.predictor_tabular = TabularPredictor(args, self.cat_lengths_tabular, self.con_lengths_tabular, self.encoder_tabular.num_unique_cat)
  
  def load_weights(self, module, module_name, state_dict):
    state_dict_module = {}
    for k in list(state_dict.keys()):
      if k.startswith(module_name) and not 'projection_head' in k and not 'prototypes' in k:
        state_dict_module[k[len(module_name):]] = state_dict[k]
    print(f'Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}')
    log = module.load_state_dict(state_dict_module, strict=True)
    assert len(log.missing_keys) == 0

  def forward(self, x, missing_mask, visualize=False) -> torch.Tensor:
    # check missing_mask, TIP don't support missing imaging features
    x_i, x_t = x['img'], x['tabular']
    x_i = self.encoder_imaging(x_i)[-1]   # (B,C,H,W)

    x_t = self.encoder_tabular(x=x_t, mask=missing_mask, mask_special=missing_mask)

    x_m = self.encoder_multimodal(x=x_t, image_features=x_i)

    out = self.predictor_tabular(x_m)  # cat_x, con_x

    # recover tabular features 
    origin_out = copy.deepcopy(out)
    out_cat, out_con = out
    with torch.no_grad():
      out_cat = out_cat.argmax(dim=-1)
      out_cat = out_cat - self.encoder_tabular.cat_offsets
      T = torch.tensor(self.cat_lengths_tabular, device=out_cat.device).unsqueeze(0)
      out_cat_clamped = out_cat.clamp(min=torch.zeros_like(T, device=T.device), max=T-1)
      out = torch.cat([out_cat_clamped.float(), out_con.squeeze(-1)], dim=-1)
      # replace the non-missing values with the original values   1 means missing
      out = out * missing_mask.float() + x['tabular'] * (~missing_mask).float()

    if visualize:
      return {'img': x['img'], 'tabular': out, 'tabular_missing_mask': missing_mask}, origin_out
    else:
      return {'img': x['img'], 'tabular': out, 'tabular_missing_mask': missing_mask}



if __name__ == "__main__":
  args = OmegaConf.create({'model': 'resnet50', 'checkpoint': '/bigdata/siyi/data/result/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt', 
                  'num_cat': 4, 'num_con': 13, 'num_classes': 283, 
                  'DATA_field_lengths_tabular': '/bigdata/siyi/data/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                  'tabular_embedding_dim': 512, 'tabular_transformer_num_layers': 4, 'multimodal_transformer_layers': 4,'embedding_dropout': 0.0, 'drop_rate':0.0,
                    'embedding_dim': 2048, 
                    'multimodal_embedding_dim': 512, 'multimodal_transformer_num_layers': 4, 'missing_tabular': True})
  model = ReconTIP(args)
  x_i = torch.randn(2, 3, 128, 128)
  x_t = torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                    [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)
  mask = torch.tensor([[False, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False],
                       [False, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, False]])
  y = model(x={'img': x_i, 'tabular': x_t}, missing_mask=mask)
  for k, v in y.items():
    print(k, v.shape)