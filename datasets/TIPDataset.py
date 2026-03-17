'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2024
* Based on MMCL codebase https://github.com/paulhager/MMCL-Tabular-Imaging/blob/main/datasets/ImagingAndTabularDataset.py
'''
from typing import List, Tuple
import random
import csv
import copy

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
from torchvision.io import read_image
import albumentations as A
import numpy as np
import os
import sys
from os.path import join
import albumentations as A

current_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from utils.utils import get_transforms

def convert_to_float(x):
  return x.float()

def convert_to_ts(x, **kwargs):
  x = np.clip(x, 0, 255) / 255
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

def convert_to_ts_01(x, **kwargs):
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x


class ImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that imaging and tabular data for evaluation.
  Load mask csv to imitate missing tabular data
  missing_strategy: value or feature
  missing_rate: 0.0 to 1.0

  The imaging view has {eval_train_augment_rate} chance of being augmented.
  The tabular view corruption rate to be augmented.
  """
  def __init__(
      self,
      data_path_imaging: str, delete_segmentation: bool, eval_train_augment_rate: float, 
      data_path_tabular: str, field_lengths_tabular: str, eval_one_hot: bool,
      labels_path: str, img_size: int, live_loading: bool, train: bool, target: str,
      corruption_rate: float, data_base: str, missing_path: str, missing_tabular: str=False, missing_strategy: str='value', missing_rate: float=0.0, augmentation_speedup: bool=False, 
      algorithm_name: str=None, return_idx: bool=False) -> None:

    # Imaging
    self.missing_tabular = missing_tabular
    self.data_imaging = torch.load(data_path_imaging)
    self.delete_segmentation = delete_segmentation
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading
    self.augmentation_speedup = augmentation_speedup
    self.split_dataset_name = data_path_tabular.split('/')[-1].split('_')[0]

    self.missing_path = str(missing_path)
    if self.missing_path == 'none':
        mask = [0, 0]
    else:
        mask = [str(i) in self.missing_path for i in range(2)]
    self.mask = mask
    modality_names = ['img', 'tabular']
    # print missing modalities mask=True
    print(f'missing mask examples: {self.mask}, ', [modality_names[i] for i in range(len(mask)) if mask[i]])


    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    self.transform_train = get_transforms(img_size, target, mode='train')

    if augmentation_speedup:
      if self.split_dataset_name == 'dvm':
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts)
        ])
        print('Using dvm transform for default transform for evaluation')
      elif self.split_dataset_name == 'cardiac':
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts_01)
        ])
        print('Using cardiac transform for default transform in ImagingAndTabularDataset evaluation')
      else:
        raise print('Only support dvm and cardiac datasets in ImagingAndTabularDataset')
    else:
      self.default_transform = transforms.Compose([
        transforms.Resize(size=(img_size,img_size)),
        transforms.Lambda(convert_to_float)
      ])

    # Tabular
    self.data_tabular = np.array(self.read_and_parse_csv(data_path_tabular))
    self.generate_marginal_distributions()
    self.field_lengths_tabular = np.array(torch.load(field_lengths_tabular))
    self.eval_one_hot = eval_one_hot
    self.c = corruption_rate if corruption_rate else None

    # Missing mask
    self.missing_strategy = missing_strategy
    self.algorithm_name = algorithm_name
    if self.missing_tabular:
      tabular_name = data_path_tabular.split('/')[-1][: -4]
      if 'filled' in tabular_name:
        print('Use filled tabular data')
        assert str(missing_rate) in tabular_name
        # Remove f'_mask_{missing_rate}_filled' from tabular_name
        tabular_name = tabular_name.replace(f'_mask_{missing_rate}_filled','')
      tmp_target = target.lower() if target=='DVM' else target
      missing_mask_path = join(data_base, 'missing_mask', f'{tabular_name}_{tmp_target}_{missing_strategy}_{missing_rate}.npy')
      self.missing_mask_data = np.load(missing_mask_path)
      print(f'Load missing mask from {missing_mask_path}')
      assert len(self.data_imaging) == self.missing_mask_data.shape[0]
      if self.eval_one_hot and missing_strategy in set(['feature', 'MI', 'LI']):
        self.field_lengths_tabular = self.field_lengths_tabular[~self.missing_mask_data[0]]
        print('Onehot input tabular feature size: ', len(self.field_lengths_tabular), int(np.sum(self.field_lengths_tabular)))
      else:
        print('Transformer input tabular feature size: ', len(self.field_lengths_tabular), len(self.field_lengths_tabular))

    # Classifier
    self.labels = torch.load(labels_path)

    self.train = train
    assert len(self.data_imaging) == len(self.data_tabular) == len(self.labels) 
    self.return_idx = return_idx
  
  def read_and_parse_csv(self, path_tabular: str) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    with open(path_tabular,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def generate_marginal_distributions(self) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    data = np.array(self.data_tabular)
    self.marginal_distributions = np.transpose(data)

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)
    subject = np.array(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      marg_dist = self.marginal_distributions[i][~self.missing_mask_data[:,i]] if self.missing_tabular else self.marginal_distributions[i]
      if marg_dist.size != 0:
        value = np.random.choice(marg_dist, size=1)
        subject[i] = value
    return subject

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(np.sum(self.field_lengths_tabular))
    else:
      return len(self.field_lengths_tabular)

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    if self.missing_tabular and self.missing_strategy in set(['feature', 'MI', 'LI']) and self.eval_one_hot:
      subject = subject[~self.missing_mask_data[0]]
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(torch.clamp(subject[i],min=0,max=self.field_lengths_tabular[i]-1).long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    im = self.data_imaging[index]
    path = im
    if self.live_loading:
      if self.augmentation_speedup:
        im = np.load(im[:-4]+'.npy', allow_pickle=True)
      else:
        im = read_image(im)
        im = im / 255

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(image=im)['image'] if self.augmentation_speedup else self.transform_train(im)
      if self.c > 0:
        tab = torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)
      else:
        tab = torch.tensor(self.data_tabular[index], dtype=torch.float)
    else:
      im = self.default_transform(image=im)['image'] if self.augmentation_speedup else self.default_transform(im)
      tab = torch.tensor(self.data_tabular[index], dtype=torch.float)

    if self.eval_one_hot:
      tab = self.one_hot_encode(tab).to(torch.float)

    label = torch.tensor(self.labels[index], dtype=torch.long)
    total_mask = torch.tensor(self.mask, dtype=bool)

    sample = {'img': im, 'tabular': tab}

    if self.missing_tabular:
      missing_mask = torch.from_numpy(self.missing_mask_data[index])
      assert total_mask[1] == True, f"Missing intra-tabular data but total tabular mask is {total_mask[1]} at index {index}"
      sample['tabular_missing_mask'] = missing_mask
    
    if self.return_idx:
      sample['index'] = torch.LongTensor([index])
    return sample, total_mask, label
    #   return {'img': im, 'tabular': tab, 'tabular_missing_mask': missing_mask}, total_mask, label
    # else:
    #   return {'img': im, 'tabular': tab}, total_mask, label

  def __len__(self) -> int:
    return len(self.data_tabular)
  
  
if __name__ == '__main__':
  BASE = '/bigdata/siyi/data/UKBB/cardiac_segmentations/projects/SelfSuperBio/18545/final/'
  dataset = ImagingAndTabularDataset(
    data_path_imaging=BASE+'cardiac_train_paths_imaging_CAD_balanced.pt', delete_segmentation=False, eval_train_augment_rate=0.8, 
          data_path_tabular=BASE+'cardiac_features_train_imputed_noOH_tabular_imaging_CAD_balanced_reordered.csv', 
          field_lengths_tabular=BASE+'tabular_lengths_reordered.pt', eval_one_hot=False,
          labels_path=BASE+'cardiac_labels_CAD_train_balanced.pt', img_size=128, live_loading=True, train=True, target='CAD',
          corruption_rate=0.3, data_base=BASE, missing_tabular=True, 
          missing_strategy='feature', missing_rate=0.7, augmentation_speedup=True, algorithm_name='TIP', missing_path='1',
  )
  sample, total_mask, label = dataset[0]
  print(sample['img'].shape, sample['tabular'].shape, label.shape)
  print(dataset.missing_mask_data.sum()/dataset.missing_mask_data.size)
  print('Number of samples: ', len(dataset))

  # tab=self.one_hot_encode(tab).to(torch.float)