from typing import List, Tuple
from os.path import join
import os
import sys
from torch import nn
import albumentations as A

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np

def convert_to_ts(x, **kwargs):
  x = np.clip(x, 0, 255) / 255
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

def convert_to_ts_01(x, **kwargs):
  x  = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

def create_logdir(run_name: str, resume_training: bool, args):
    basepath = os.path.dirname(os.path.abspath(sys.argv[0]))
    basepath = os.path.join(os.path.dirname(os.path.dirname(basepath)), 'result')
    basepath = join(basepath,'runs', 'Dynamic_Missing')
    logdir = join(basepath,run_name)
    if os.path.exists(logdir) and not resume_training:
        raise Exception(f'Run {run_name} already exists. Please delete the folder {logdir} or choose a different run name.')
    os.makedirs(logdir,exist_ok=True)
    return logdir


def prepend_paths(hparams):
    db = hparams.data_base

    for hp in hparams:
        if hp.startswith('DATA_') and hp[-6:] != '_short' and hparams[hp]:
            hparams['{}_short'.format(hp)] = hparams[hp]
            hparams[hp] = join(db, hparams[hp])

    return hparams


def re_prepend_paths(hparams):
    db = hparams.data_base

    for hp in hparams:
        if hp.startswith('DATA_') and hp[-6:] != '_short' and hparams[hp]:
            hparams[hp] = join(db, hparams['{}_short'.format(hp)])

    return hparams


def fill_specific_params(hparams):
    '''
    Get specific parameters for the model and dataset and fill the global hparams dict
    '''
    specific_params = hparams[hparams.dataset_name]
    model_specific_keys = []
    for key in specific_params.keys():
        try:
            hparams[key] = specific_params[key]
        except:
            model_specific_keys.append(key)
    print(f'Model specific keys: {model_specific_keys}')
    return hparams


class AverageMeter(object):
    """Computes and stores the average and current value
    from https://github.com/LiheYoung/ShrinkMatch/blob/main/ImageNet/utils/utils.py"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class InstantMeter(object):
    """Computes and stores the average and current value
    from https://github.com/LiheYoung/ShrinkMatch/blob/main/ImageNet/utils/utils.py"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def grab_arg_from_checkpoint(args: str, arg_name: str):
  """
  Loads a lightning checkpoint and returns an argument saved in that checkpoints hyperparameters
  """
  if args.checkpoint:
    ckpt = torch.load(args.checkpoint)
    load_args = ckpt['hyper_parameters']
  else:
    load_args = args
  return load_args[arg_name]


def get_transforms_PolyMNIST(img_size: Tuple[int, int], mode: str):
    use_transforms = transforms.Compose([transforms.ToTensor()])
    return use_transforms

def get_transforms_MST(img_size: Tuple[int, int], mode: str):
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Resize(size=(img_size, img_size), interpolation=InterpolationMode.BICUBIC),
                                              transforms.ToTensor()])
    transform_svhn = transforms.Compose([transforms.ToTensor()])
    use_transforms = [transform_mnist, transform_svhn]
    return use_transforms

def get_transforms_CelebA(img_size: Tuple[int, int], mode: str):
    offset_height = (218 - 148) // 2
    offset_width = (178 - 148) // 2
    crop = lambda x: x[:, offset_height:offset_height + 147,
                        offset_width:offset_width + 147]
    use_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(crop),
                                    transforms.ToPILImage(),
                                    transforms.Resize(size=(64,
                                                            64),
                                                        interpolation=InterpolationMode.BICUBIC),
                                    transforms.ToTensor()])
    return use_transforms

def get_transforms_DVM(img_size: int, mode: str):
    use_transforms = A.Compose([
                        A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, p=0.8),
                        A.ToGray(p=0.2),
                        A.GaussianBlur(blur_limit=(29,29), sigma_limit=(0.1,2.0), p=0.5),
                        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333)),
                        A.HorizontalFlip(p=0.5),
                        A.Lambda(name='convert2tensor', image=convert_to_ts)
                    ])
    return use_transforms

def get_transforms_cardiac(img_size: int, mode: str):
    use_transforms = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=45),
                    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                    A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0)),
                    A.Lambda(name='convert2tensor', image=convert_to_ts_01)
                ])
    return use_transforms

def get_transforms(img_size, target, mode):
    if target.lower() == 'polymnist':
        use_transforms = get_transforms_PolyMNIST(img_size, mode)
        print(f"Using PlyMNIST transforms for {mode} mode")
    elif target.lower() == 'mst':
        use_transforms = get_transforms_MST(img_size, mode)
        print(f"Using MST transforms for {mode} mode")
    elif target.lower() == 'celeba':
        use_transforms = get_transforms_CelebA(img_size, mode)
        print(f"Using CelebA transforms for {mode} mode")
    elif target.lower() == 'dvm':
        use_transforms = get_transforms_DVM(img_size, mode)
        print(f"Using DVM transforms for {mode} mode")
    elif target.lower() in set(['cad', 'infarction']):
        use_transforms = get_transforms_cardiac(img_size, mode)
        print(f"Using cardiac transforms for {mode} mode")
    else:
        raise ValueError(f"Unknown target {target}")
    
    return use_transforms
 
