import os 
from os.path import join
import wandb

import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import json

from utils.utils import get_transforms, grab_arg_from_checkpoint
from experiments.DownstreamExperiment import DownstreamExperiment


def load_datasets(hparams):
    if hparams.dataset_name == 'PolyMNIST':
        image_size = grab_arg_from_checkpoint(hparams, 'image_size')
        from datasets.PolyMNISTDataset import PolyMNISTDataset
        test_dataset = PolyMNISTDataset(
            unimodal_datapaths=hparams.DATA_test, data_base=hparams.data_base, missing_path=hparams.missing_test, transform=get_transforms(image_size,hparams.target,'test'),
            target_transform=None,)
    elif hparams.dataset_name == 'MST':
        from datasets.MSTDataset import SVHNMNIST
        flags = {'dir_data': hparams.data_base, 'len_sequence': 8, 'data_multiplications': 20}
        flags = OmegaConf.create(flags)
        alphabet_path = join(hparams.data_base, 'alphabet.json')
        with open(alphabet_path) as alphabet_file:
            alphabet = str(''.join(json.load(alphabet_file)))
        test_dataset = SVHNMNIST(flags, alphabet, train='test', missing_path=hparams.missing_test, transform=get_transforms(hparams.image_size, hparams.target, 'test'))
        hparams.alphabet = alphabet
    elif hparams.dataset_name == 'CelebA':
        from datasets.CelebADataset import CelebaDataset
        flags = {'dir_data': hparams.data_base, 'dir_text': hparams.data_base, 'len_sequence': 256, 'random_text_ordering': False, 'random_text_startindex': True}
        flags = OmegaConf.create(flags)
        alphabet_path = join(hparams.data_base, 'alphabet.json')
        with open(alphabet_path) as alphabet_file:
            alphabet = str(''.join(json.load(alphabet_file)))
        test_dataset = CelebaDataset(flags, alphabet, missing_path=hparams.missing_test, partition=2, transform=get_transforms(hparams.image_size, hparams.target, 'test'))
        hparams.alphabet = alphabet
    elif hparams.dataset_name in set(['DVM', 'CAD', 'Infarction']):
        from datasets.TIPDataset import ImagingAndTabularDataset
        test_dataset = ImagingAndTabularDataset(
                hparams.DATA_data_test_eval_imaging, hparams.delete_segmentation, 0, 
                hparams.DATA_data_test_eval_tabular, hparams.DATA_field_lengths_tabular, hparams.eval_one_hot,
                hparams.DATA_labels_test_eval_imaging, hparams.image_size, hparams.live_loading, train=False, target=hparams.target,
                corruption_rate=0.0, data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
                augmentation_speedup=hparams.augmentation_speedup,algorithm_name=hparams.algorithm_name,
                missing_path=hparams.missing_test)
        hparams.input_size = test_dataset.get_input_size()
    print(f"Test: {len(test_dataset)}")

    return test_dataset


def test(hparams, wandb_run):
    # seed everything
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    test_dataset = load_datasets(hparams)
    test_loader = DataLoader(test_dataset,num_workers=hparams.num_workers, batch_size=512,  
      pin_memory=True, shuffle=False, drop_last=False, persistent_workers=True)

    sub_logdir = join(hparams.logdir, 'test')
    os.mkdir(sub_logdir)
    
    tmp_hparams = OmegaConf.create(OmegaConf.to_container(hparams, resolve=True))
    tmp_hparams.checkpoint = None
    experiment = DownstreamExperiment(tmp_hparams, wandb_run, sub_logdir)

    test_results = experiment.test(test_loader, hparams.checkpoint)
    df = pd.DataFrame([test_results])
    df.to_csv(join(sub_logdir, 'test_results.csv'), index=False)