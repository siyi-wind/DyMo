import os 
from os.path import join
import wandb

import numpy as np
import pandas as pd
import random
import torch
import json
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.utils import get_transforms, grab_arg_from_checkpoint
from experiments.ReconstructionExperiment import ReconstructionExperiment


def load_datasets(hparams):
    if hparams.dataset_name == 'PolyMNIST':
        from datasets.PolyMNISTDataset import PolyMNISTDataset
        image_size = grab_arg_from_checkpoint(hparams, 'image_size')
        train_dataset = PolyMNISTDataset(
            unimodal_datapaths=hparams.DATA_train, data_base=hparams.data_base, missing_path=hparams.missing_train, transform=get_transforms(image_size,hparams.target,'train'),
            target_transform=None, low=hparams.low)
        val_dataset = PolyMNISTDataset(
            unimodal_datapaths=hparams.DATA_val, data_base=hparams.data_base, missing_path=hparams.missing_val, transform=get_transforms(image_size,hparams.target,'val'),
            target_transform=None,)
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
        train_dataset = SVHNMNIST(flags, alphabet, train='train', missing_path=hparams.missing_train, transform=get_transforms(hparams.image_size, hparams.target, 'train'))
        val_dataset = SVHNMNIST(flags, alphabet, train='val', missing_path=hparams.missing_val, transform=get_transforms(hparams.image_size, hparams.target, 'val'))
        test_dataset = SVHNMNIST(flags, alphabet, train='test', missing_path=hparams.missing_test, transform=get_transforms(hparams.image_size, hparams.target, 'test'))
        hparams.alphabet = alphabet
    elif hparams.dataset_name == 'CelebA':
        from datasets.CelebADataset import CelebaDataset
        flags = {'dir_data': hparams.data_base, 'dir_text': hparams.data_base, 'len_sequence': 256, 'random_text_ordering': False, 'random_text_startindex': True}
        flags = OmegaConf.create(flags)
        alphabet_path = join(hparams.data_base, 'alphabet.json')
        with open(alphabet_path) as alphabet_file:
            alphabet = str(''.join(json.load(alphabet_file)))
        train_dataset = CelebaDataset(flags, alphabet, missing_path=hparams.missing_train, partition=0, transform=get_transforms(hparams.image_size, hparams.target, 'train'))
        val_dataset = CelebaDataset(flags, alphabet, missing_path=hparams.missing_val, partition=1, transform=get_transforms(hparams.image_size, hparams.target, 'val'))
        test_dataset = CelebaDataset(flags, alphabet, missing_path=hparams.missing_test, partition=2, transform=get_transforms(hparams.image_size, hparams.target, 'test'))
        hparams.alphabet = alphabet
    elif hparams.dataset_name == 'DVM':
        from datasets.TIPDataset import ImagingAndTabularDataset
        train_dataset = ImagingAndTabularDataset(
                    hparams.DATA_data_train_eval_imaging, hparams.delete_segmentation, hparams.augmentation_rate, 
                    hparams.DATA_data_train_eval_tabular, hparams.DATA_field_lengths_tabular, hparams.eval_one_hot,
                    hparams.DATA_labels_train_eval_imaging, hparams.image_size, hparams.live_loading, train=True, target=hparams.target,
                    corruption_rate=hparams.corruption_rate, data_base=hparams.data_base, augmentation_speedup=hparams.augmentation_speedup,
                    missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,algorithm_name=hparams.algorithm_name,
                    missing_path=hparams.missing_train)
        val_dataset = ImagingAndTabularDataset(
                    hparams.DATA_data_val_eval_imaging, hparams.delete_segmentation, hparams.augmentation_rate, 
                    hparams.DATA_data_val_eval_tabular, hparams.DATA_field_lengths_tabular, hparams.eval_one_hot,
                    hparams.DATA_labels_val_eval_imaging, hparams.image_size, hparams.live_loading, train=False, target=hparams.target,
                    corruption_rate=hparams.corruption_rate, data_base=hparams.data_base, augmentation_speedup=hparams.augmentation_speedup,
                    missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,algorithm_name=hparams.algorithm_name,
                    missing_path=hparams.missing_val)
        test_dataset = ImagingAndTabularDataset(
                hparams.DATA_data_test_eval_imaging, hparams.delete_segmentation, 0, 
                hparams.DATA_data_test_eval_tabular, hparams.DATA_field_lengths_tabular, hparams.eval_one_hot,
                hparams.DATA_labels_test_eval_imaging, hparams.image_size, hparams.live_loading, train=False, target=hparams.target,
                corruption_rate=0.0, data_base=hparams.data_base, missing_tabular=hparams.missing_tabular, missing_strategy=hparams.missing_strategy, missing_rate=hparams.missing_rate,
                augmentation_speedup=hparams.augmentation_speedup,algorithm_name=hparams.algorithm_name,
                missing_path=hparams.missing_test)
        hparams.input_size = test_dataset.get_input_size()
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset
    

def pretrain(hparams, wandb_run):
    # seed everything
    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    train_dataset, val_dataset, test_dataset = load_datasets(hparams)

    train_loader = DataLoader(train_dataset, num_workers=hparams.num_workers, batch_size=hparams.batch_size, 
                        pin_memory=True, shuffle=True, drop_last=False, persistent_workers=True)
    val_loader = DataLoader(val_dataset, num_workers=hparams.num_workers, batch_size=hparams.batch_size,
                        pin_memory=True, shuffle=False, persistent_workers=True)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    sub_logdir = join(hparams.logdir, 'pretrain')
    os.mkdir(sub_logdir)
    experiment = ReconstructionExperiment(hparams, wandb_run, sub_logdir)
    experiment.train(train_loader, val_loader)
