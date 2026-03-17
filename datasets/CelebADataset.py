from typing import List
import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image as Image

import sys
from os.path import abspath, dirname, join
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../'))
sys.path.append(project_path)

import torch
from utils import text as text
from omegaconf import OmegaConf
import json

print('current_path:', current_path)
print('project_path:', project_path)


class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, args, alphabet, missing_path: List, partition=0, transform=None, return_idx=False):

        # self.dir_dataset_base = os.path.join(args.dir_data, args.dataset)
        # use blond_hair as labels
        label_name = 'Blond_Hair'
        print('label_name:', label_name)

        self.missing_path = str(missing_path)
        if self.missing_path == 'none':
            mask = [0, 0]
        else:
            mask = [str(i) in self.missing_path for i in range(2)]
        self.mask = mask
        modality_names = ['img', 'text']
        # print missing modalities mask=True
        print(f'missing mask examples: {self.mask}, ', [modality_names[i] for i in range(len(mask)) if mask[i]])

        self.dir_dataset_base = args.dir_data

        filename_text = os.path.join(args.dir_text, 'list_attr_text_' + str(args.len_sequence).zfill(3) + '_' + str(args.random_text_ordering) + '_' + str(args.random_text_startindex) + '_celeba.csv');
        filename_partition = os.path.join(self.dir_dataset_base, f'list_eval_partition_{label_name}_balanced.csv');
        filename_attributes = os.path.join(self.dir_dataset_base, 'list_attr_celeba.csv')

        df_text = pd.read_csv(filename_text)
        df_partition = pd.read_csv(filename_partition);
        df_attributes = pd.read_csv(filename_attributes);

        self.args = args;
        self.img_dir = os.path.join(self.dir_dataset_base, 'img_align_celeba');
        self.txt_path = filename_text
        self.attrributes_path = filename_attributes;
        self.partition_path = filename_partition;

        self.alphabet = alphabet;
        self.img_names = df_text.loc[df_partition['partition'] == partition]['image_id'].values
        self.attributes = df_attributes.loc[df_partition['partition'] == partition];
        self.labels = df_attributes.loc[df_partition['partition'] == partition]; #atm, i am just using blond_hair as labels
        self.labels = self.labels[['image_id', label_name]].values
        self.y = df_text.loc[df_partition['partition'] == partition]['text'].values
        self.transform = transform
        self.return_idx = return_idx

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)
        text_str = text.one_hot_encode(self.args.len_sequence, self.alphabet, self.y[index])
        # label = torch.from_numpy((self.labels[index,1:] > 0).astype(int)).float();
        label = int(self.labels[index, 1:] > 0)
        assert self.labels[index, 0] == self.img_names[index]
        sample = {'img': img, 'text': text_str}

        mask = torch.tensor(self.mask, dtype=bool)
        if self.return_idx:
            sample['index'] = torch.LongTensor([index])
        return sample, mask, label

    def __len__(self):
        return self.y.shape[0]


    def get_text_str(self, index):
        return self.y[index];



if __name__ == "__main__":
    alphabet_path = join(project_path, 'datasets/alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    args = {'dir_data': '/bigdata/siyi/data/MoPoE/CelebA', 'dataset': 'CelebA', 'dir_text':'/bigdata/siyi/data/MoPoE/CelebA', 'len_sequence': 256, 'random_text_ordering': False, 'random_text_startindex': True}
    args = OmegaConf.create(args)
    offset_height = (218 - 148) // 2
    offset_width = (178 - 148) // 2
    crop = lambda x: x[:, offset_height:offset_height + 147,
                        offset_width:offset_width + 147]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(crop),
                                    transforms.ToPILImage(),
                                    transforms.Resize(size=(64,
                                                            64),
                                                        interpolation=Image.BICUBIC),
                                    transforms.ToTensor()])
    train_data = CelebaDataset(args, alphabet, missing_path='0', partition=1, transform=transform)
    sample, mask, label = train_data[0]
    print(label)
    print(mask)
    for key, value in sample.items():
        print(key, value.shape)