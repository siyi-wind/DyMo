
import sys
import random
from typing import List, Tuple

import torch
import torch.utils.data as data

import warnings
from PIL import Image
import os
import os.path
import gzip
import numpy as np
import torch
import codecs

import sys
import sys

from os.path import abspath, dirname, join
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../'))
sys.path.append(project_path)

from utils.text import one_hot_encode
from utils.text import create_text_from_label_mnist
from utils.text import char2Index
from omegaconf import OmegaConf
import json
from torchvision import transforms
from torchvision.transforms import InterpolationMode

digit_text_german = ['null', 'eins', 'zwei', 'drei', 'vier', 'fuenf', 'sechs', 'sieben', 'acht', 'neun'];
digit_text_english = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];


def rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    torch.manual_seed(2025)
    _idx1, _idx2 = [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)


class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.dataset
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, 'transform') and self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transforms: ")
        if hasattr(self, 'target_transform') and self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class SVHNMNIST(VisionDataset):
    training_file_svhn_idx = 'train-ms-svhn-idx.pt';
    training_file_mnist_idx = 'train-ms-mnist-idx.pt';
    training_file_mnist = 'training.pt';
    training_file_svhn = 'train_32x32.mat';

    # test_file_svhn_idx = 'test-ms-mnist-idx.pt';
    # test_file_mnist_idx = 'test-ms-svhn-idx.pt';
    test_file_svhn_idx = 'test-ms-svhn-idx.pt';
    test_file_mnist_idx = 'test-ms-mnist-idx.pt';
    test_file_mnist = 'test.pt';
    test_file_svhn = 'test_32x32.mat';

    val_file_svhn_idx = 'val-ms-svhn-idx.pt';
    val_file_mnist_idx = 'val-ms-mnist-idx.pt';
    val_file_mnist = 'val.pt';
    val_file_svhn = 'val_32x32.mat';
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels_mnist(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets_mnist

    @property
    def train_labels_svhn(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets_svhn

    @property
    def test_labels_mnist(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets_mnist

    @property
    def test_labels_svhn(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets_svhn

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data_mnist

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data_mnist

    def __init__(self, flags,  alphabet, missing_path:List, train='train', transform=None, target_transform=None, return_idx=False):
        super(SVHNMNIST, self).__init__(flags.dir_data)
        self.return_idx = return_idx
        self.flags = flags;
        self.dataset = 'MNIST_SVHN';
        self.dataset_mnist = 'MNIST';
        self.dataset_svhn = 'SVHN';
        self.len_sequence = flags.len_sequence
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.alphabet = alphabet;
        self.missing_path = str(missing_path)
        if self.missing_path == 'none':
            mask = [0, 0, 0]
        else:
            mask = [str(i) in self.missing_path for i in range(3)]
        self.mask = mask
        modality_names = ['mnist', 'svhn', 'text']
        # print missing modalities mask=True
        print(f'missing mask examples: {self.mask}, ', [modality_names[i] for i in range(len(mask)) if mask[i]])

        self.dir_svhn = os.path.join(self.root, self.dataset_svhn);
        print(self.dir_svhn)

        if not self._check_exists_mnist():
            raise RuntimeError('Dataset MNIST not found.')
        if not self._check_exists_svhn():
            raise RuntimeError('Dataset SVHN not found.')


        if self.train == 'train':
            id_file_svhn = self.training_file_svhn_idx;
            id_file_mnist = self.training_file_mnist_idx;
            data_file_svhn = self.training_file_svhn;
            data_file_mnist = self.training_file_mnist;
        # for val and test, only idx files are changed
        elif self.train == 'val':
            id_file_svhn = self.val_file_svhn_idx;
            id_file_mnist = self.val_file_mnist_idx;
            data_file_svhn = self.val_file_svhn;
            data_file_mnist = self.val_file_mnist;
        elif self.train == 'test':
            id_file_svhn = self.test_file_svhn_idx;
            id_file_mnist = self.test_file_mnist_idx;
            data_file_svhn = self.test_file_svhn;
            data_file_mnist = self.test_file_mnist;          

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        print(os.path.join(self.dir_svhn, data_file_svhn))
        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.dir_svhn, data_file_svhn))

        self.data_svhn = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels_svhn = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels_svhn, self.labels_svhn == 10, 0)
        self.data_svhn = np.transpose(self.data_svhn, (3, 2, 0, 1))
        samples_svhn = self.data_svhn.shape[0];
        channels_svhn = self.data_svhn.shape[1];
        width_svhn = self.data_svhn.shape[2];
        height_svhn = self.data_svhn.shape[3];

        self.data_mnist, self.labels_mnist = torch.load(os.path.join(self.processed_folder, data_file_mnist));
        print('mnist data file:', os.path.join(self.processed_folder, data_file_mnist))

        # # get transformed indices
        self.labels_svhn = torch.LongTensor(self.labels_svhn);
        mnist_l, mnist_li = self.labels_mnist.sort()
        svhn_l, svhn_li = self.labels_svhn.sort()
        # if id_file exists, load the indices
        self.dir_mnist_svhn = os.path.join(self.root, self.dataset, 'processed')
        if os.path.exists(os.path.join(self.dir_mnist_svhn, id_file_svhn)) and \
                os.path.exists(os.path.join(self.dir_mnist_svhn, id_file_mnist)):
            print(f'Loading indices from file, {os.path.join(self.dir_mnist_svhn, id_file_svhn)} and {os.path.join(self.dir_mnist_svhn, id_file_mnist)}')
            self.mnist_idx = torch.load(os.path.join(self.dir_mnist_svhn, id_file_mnist))
            self.svhn_idx = torch.load(os.path.join(self.dir_mnist_svhn, id_file_svhn))
        else: 
            print(f'Generating indices, {os.path.join(self.dir_mnist_svhn, id_file_svhn)} and {os.path.join(self.dir_mnist_svhn, id_file_mnist)}')
            self.mnist_idx, self.svhn_idx = rand_match_on_idx(mnist_l, mnist_li,
                                                            svhn_l, svhn_li,
                                                            max_d=10000,
                                                            dm=flags.data_multiplications)
            # save the indices
            torch.save(self.mnist_idx, os.path.join(self.dir_mnist_svhn, id_file_mnist))
            torch.save(self.svhn_idx, os.path.join(self.dir_mnist_svhn, id_file_svhn))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        idx_mnist = self.mnist_idx[index];
        idx_svhn = self.svhn_idx[index];
        img_svhn, target_svhn = self.data_svhn[idx_svhn], int(self.labels_svhn[idx_svhn]);
        img_mnist, target_mnist = self.data_mnist[idx_mnist], int(self.labels_mnist[idx_mnist]);

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_svhn = Image.fromarray(np.transpose(img_svhn, (1, 2, 0)))
        img_mnist = Image.fromarray(img_mnist.numpy(), mode='L')

        if self.transform is not None:
            if self.transform[0] is not None:
                img_mnist = self.transform[0](img_mnist);
                img_svhn = self.transform[1](img_svhn);
        
        if target_mnist == target_svhn:
            text_target = create_text_from_label_mnist(self.len_sequence, target_mnist, self.alphabet)
        else:
            print(target_svhn)
            print(target_mnist)
            print('targets do not match...exit')
            sys.exit();

        if self.target_transform is not None:
            target = self.target_transform(target_mnist)
        else:
            target = target_mnist;

        batch = {'mnist': img_mnist, 'svhn': img_svhn, 'text': text_target}
        mask = torch.tensor(self.mask, dtype=bool)
        if self.return_idx:
            batch['index'] = torch.LongTensor([index])
        return batch, mask, target

    def __len__(self):
        return len(self.mnist_idx)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.dataset, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.dataset_mnist, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists_mnist(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file_mnist)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file_mnist)))

    def _check_exists_svhn(self):
        return (os.path.exists(os.path.join(self.dir_svhn,
                                            self.training_file_svhn)) and
                os.path.exists(os.path.join(self.dir_svhn,
                                            self.test_file_svhn)))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")



if __name__ == '__main__':
    flags = {'dir_data': '/bigdata/siyi/data/MoPoE/MNIST_SVHN_TEXT', 'len_sequence': 8, 'data_multiplications': 20}
    flags = OmegaConf.create(flags)
    alphabet_path = '/home/siyi/project/mm/mul_foundation/MoPoE/alphabet.json'
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Resize(size=(28, 28), interpolation=InterpolationMode.BICUBIC),
                                              transforms.ToTensor()])
    transform_svhn = transforms.Compose([transforms.ToTensor()])
    transforms_both = [transform_mnist, transform_svhn]
    
    
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    mode = 'train'
    dataset = SVHNMNIST(flags, alphabet, missing_path=12, train=mode, transform=transforms_both)
    print('dataset length:', len(dataset))
    sample, mask, target = dataset[0]
    print(sample['text'].shape)
    print('mnist min max:', sample['mnist'].min(), sample['mnist'].max())
    print('svhn min max:', sample['svhn'].min(), sample['svhn'].max())
    print('text:', sample['text'][0])
    print('mask:', mask)
    print('len of alphabet:', len(alphabet))
    print('target:', target)

