import numpy as np
import argparse
import torch
import os
import glob
import typing

from os.path import join
import pandas as pd
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import datasets, transforms
from PIL import Image


class PolyMNISTDataset(Dataset):
    """Multimodal MNIST Dataset."""
    def __init__(self, unimodal_datapaths:str, data_base:str, missing_path:str, transform=None, target_transform=None, low=1.0, return_idx=False):
        """
            Args: unimodal_datapaths (list): list of paths to weakly-supervised unimodal datasets with samples that
                    correspond by index. Therefore the numbers of samples of all datapaths should match.
                transform: tranforms on colored MNIST digits.
                target_transform: transforms on labels.
                low: if use low-ratio data regime
        """
        super().__init__()
        unimodal_datapaths = [join(unimodal_datapaths, f"m{i}") for i in range(5)]
        self.num_modalities = len(unimodal_datapaths)
        self.unimodal_datapaths = unimodal_datapaths
        self.transform = transform
        self.target_transform = target_transform
        self.low = low

        # save all paths to individual files
        self.file_paths = {dp: [] for dp in self.unimodal_datapaths}
        for dp in unimodal_datapaths:
            files = glob.glob(os.path.join(dp, "*.png"))
            self.file_paths[dp] = files
        # assert that each modality has the same number of images
        num_files = len(self.file_paths[dp])
        for files in self.file_paths.values():
            assert len(files) == num_files
        self.num_files = num_files

        missing_path = join(data_base, 'missing_modality', missing_path+'.csv')
        self.missing_mask_data = pd.read_csv(missing_path, header=None)
        self.missing_mask_data = self.missing_mask_data.to_numpy()
        print(f"Missing mask data loaded from {missing_path}")
        print('Missing mask example:', self.missing_mask_data[0])
        assert len(self.missing_mask_data) == self.num_files, f"{len(self.missing_mask_data)} in {missing_path} != {self.num_files} in {unimodal_datapaths[0]}"

        if self.low < 1.0:
            original_num_files = self.num_files
            self.num_files = int(self.num_files * self.low)
            self.file_paths = {dp: self.file_paths[dp][:self.num_files] for dp in self.unimodal_datapaths}
            self.missing_mask_data = self.missing_mask_data[:self.num_files]
            print(f"Using {self.low} data regime, num_files: {self.num_files}/{original_num_files}")
        self.return_idx = return_idx


    @staticmethod
    def _create_mmnist_dataset(savepath, backgroundimagepath, num_modalities, train):
        """Created the Multimodal MNIST Dataset under 'savepath' given a directory of background images.
        
            Args:
                savepath (str): path to directory that the dataset will be written to. Will be created if it does not
                    exist.
                backgroundimagepath (str): path to a directory filled with background images. One background images is
                    used per modality.
                num_modalities (int): number of modalities to create.
                train (bool): create the dataset based on MNIST training (True) or test data (False).
        
        """

        # load MNIST data
        mnist = datasets.MNIST("/tmp", train=train, download=True, transform=None)

        # load background images
        background_filepaths = sorted(glob.glob(os.path.join(backgroundimagepath, "*.jpg")))  # TODO: handle more filetypes
        print("\nbackground_filepaths:\n", background_filepaths, "\n")
        if num_modalities > len(background_filepaths):
            raise ValueError("Number of background images must be larger or equal to number of modalities")
        background_images = [Image.open(fp) for fp in background_filepaths]

        # create the folder structure: savepath/m{1..num_modalities}
        for m in range(num_modalities):
            unimodal_path = os.path.join(savepath, "m%d" % m)
            if not os.path.exists(unimodal_path):
                os.makedirs(unimodal_path)
                print("Created directory", unimodal_path)

        # create random pairing of images with the same digit label, add background image, and save to disk
        cnt = 0
        for digit in range(10):
            ixs = (mnist.targets == digit).nonzero()
            for m in range(num_modalities):
                ixs_perm = ixs[torch.randperm(len(ixs))]  # one permutation per modality and digit label
                for i, ix in enumerate(ixs_perm):
                    # add background image
                    new_img = PolyMNISTDataset._add_background_image(background_images[m], mnist.data[ix])
                    # save as png
                    filepath = os.path.join(savepath, "m%d/%d.%d.png" % (m, i, digit))
                    save_image(new_img, filepath)
                    # log the progress
                    cnt += 1
                    if cnt % 10000 == 0:
                        print("Saved %d/%d images to %s" % (cnt, len(mnist)*num_modalities, savepath))
        assert cnt == len(mnist) * num_modalities

    @staticmethod
    def _add_background_image(background_image_pil, mnist_image_tensor, change_colors=False):

        # binarize mnist image
        img_binarized = (mnist_image_tensor > 128).type(torch.bool)  # NOTE: mnist is _not_ normalized to [0, 1]

        # squeeze away color channel
        if img_binarized.ndimension() == 2:
            pass
        elif img_binarized.ndimension() == 3:
            img_binarized = img_binarized.squeeze(0)
        else:
            raise ValueError("Unexpected dimensionality of MNIST image:", img_binarized.shape)
 
        # add background image
        x_c = np.random.randint(0, background_image_pil.size[0] - 28)
        y_c = np.random.randint(0, background_image_pil.size[1] - 28)
        new_img = background_image_pil.crop((x_c, y_c, x_c + 28, y_c + 28))
        # Convert the image to float between 0 and 1
        new_img = transforms.ToTensor()(new_img)
        if change_colors:  # Change color distribution
            for j in range(3):
                new_img[:, :, j] = (new_img[:, :, j] + np.random.uniform(0, 1)) / 2.0
        # Invert the colors at the location of the number
        new_img[:, img_binarized] = 1 - new_img[:, img_binarized]

        return new_img

    def __getitem__(self, index):
        """
        Returns a tuple (images, labels) where each element is a list of
        length `self.num_modalities`.
        """
        files = [self.file_paths[dp][index] for dp in self.unimodal_datapaths]
        images = [Image.open(files[m]) for m in range(self.num_modalities)]
        labels = [int(files[m].split(".")[-2]) for m in range(self.num_modalities)]
        assert all([files[m][-7:] == files[0][-7:] for m in range(1, self.num_modalities)])  # all labels should be the same

        # transforms
        if self.transform:
            images = [self.transform(img) for img in images]
        if self.target_transform:
            labels = [self.target_transform(label) for label in labels]
        
        mask = torch.from_numpy(self.missing_mask_data[index])

        images_dict = {"m%d" % m: images[m] for m in range(self.num_modalities)}

        if self.return_idx:
            images_dict['index'] = torch.LongTensor([index])  # add index to the sample
        return images_dict, mask, torch.tensor(labels[0], dtype=torch.long)  # NOTE: for MMNIST, labels are shared across modalities, so can take one value

    def __len__(self):
        return self.num_files


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    DIR_DATA = "/bigdata/siyi/data/MoPoE/PolyMNIST"
    unimodal_train_paths = f"{DIR_DATA}/train"
    train_data = PolyMNISTDataset(unimodal_train_paths, transform=transform, data_base=DIR_DATA, missing_path="missing_train_whole_1_2_3_4", low=0.1)
    sample, mask, label = train_data[0]
    print(sample['m0'].min(), sample['m0'].max())
