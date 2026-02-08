import torch
import torch.nn as nn
import sys
from omegaconf import OmegaConf
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)
from models.CelebA.networks.FeatureExtractorImg import FeatureExtractorImg
from models.CelebA.networks.FeatureCompressor import LinearFeatureCompressor
from models.CelebA.networks.DataGeneratorImg import DataGeneratorImg

class EncoderImg(nn.Module):
    def __init__(self, flags):
        super(EncoderImg, self).__init__();
        args = {'image_channels': 3, 'DIM_img': 128}
        args = OmegaConf.create(args)
        self.feature_extractor = FeatureExtractorImg(args, a=2.0, b=0.3)
        self.proj = nn.Linear(5*128, 32)

    def forward(self, x_img):
        h_img = self.feature_extractor(x_img);
        h_img = h_img.view(h_img.shape[0], -1)
        h_img = self.proj(h_img)
        return  h_img;


class DecoderImg(nn.Module):
    def __init__(self, flags):
        super(DecoderImg, self).__init__();
        self.class_dim = 32
        self.num_layers_img = 5
        self.DIM_img = 128
        self.feature_generator = nn.Linear(self.class_dim, self.num_layers_img * self.DIM_img, bias=True);
        args = {'image_channels': 3, 'DIM_img': 128}
        args = OmegaConf.create(args)
        self.img_generator = DataGeneratorImg(args, a=2.0, b=0.3)

    def forward(self, z_content):
        z = z_content
        img_feat_hat = self.feature_generator(z);
        img_feat_hat = img_feat_hat.view(img_feat_hat.size(0), img_feat_hat.size(1), 1, 1);
        img_hat = self.img_generator(img_feat_hat)
        return img_hat


if __name__ == '__main__':
    model = EncoderImg(None)
    x = torch.randn(2,3,64,64)
    out = model(x)
    print(out.shape)
    decoder = DecoderImg(None)
    out = decoder(out)
    print(out.shape)
