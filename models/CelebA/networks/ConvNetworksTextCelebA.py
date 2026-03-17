
import torch
import torch.nn as nn
import sys
from omegaconf import OmegaConf
from os.path import join, abspath
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../../../'))
sys.path.append(project_path)

from models.CelebA.networks.FeatureExtractorText import FeatureExtractorText
from models.CelebA.networks.FeatureCompressor import LinearFeatureCompressor
from models.CelebA.networks.DataGeneratorText import DataGeneratorText


class EncoderText(nn.Module):
    def __init__(self, flags):
        super(EncoderText, self).__init__();
        args = {'num_features': 71, 'DIM_text': 128}
        args = OmegaConf.create(args)
        self.feature_extractor = FeatureExtractorText(args, a=2.0, b=0.3)
        self.proj = nn.Linear(5*128, 32)

    def forward(self, x_text):
        h_text = self.feature_extractor(x_text);
        h_text = h_text.view(h_text.shape[0], -1)
        h_text = self.proj(h_text)
        return h_text;


class DecoderText(nn.Module):
    def __init__(self, flags):
        super(DecoderText, self).__init__();
        self.class_dim = 32
        args = {'num_features': 71, 'DIM_text': 128}
        args = OmegaConf.create(args)
        self.feature_generator = nn.Linear(self.class_dim,
                                           5*args.DIM_text, bias=True);
        self.text_generator = DataGeneratorText(args, a=2.0, b=0.3)

    def forward(self, z_content):
        z = z_content
        text_feat_hat = self.feature_generator(z);
        text_feat_hat = text_feat_hat.unsqueeze(-1);
        text_hat = self.text_generator(text_feat_hat)
        text_hat = text_hat.transpose(-2,-1);
        return text_hat


if __name__ == '__main__':
    model = EncoderText(None)
    x = torch.randn(2,256,71)
    out = model(x)
    print(out.shape)
    decoder = DecoderText(None)
    out = decoder(out)
    print(out.shape)