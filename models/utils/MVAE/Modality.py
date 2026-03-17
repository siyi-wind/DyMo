from abc import ABC, abstractmethod
import os
from PIL import Image
from torchvision.transforms import InterpolationMode

import torch
import torch.distributions as dist
from torchvision import transforms

class Modality(ABC):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name):
        self.name = name;
        self.encoder = enc;
        self.decoder = dec;
        self.class_dim = class_dim;
        self.style_dim = style_dim;
        self.likelihood_name = lhood_name;
        self.likelihood = self.get_likelihood(lhood_name);


    def get_likelihood(self, name):
        if name == 'laplace':
            pz = dist.Laplace;
        elif name == 'bernoulli':
            pz = dist.Bernoulli;
        elif name == 'normal':
            pz = dist.Normal;
        elif name == 'categorical':
            pz = dist.OneHotCategorical;
        else:
            print('likelihood not implemented')
            pz = None;
        return pz;

    def calc_log_prob(self, out_dist, target, norm_value):
        log_prob = out_dist.log_prob(target).sum();
        mean_val_logprob = log_prob/norm_value;
        return mean_val_logprob;



class SVHN(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name,
                 plotImgSize):
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name);
        self.data_size = torch.Size((3, 32, 32));
        self.plot_img_size = plotImgSize;
        self.transform_plot = self.get_plot_transform();
        self.gen_quality_eval = True;
        self.file_suffix = '.png';

    def get_plot_transform(self):
        transf = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(size=list(self.plot_img_size)[1:],
                                                       interpolation=InterpolationMode.BICUBIC),
                                     transforms.ToTensor()])
        return transf;


class MNIST(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name):
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name);
        self.data_size = torch.Size((1, 28, 28));
        self.gen_quality_eval = True;
        self.file_suffix = '.png';


class Text(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name,
                 len_sequence, alphabet, plotImgSize, font):
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name)
        self.alphabet = alphabet;
        self.len_sequence = len_sequence;
        self.data_size = torch.Size([len_sequence]);
        self.plot_img_size = plotImgSize;
        self.font = font;
        self.gen_quality_eval = False;
        self.file_suffix = '.txt';


class CelebAImg(Modality):
    def __init__(self, enc, dec, plotImgSize, style_dim):
        self.name = 'img';
        self.likelihood_name = 'laplace';
        self.data_size = torch.Size((3, 64, 64));
        self.plot_img_size = plotImgSize;
        # self.transform_plot = self.get_plot_transform();
        self.gen_quality_eval = True;
        self.file_suffix = '.png';
        self.encoder = enc;
        self.decoder = dec;
        self.likelihood = self.get_likelihood(self.likelihood_name);
        self.style_dim = style_dim;


class CelebAText(Modality):
    def __init__(self, enc, dec, len_sequence, alphabet, plotImgSize, font, style_dim):
        self.name = 'text';
        self.likelihood_name = 'categorical';
        self.alphabet = alphabet;
        self.len_sequence = len_sequence;
        #self.data_size = torch.Size((len(alphabet), len_sequence));
        self.data_size = torch.Size([len_sequence]);
        self.plot_img_size = plotImgSize;
        self.font = font;
        self.gen_quality_eval = False;
        self.file_suffix = '.txt';
        self.encoder = enc;
        self.decoder = dec;
        self.likelihood = self.get_likelihood(self.likelihood_name);
        self.style_dim = style_dim;