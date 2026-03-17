
import torch
import torch.nn as nn
import json

alphabet = json.load(open('/home/siyi/project/mm/mul_foundation/MoPoE-main/alphabet.json', 'r'))

class FeatureEncText(nn.Module):
    def __init__(self, dim, num_features):
        super(FeatureEncText, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv1d(num_features, 2*self.dim, kernel_size=1);
        self.conv2 = nn.Conv1d(2*self.dim, 2*self.dim, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv5 = nn.Conv1d(2*self.dim, 2*self.dim, kernel_size=4, stride=2, padding=0, dilation=1);
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(-2,-1);
        out = self.conv1(x);
        out = self.relu(out);
        out = self.conv2(out);
        out = self.relu(out);
        out = self.conv5(out);
        out = self.relu(out);
        h = out.view(-1, 2*self.dim)
        return h;


class EncoderText(nn.Module):
    def __init__(self, flags, num_features):
        super(EncoderText, self).__init__()
        self.flags = flags
        dim = 64
        proj_dim = 32
        num_features = num_features
        self.text_feature_enc = FeatureEncText(dim, num_features);
        self.proj = nn.Linear(2 * dim, proj_dim);

    def forward(self, x):
        h = self.text_feature_enc(x);
        h = self.proj(h);
        return h


class DecoderText(nn.Module):
    def __init__(self, flags, num_features):
        super(DecoderText, self, ).__init__()
        self.flags = flags;
        self.class_dim = 32
        self.dim = 64
        self.linear = nn.Linear(self.class_dim, 2*self.dim)
        self.conv1 = nn.ConvTranspose1d(2*self.dim, 2*self.dim,
                                        kernel_size=4, stride=1, padding=0, dilation=1);
        self.conv2 = nn.ConvTranspose1d(2*self.dim, 2*self.dim,
                                        kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv_last = nn.Conv1d(2*self.dim, num_features, kernel_size=1);
        self.relu = nn.ReLU()
        self.out_act = nn.LogSoftmax(dim=-2);

    def forward(self, class_latent_space):
        z = self.linear(class_latent_space)
        x_hat = z.view(z.size(0), z.size(1), 1);
        x_hat = self.conv1(x_hat);
        x_hat = self.relu(x_hat);
        x_hat = self.conv2(x_hat)
        x_hat = self.relu(x_hat);
        x_hat = self.conv_last(x_hat)
        log_prob = self.out_act(x_hat)
        log_prob = log_prob.transpose(-2,-1);
        return log_prob



if __name__ == '__main__':
    model = EncoderText(flags=None, num_features=71)
    x = torch.randn(2, 8, 71)
    out = model(x)
    print(out.shape)
    decoder = DecoderText(flags=None, num_features=71)
    out = decoder(out)
    print(out.shape)