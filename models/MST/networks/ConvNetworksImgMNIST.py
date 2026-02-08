
import torch
import torch.nn as nn

dataSize = torch.Size([1, 28, 28])

class EncoderImg(nn.Module):
    def __init__(self, flags):
        super(EncoderImg, self).__init__()
        self.flags = flags;
        self.hidden_dim = 400;
        num_hidden_layers = 1
        proj_dim = 32

        modules = []
        modules.append(nn.Sequential(nn.Linear(784, self.hidden_dim), nn.ReLU(True)))
        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.relu = nn.ReLU();
        self.proj = nn.Linear(self.hidden_dim, proj_dim);


    def forward(self, x):
        h = x.view(*x.size()[:-3], -1);
        h = self.enc(h);
        h = h.view(h.size(0), -1);
        h = self.proj(h);
        return h



class DecoderImg(nn.Module):
    def __init__(self, flags):
        super(DecoderImg, self).__init__();
        self.flags = flags;
        self.hidden_dim = 400;
        self.class_dim = 32
        self.num_hidden_layers = 1
        modules = []

        modules.append(nn.Sequential(nn.Linear(self.class_dim, self.hidden_dim), nn.ReLU(True)))

        modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
                        for _ in range(self.num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc3 = nn.Linear(self.hidden_dim, 784)
        self.relu = nn.ReLU();
        self.sigmoid = nn.Sigmoid();

    def forward(self, class_latent_space):
        z = class_latent_space;
        x_hat = self.dec(z);
        x_hat = self.fc3(x_hat);
        x_hat = self.sigmoid(x_hat)
        x_hat = x_hat.view(*z.size()[:-1], *dataSize)
        return x_hat


if __name__ == '__main__':
    model = EncoderImg(flags=None)
    x = torch.randn(3, 1, 28, 28)
    z = model(x)
    print(z.shape)
    decoder = DecoderImg(flags=None)
    x_hat = decoder(z)
    print(x_hat.shape)