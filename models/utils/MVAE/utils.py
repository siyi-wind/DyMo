import torch
from torch.autograd import Variable

def calc_elbo(flags, modalities, rec_weights, style_weights, modality, recs, klds):
    flags = flags;
    mods = modalities;
    s_weights = style_weights;
    r_weights = rec_weights;
    kld_content = klds['content'];
    if modality == 'joint':
        w_style_kld = 0.0;
        w_rec = 0.0;
        klds_style = klds['style']
        for k, m_key in enumerate(mods.keys()):
                w_style_kld += s_weights[m_key] * klds_style[m_key];
                w_rec += r_weights[m_key] * recs[m_key];
        kld_style = w_style_kld;
        rec_error = w_rec;
    else:
        beta_style_mod = s_weights[modality];
        #rec_weight_mod = r_weights[modality];
        rec_weight_mod = 1.0;
        kld_style = beta_style_mod * klds['style'][modality];
        rec_error = rec_weight_mod * recs[modality];
    div = flags.beta_content * kld_content + flags.beta_style * kld_style;
    elbo = rec_error + flags.beta * div;
    return elbo;


def reweight_weights(w):
    w = w / w.sum();
    return w;

def reparameterize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(torch.nn.Module):
    def __init__(self, ndims):
        super(Unflatten, self).__init__()
        self.ndims = ndims

    def forward(self, x):
        return x.view(x.size(0), *self.ndims)

def mixture_component_selection(flags, mus, logvars, w_modalities=None):
    #if not defined, take pre-defined weights
    num_components = mus.shape[0];
    num_samples = mus.shape[1];
    if w_modalities is None:
        w_modalities = torch.Tensor(flags.alpha_modalities).to(flags.device);
    idx_start = [];
    idx_end = []
    for k in range(0, num_components):
        if k == 0:
            i_start = 0;
        else:
            i_start = int(idx_end[k-1]);
        if k == w_modalities.shape[0]-1:
            i_end = num_samples;
        else:
            i_end = i_start + int(torch.floor(num_samples*w_modalities[k]));
        idx_start.append(i_start);
        idx_end.append(i_end);
    idx_end[-1] = num_samples;
    mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
    logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])]);
    return [mu_sel, logvar_sel];