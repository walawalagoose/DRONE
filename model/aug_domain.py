import numpy as np
import torch
import torch.nn as nn
import random
from skimage.exposure import match_histograms
from utils import GradientReverseLayer

class EFDMix(nn.Module):
    """EFDMix.

    Reference:
      Zhang et al, Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization, CVPR2022
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        # self._activated = True

    def forward(self, x):
        # if not self.training or not self._activated:
        #     return x

        # if random.random() > self.p: # 默认使用
        #     return x

        B,C,W,H = x.size(0), x.size(1), x.size(2),  x.size(3)
        x_view = x.view(B,C, -1)
        ## sort input vectors.
        value_x, index_x = torch.sort(x_view)
        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)
        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        inverse_index = index_x.argsort(-1)
        x_view_copy = value_x[perm].gather(-1, inverse_index) * (1-lmda)
        new_x = x_view + (x_view_copy - x_view.detach() * (1-lmda))
        return new_x.view(B, C, W, H)
    
    
class DSU(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6, factor=1.0):
        super(DSU, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = factor

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        # if (not self.training) or (np.random.random()) > self.p:
        #     return x

        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)

        return x
    
    
class MixStyle(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix
    

class MixHistogram(nn.Module):
    """MixHistogram.

    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixHistogram.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x
        B,C,W,H = x.size(0), x.size(1), x.size(2),  x.size(3)
        ############################# mixhist via histogram matching.
        x_view = x.view(-1, W, H)
        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        #### Mixing histogram
        image1_temp = match_histograms(np.array(x_view.detach().clone().cpu().float().transpose(0,2)), np.array(x[perm].view(-1, W, H).detach().clone().cpu().float().transpose(0,2)), channel_axis=-1) # 修改，将multichannel改为channel_axis
        image1_temp = torch.from_numpy(image1_temp).float().to(x.device).transpose(0,2).view(B,C,W,H)
        return x + (image1_temp - x).detach() * (1 - lmda)
    
class AdvStyle(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random', channel_num=0, adv_weight=1.0, mix_weight=1.0, **kwargs):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True
        self.channel_num = channel_num
        self.adv_mean_std = torch.nn.Parameter(torch.FloatTensor(1,channel_num,1,1), requires_grad=True)
        self.adv_mean_std.data.fill_(0.1)  ## initialization
        self.adv_std_std = torch.nn.Parameter(torch.FloatTensor(1,channel_num,1,1), requires_grad=True)
        self.adv_std_std.data.fill_(0.1)

        self.moveavg_mean_std = torch.FloatTensor(1,channel_num,1,1).fill_(0)
        self.moveavg_std_std = torch.FloatTensor(1, channel_num, 1, 1).fill_(0)

        self.grl_mean = GradientReverseLayer()  ## fix the backpropagation weight as 1.0.
        self.grl_std = GradientReverseLayer()
        self.adv_weight = adv_weight
        self.mix_weight = mix_weight
        self.first_flag = True
        self.momentum = 0.99

    def forward(self, x):
        # if not self.training or not self._activated:
        #     return x

        # random_value = random.random()
        # if random_value < 1/2:
        #     return x
        # else:
        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_normed = (x-mu) / sig

        ### initialization the mean_std and std_std
        if self.first_flag:
            var_mu = mu.var(0, keepdim=True)
            var_sig = sig.var(0, keepdim=True)
            sig_mu = (var_mu + self.eps).sqrt()
            sig_sig = (var_sig + self.eps).sqrt()
            sig_mu = sig_mu.detach()
            sig_sig = sig_sig.detach()
            self.adv_mean_std.data = sig_mu
            self.adv_std_std.data = sig_sig
            self.first_flag = False
            ####################################### initialize std with uniform distribution.
            # self.adv_mean_std.data.uniform_(0, 1)
            # self.adv_std_std.data.uniform_(0, 1)
            # self.first_flag = False

        # initial_mean_std = torch.randn(mu.size()).cuda()
        # initial_std_std = torch.randn(sig.size()).cuda()
        # new_mu = initial_mean_std * self.grl_mean(self.adv_mean_std, self.adv_weight) + mu
        # new_sig = initial_std_std * self.grl_std(self.adv_std_std, self.adv_weight) + sig
        # return x_normed * new_sig + new_mu

        # 只用self.adv_mean_std 和 self.adv_std_std 的direction, and current batch std 的强度。
        var_mu = mu.var(0, keepdim=True) ## 1*C*1*1
        var_sig = sig.var(0, keepdim=True)
        sig_mu = (var_mu + self.eps).sqrt()  ## 这里提供强度
        sig_sig = (var_sig + self.eps).sqrt()

        qiangdu_sig_mu = self.grl_mean(self.adv_mean_std, self.adv_weight)  ## 这里提供方向
        qiangdu_sig_sig = self.grl_std(self.adv_std_std, self.adv_weight)

        used_sig_mu = qiangdu_sig_mu / torch.norm(qiangdu_sig_mu, p=2,dim=1,keepdim=True) * torch.norm(sig_mu, p=2,dim=1,keepdim=True) * self.mix_weight
        used_sig_sig = qiangdu_sig_sig / torch.norm(qiangdu_sig_sig, p=2, dim=1, keepdim=True) * torch.norm(sig_sig, p=2, dim=1,
                                                                                        keepdim=True) * self.mix_weight
        initial_mean_std = torch.randn(mu.size()).cuda()
        initial_std_std = torch.randn(sig.size()).cuda()
        new_mu = initial_mean_std * used_sig_mu + mu
        new_sig = initial_std_std * used_sig_sig + sig
        return x_normed * new_sig + new_mu
    

