import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class PadPrompter(nn.Module): # 默认：69840参数
    def __init__(self, image_size=224, prompt_size=30):
        super(PadPrompter, self).__init__()
        pad_size = prompt_size
        image_size = image_size
        
        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


class FixedPatchPrompter(nn.Module): # 默认：30000参数
    def __init__(self, image_size=224, prompt_size=100):
        super(FixedPatchPrompter, self).__init__()
        self.isize = image_size
        self.psize = prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module): # 默认：187500参数
    def __init__(self, image_size=224, prompt_size=250):
        super(RandomPatchPrompter, self).__init__()
        self.isize = image_size
        self.psize = prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt


class VisualPrompter(nn.Module):
    def __init__(self, method, prompt_size=30, image_size=224, device='cuda'):
        super(VisualPrompter, self).__init__()
        
        self.prompters_dict = {
            'padding': PadPrompter,
            'random_patch': RandomPatchPrompter,
            'fixed_patch': FixedPatchPrompter
        }
        if method not in self.prompters_dict:
            raise ValueError(f"Unknown prompting method: {method}, available methods: {list(self.prompters_dict.keys())}")
        
        self.prompter = self.prompters_dict[method](image_size=image_size, prompt_size=prompt_size).to(device)
    
    def forward(self, images):
        return self.prompter(images)
    

class LoR_VP(nn.Module):
    def __init__(self, rank, image_size=224, init_methods=['zero','random'], normalize=None):
        super(LoR_VP, self).__init__()
        self.normalize=normalize
        self.left_bar = torch.nn.Parameter(torch.empty(3, image_size, rank)) # B in ori paper
        self.get_init(init_methods[0], self.left_bar)
        self.right_bar = torch.nn.Parameter(torch.empty(3, rank, image_size)) # A in ori paper
        self.get_init(init_methods[1], self.right_bar)
        self.program = torch.bmm(self.left_bar, self.right_bar)

    def get_init(self, init_method, params):
        if init_method == 'zero':
            params.data.fill_(0)
        elif init_method == 'random':
            params.data.normal_(0, 1)
        elif init_method == 'xavier':
            torch.nn.init.xavier_uniform_(params)
        elif init_method == 'kaiming':
            torch.nn.init.kaiming_uniform_(params, nonlinearity='relu')
        elif init_method == 'uniform':
            torch.nn.init.uniform_(params, a=-0.1, b=0.1)
        elif init_method == 'normal':
            torch.nn.init.normal_(params, mean=0.0, std=0.01)

    def forward(self, x):
        self.program = torch.bmm(self.left_bar, self.right_bar)
        x = x + self.program
        if self.normalize is not None:
            x = self.normalize(x)
        return x, self.program
    
class Co_LoR_VP(nn.Module):
    def __init__(self, rank, image_size=224, text_dim=512, init_methods=['zero','random'], normalize=None):
        super(Co_LoR_VP, self).__init__()
        self.normalize=normalize
        self.text_dim = text_dim
        self.left_bar = torch.nn.Parameter(torch.empty(3, image_size, rank)) # B in ori paper
        self.get_init(init_methods[0], self.left_bar)
        self.right_bar = torch.nn.Parameter(torch.empty(3, rank, image_size)) # A in ori paper
        self.get_init(init_methods[1], self.right_bar)
        self.program = torch.bmm(self.left_bar, self.right_bar)
        
        self.meta_net_left = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(text_dim, text_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(text_dim // 16, image_size))
        ]))
        self.meta_net_right = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(text_dim, text_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(text_dim // 16, image_size))
        ]))

    def get_init(self, init_method, params):
        if init_method == 'zero':
            params.data.fill_(0)
        elif init_method == 'random':
            params.data.normal_(0, 1)
        elif init_method == 'xavier':
            torch.nn.init.xavier_uniform_(params)
        elif init_method == 'kaiming':
            torch.nn.init.kaiming_uniform_(params, nonlinearity='relu')
        elif init_method == 'uniform':
            torch.nn.init.uniform_(params, a=-0.1, b=0.1)
        elif init_method == 'normal':
            torch.nn.init.normal_(params, mean=0.0, std=0.01)

    def forward(self, x, text_features):
        # device = x.device
        batch_size = x.size(0)
        
        if text_features is not None:
            if text_features.dim() == 1:
                # covp: (text_dim,) -> (batch_size, text_dim)
                text_features = text_features.unsqueeze(0).expand(batch_size, -1)
            left_bias = self.meta_net_left(text_features).view(batch_size, 1, -1, 1) # (N, 1, image_size, 1)
            right_bias = self.meta_net_right(text_features).view(batch_size, 1, 1, -1) # (N, 1, 1, image_size)
            
            self.program = torch.matmul(
            self.left_bar.expand(batch_size, -1, -1, -1) + left_bias,    # (N, 3, image_size, rank)
            self.right_bar.expand(batch_size, -1, -1, -1) + right_bias   # (N, 3, rank, image_size)
        ) # (N, 3, image_size, image_size)
        
        else:
            # No text features, non-conditional
            self.program = torch.matmul(
            self.left_bar.expand(batch_size, -1, -1, -1),
            self.right_bar.expand(batch_size, -1, -1, -1)
        )
        x = x + self.program
        if self.normalize is not None:
            x = self.normalize(x)
        return x, self.program
