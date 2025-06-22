import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import Parameter
from utils import generate_hadamard_matrix, is_power_of_two

class CosSim(nn.Module):
    def __init__(self, code_length, num_classes, codebook_mode, if_trainable=True):
        super(CosSim, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.codebook_mode = codebook_mode # 初始化codebook的模式
        self.if_trainable = if_trainable # 是否从数据中学习类中心

        if self.codebook_mode == 'hadamard' and is_power_of_two(self.code_length):
            codebook = generate_hadamard_matrix(self.code_length)[:self.num_classes] # TODO，这里维度错误了
        elif self.codebook_mode == 'bernoulli':
            codebook = torch.bernoulli(torch.ones(self.num_classes, self.code_length) * 0.5) * 2 - 1
        else:
            codebook = torch.randn(self.num_classes, self.code_length) # 论文的默认操作，TODO，论文说是Bern(0.5)的初始化，但实际上却采用了N(0,1)...不知道有没有影响！

        # 可训练的类中心code
        self.centroids = nn.Parameter(codebook.clone())
        if not if_trainable:
            self.centroids.requires_grad_(False) # 为false时不更新参数

    def forward(self, x):
        # L2归一化
        norm_features = F.normalize(x, p=2, dim=-1)
        norm_centroids = F.normalize(self.centroids, p=2, dim=-1)

        # 计算余弦相似度并softmax
        logits = torch.matmul(norm_features, norm_centroids.T)
        logits = nn.Softmax(dim=-1)(logits)

        return logits


class OrthoHash(nn.Module):
    def __init__(self,
                 embedding_dim: int, code_length: int,
                 num_classes: int,
                 codebook_mode = 'bernoulli', # 初始化codebook模式
                 if_ce = True, # 是否要ce
                 if_trainable_ce = False, # 是否训练ce中参数，论文里默认为False！
                 ):
        super(OrthoHash, self).__init__()

        # 定义一些属性
        self.embedding_dim = embedding_dim
        self.code_length = code_length
        self.num_classes = num_classes

        # 将特征向量投影到码长的latent layer，结构为fcn（这里的bias是按论文源码去的，还没仔细研究，TODO）
        self.latent_layer = nn.Sequential(
            # nn.ReLU(), # LOG添加了激活函数，我认为不一定需要添加？
            nn.Linear(self.embedding_dim, self.code_length, bias=False), # (N, dim)->(N, K)，bias是论文里设置的，TODO！
        )

        # batch norm层
        self.bn = nn.BatchNorm1d(self.code_length, momentum=0.1) # (N, K)

        # OrthoHashing的核心: CosSim层，(N, K)->(N,num_classes)
        if if_ce:
            self.ce_layer = CosSim(self.code_length, self.num_classes, codebook_mode, if_trainable=if_trainable_ce)
        else:
            # 不选择的话，就是简单的线性层
            self.ce_layer = nn.Linear(self.code_length, self.num_classes)

        # self.tanh = nn.Tanh() # 由于loss的优化，tanh不是必需的，backup plan
        self._initialize_weights() # 初始化权重

    def _initialize_weights(self):
        """初始化权重"""
        nn.init.normal_(self.latent_layer[0].weight, std=0.01)
        # nn.init.zeros_(self.hash_fc.bias)

    def forward(self, x):
        x = self.latent_layer(x)
        code_logits = self.bn(x) # 生成的连续code，只需要量化就能转化为最终code
        logits = self.ce_layer(code_logits) # 分类logits，用于优化模型！
        return code_logits, logits
