import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from .clip_ori import clip
from .clip_ori.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


# 包括prompt的Text Encoder
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final # LayerNorm
        self.text_projection = clip_model.text_projection # 调整维度的线性层
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype) # (batch_size, n_ctx, transformer.width)
        x = x.permute(1, 0, 2)  # NLD -> LND, (n_ctx, batch_size, transformer.width)
        x = self.transformer(x) # (n_ctx, batch_size, transformer.width)
        x = x.permute(1, 0, 2)  # LND -> NLD, (batch_size, n_ctx, transformer.width)
        x = self.ln_final(x).type(self.dtype) # (batch_size, n_ctx, transformer.width)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # 取EOT token的特征，投影
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # (batch_size, n_ctx, dim_text)

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=8, ctx_init=None, prec="fp32"):
        super().__init__()
        n_cls = len(classnames) # 类别数
        # n_ctx = cfg.TRAINER.COCOOP.N_CTX # 上下文词数
        # ctx_init = cfg.TRAINER.COCOOP.CTX_INIT # 上下文初始化方式
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0] # 上下文向量维度
        vis_dim = clip_model.visual.output_dim # 图像特征维度
        clip_imsize = clip_model.visual.input_resolution # 图像尺寸

        # 两种初始化方式：一种是给定上下文词("This is a [CLS]")，一种是随机初始化
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors) # (n_ctx, ctx_dim)
        self.prec = prec

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        if self.prec == "fp16":
            self.meta_net.half()

        # Prepare token vectors for class names
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn), n_tkn = n_ctx + len(name) + 1，仅仅用于定位EOT token！
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # (n_cls, n_tkn, ctx_dim)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # 缓存的token vectors，前缀和后缀，不需要更新
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    # 构造提示词，使用给定的上下文向量、前缀和后缀，将它们合并成一个完整的提示
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts # (dim0, n_tkn, ctx_dim)

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        
        if self.prec == "fp16":
            im_features = im_features.half()
        bias = self.meta_net(im_features)  # (batch_size, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch_size, 1, ctx_dim)
        # if self.prec == "fp16":
        #     bias = bias.float()
            
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch_size, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        # COCOOP的核心：为每个类别生成一个提示！
        prompts = []
        for ctx_shifted_i in ctx_shifted: # ctx_shifted: (n_ctx, ctx_dim)
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1) # (n_cls, n_ctx, ctx_dim)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts # (batch_size, n_cls, n_tkn, ctx_dim)


class CustomCLIP(nn.Module):
    def __init__(self,  classnames, clip_model, n_ctx=8, ctx_init=None, prec="fp32"):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, n_ctx,ctx_init, prec)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prec = prec
        
        # 梯度检查，节约memory
        if hasattr(self.text_encoder.transformer, "gradient_checkpointing"):
            self.text_encoder.transformer.gradient_checkpointing = True
        if hasattr(self.image_encoder, "gradient_checkpointing"): 
            self.image_encoder.gradient_checkpointing = True

    def forward(self, image):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (batch_size, image_dim)

        prompts = self.prompt_learner(image_features) # (batch_size, n_cls, n_tkn, ctx_dim)
        
        # logits = []
        total_text_features = []
        for pts_i, imf_i in zip(prompts, image_features): # pts_i: (n_cls, n_tkn, ctx_dim), imf_i: (image_dim, )
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True) # (n_cls, dim_text), dim_text = image_dim
            # l_i = logit_scale * imf_i @ text_features.t() 
            # logits.append(l_i)
            total_text_features.append(text_features)
            
        # logits = torch.stack(logits) # (batch_size, n_cls)
        total_text_features = torch.stack(total_text_features) # (batch_size, n_cls, dim_text)
        
        return {
            'image_features': image_features,
            'text_features': total_text_features,
            # 'logits': logits
        }
        
        # return logits, image_features # 前者用于训练，后者用于推理


class CoCoOp(nn.Module):
    def __init__(self, 
                 classnames,
                 clip_model,
                 n_ctx=4,                  # prompt context length
                 ctx_init=None,             # initial context tokens
                 prec="fp32",               # precision: fp16, fp32, amp
                 device="cuda"):
        super().__init__()
        self.device = device
        self.prec = prec
        self.scaler = GradScaler() if prec == "amp" else None # 节约memory
        
        if prec == "fp32" or prec == "amp":
            clip_model.float()
            
        self.model = CustomCLIP(
            classnames=classnames,
            clip_model=clip_model,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            prec=prec,
        )
        
        self.model.to(device)
        if device == "cuda" and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            
    def forward(self, image):
        # 节约memory
        if self.prec == "amp":
            with autocast():
                return self.model(image)
        return self.model(image)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        
        # Filter fixed tokens
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]
        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]
            
        self.model.load_state_dict(state_dict, strict=False)
    
    @torch.no_grad()    
    def get_image_features(self, image):
        image_features = self.model.image_encoder(image)
        return image_features / image_features.norm(dim=-1, keepdim=True)
        
    @torch.no_grad()
    def classify(self, image):
        logits, _ = self.forward(image)
        return logits.softmax(dim=-1)
