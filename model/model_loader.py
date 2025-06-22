import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import Parameter
from model.ortho_hash import OrthoHash
from model.clip_ori import clip
from model.cocoop import TextEncoder
from model.visual_prompting import LoR_VP, Co_LoR_VP

class BaseWrapper(nn.Module):
    """Base wrapper for different model architectures"""
    def __init__(self, backbone, embedding_dim, code_length, num_classes, logit_scale, backbone_type='default', device='cpu', knn=3,**hash_kwargs):
        super(BaseWrapper, self).__init__()
        self.backbone = backbone
        self.hash_layer = OrthoHash(
            embedding_dim=embedding_dim,
            code_length=code_length,
            num_classes=num_classes,
            **hash_kwargs
        )
        self.backbone_type = backbone_type
        self.knn=knn
        
        # 添加文本编码相关属性（'zero_shot'和'vp'，cocoop相关的从cocoop动态获取）
        self.text_features = None
        
        # 添加获取logit_scale的属性，之后要拓展
        self.logit_scale = logit_scale
        
        # 添加visual prompting模块，之后要拓展
        if self.backbone_type in ['vp', 'cocoop_vp']:
            self.visual_prompter = LoR_VP(rank=8)
        elif self.backbone_type in ['covp', 'cocoop_covp']:
            self.visual_prompter = Co_LoR_VP(rank=self.knn, text_dim=embedding_dim)
        else:
            self.visual_prompter = None
        
    def forward(self, x, x_aug=None):
        if self.backbone_type == 'cocoop':  # cocoop
            outputs = self.backbone(x)
            image_features = outputs['image_features']
            # clip_logits = outputs['logits']
            text_features = outputs['text_features']
            code_logits, logits = self.hash_layer(image_features)
            return code_logits, logits, (image_features, text_features)
        
        elif self.backbone_type == 'cocoop_vp': # cocoop_vp
            _, visual_prompts = self.visual_prompter(x)
            outputs = self.backbone(x + visual_prompts)
            image_features = outputs['image_features']
            # clip_logits = outputs['logits']
            text_features = outputs['text_features']
            if x_aug is not None:
                num_aug = x_aug.size(0) // x.size(0)
                x_all = torch.cat([x, x_aug], dim=0)
                # 复制visual prompts，但要注意维度和covp不同
                visual_prompts_all = visual_prompts.expand(x_all.size(0), -1, -1, -1)
                x_all = x_all + visual_prompts_all
                image_features_all = self.backbone.model.image_encoder(x_all)
                image_features_all = image_features_all / image_features_all.norm(dim=-1, keepdim=True)
                text_features_all = torch.cat([text_features] * (num_aug+1), dim=0) # 复制text features
                code_logits_all, logits_all = self.hash_layer(image_features_all)
                return code_logits_all, logits_all, (image_features_all, text_features_all)
            else:
                code_logits, logits = self.hash_layer(image_features)
                return code_logits, logits, (image_features, text_features)
            
        elif self.backbone_type == 'covp':
            # x = self.visual_prompter(x, None)
            image_features = self.backbone(x)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # (batch_size, embed_dim)
            text_features = self.text_features.to(x.device) # (num_classes, embed_dim)
            similarity = torch.matmul(image_features, text_features.t())  # (batch_size, num_classes)
            
            # 最大相似度对应的文本特征
            _, max_sim_indices = similarity.max(dim=1) # (batch_size,)
            matched_text_features = text_features[max_sim_indices]
                        
            x_prompted, _ = self.visual_prompter(x, matched_text_features)
            # 重新通过backbone获取新的image features
            image_features_new = self.backbone(x_prompted)
            code_logits, logits = self.hash_layer(image_features_new)
            return code_logits, logits, (image_features_new, text_features)
        
        elif self.backbone_type == 'cocoop_covp': # cocoop_covp, and final version (TO BE DONE)
            # 3-stage mutual prompting pipeline
            # 1. image prompts text, 通过cocoop从原始image的image_features获取对应的text_features
            outputs = self.backbone(x)
            image_features = outputs['image_features'] # (batch_size, embed_dim)
            text_features = outputs['text_features'] # (batch_size, num_classes, embed_dim)，和covp不同！
            
            # 2. text prompts image, 通过cocp从text_features计算相似度，获取最匹配的text_features进行conditioned操作
            similarity = torch.bmm(
                image_features.unsqueeze(1), # (batch_size, 1, embed_dim)
                text_features.transpose(2, 1), # (batch_size, embed_dim, num_classes)
                ).squeeze(1) # (batch_size, num_classes)
            # 最大相似度对应的文本特征
            _, max_sim_indices = similarity.max(dim=1) # (batch_size,)
            matched_text_features = text_features[torch.arange(image_features.size(0)), max_sim_indices]
            
            # 3. conditioned visual prompting，冻结text prompt，获取新的image_features
            if x_aug is not None:
                num_aug = x_aug.size(0) // x.size(0)
                _, visual_prompts = self.visual_prompter(x, matched_text_features)
                x_all = torch.cat([x, x_aug], dim=0)
                visual_prompts_all = torch.cat([visual_prompts] * (num_aug+1), dim=0)  # 复制visual prompts
                x_all = x_all + visual_prompts_all
                image_features_new_all = self.backbone.model.image_encoder(x_all)
                image_features_new_all = image_features_new_all / image_features_new_all.norm(dim=-1, keepdim=True)
                text_features_all = torch.cat([text_features] * (num_aug+1), dim=0) # 复制text features
                code_logits_all, logits_all = self.hash_layer(image_features_new_all)
                return code_logits_all, logits_all, (image_features_new_all, text_features_all)
            else: 
                x_prompted, _ = self.visual_prompter(x, matched_text_features)
                # image_features_new = self.backbone(x_prompted) # 备用方案：两次cocoop
                image_features_new = self.backbone.model.image_encoder(x_prompted) # 不更新text prompt，只更新image prompt
                image_features_new = image_features_new / image_features_new.norm(dim=-1, keepdim=True)
                
                # hash_layers
                code_logits, logits = self.hash_layer(image_features_new)
                return code_logits, logits, (image_features_new, text_features)
        
        else: # zero_shot, vp, default
            if self.backbone_type == 'vp':
                x = self.visual_prompter(x)
            image_features = self.backbone(x) # 对于卷积模型，维度为(N, C, H, W)，对于全连接模型，维度为(N, C)
            if image_features.dim() == 4: # 卷积模型
                image_features = image_features.flatten(1)
            code_logits, logits = self.hash_layer(image_features)
            return code_logits, logits, (image_features, self.text_features)
    

def load_clip_model(clip_variant: str,
                   code_length: int,
                   num_classes: int,
                   clip_model_type: str = "ViT-B/16", # 在这里调整backbone
                   freeze_backbone: bool = True,
                   device: str = 'cpu',
                   class_names: list = None,
                   knn: int = 3,
                   **hash_kwargs) -> nn.Module:
    """
    Load a CLIP-based model with hashing layer.
    
    Args:
        clip_variant: Type of CLIP implementation ('zero_shot', 'cocoop', etc.)
        code_length: Length of the hash code
        num_classes: Number of classes for classification
        clip_model_type: CLIP model variant to use if backbone_name is 'clip'
        freeze_backbone: Whether to freeze the backbone parameters
        device: Device to use ('cpu' or 'cuda')
        **hash_kwargs: Additional arguments for the hashing layer
    """
    
    clip_model, _ = clip.load(clip_model_type, device=device, mode=clip_variant)
    embedding_dim = clip_model.visual.output_dim
    logit_scale = clip_model.logit_scale

    if clip_variant in ['zero_shot', 'vp', 'covp']:
        backbone = clip_model.visual
        model = BaseWrapper(backbone, embedding_dim, code_length, num_classes, logit_scale, backbone_type=clip_variant, **hash_kwargs)
        # 添加文本编码功能
        if class_names is not None:
            # 固定prompt
            prompts = [f"a photo of a {name.replace('_', ' ')}" for name in class_names] # TODO，在使用cocoop时也可以保存这个prompt，以进行正则化，防止文本空间过度偏移
            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
            text_encoder = TextEncoder(clip_model).to(device)
            # 提前计算特征并缓存
            with torch.no_grad():
                text_embeddings = clip_model.token_embedding(tokenized_prompts)
                text_features = text_encoder(text_embeddings, tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                model.text_features = text_features           
    elif clip_variant in ['cocoop', 'cocoop_vp', 'cocoop_covp']: # 暂时不调超参
        from model.cocoop import CoCoOp
        backbone = CoCoOp(class_names, clip_model, n_ctx=4, ctx_init="a photo of a", device=device, prec='amp')
        model = BaseWrapper(backbone, embedding_dim, code_length, num_classes, logit_scale, backbone_type=clip_variant,knn=knn, **hash_kwargs)
    else:
        raise NotImplementedError(f"CLIP variant {clip_variant} not implemented yet")

    
    if freeze_backbone: # freeze the backbone parameters
        if clip_variant == 'zero_shot' or clip_variant == 'vp' or clip_variant == 'covp':
            for param in model.backbone.parameters():
                param.requires_grad = False
        elif 'cocoop' in clip_variant:
            for param in backbone.model.image_encoder.parameters():
                param.requires_grad = False
            for param in backbone.model.text_encoder.parameters():
                param.requires_grad = False
            backbone.model.prompt_learner.token_prefix.requires_grad = False
            backbone.model.prompt_learner.token_suffix.requires_grad = False
        else:
            raise NotImplementedError(f"CLIP variant {clip_variant} not implemented yet")

    # activate other parameters
    if clip_variant.startswith('cocoop'):
        backbone.model.prompt_learner.ctx.requires_grad = True
        for param in backbone.model.prompt_learner.meta_net.parameters():
            param.requires_grad = True
    if clip_variant == 'cocoop_vp' or clip_variant == 'vp' or clip_variant == 'covp':
        for param in model.visual_prompter.parameters():
            param.requires_grad = True
    for param in model.hash_layer.parameters():
        param.requires_grad = True
    return model


def load_model(backbone_name: str,
               code_length: int,
               num_classes: int,
               pretrained: bool = True,
               freeze_backbone: bool = True,
               class_names: list = None,
               knn: int = 3,
               **hash_kwargs) -> nn.Module:
    """
    Load a model with specified backbone and attach hashing layer.
    
    Args:
        backbone_name: Name of the backbone model ('clip', 'resnet50', 'vit', etc.)
        code_length: Length of the hash code
        num_classes: Number of classes for classification
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze the backbone parameters
        class_names: List of class names for classification (CoCoOp)
        **hash_kwargs: Additional arguments for the hashing layer
    """
    if backbone_name.startswith('clip'):
        parts = backbone_name.split('_', 1)
        if len(parts) != 1:
            clip_variant = parts[1] # 获取variant部分
        else:
            clip_variant = 'zero_shot' # backbone_name='clip'时，默认为zero_shot
            
        return load_clip_model(
            clip_variant=clip_variant,
            code_length=code_length,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            class_names=class_names,
            knn=knn,
            **hash_kwargs
        )

    embedding_dims = {
        'alexnet': 4096,
        'vgg16': 4096,
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'vit_b_16': 768,
        'vit_b_32': 768,
        'vit_l_16': 1024,
    }

    if backbone_name.startswith('vit'):
        import timm
        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0
        )
        
    elif backbone_name.startswith('resnet'):
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        
    elif backbone_name in ['alexnet', 'vgg16']:
        backbone = getattr(models, backbone_name)(pretrained=pretrained)
        if backbone_name == 'alexnet':
            backbone.classifier = backbone.classifier[:-2]
        else:
            backbone.classifier = backbone.classifier[:-3]
            
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    embedding_dim = embedding_dims[backbone_name]
    model = BaseWrapper(backbone, embedding_dim, code_length, num_classes, backbone_type='default', **hash_kwargs)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        for param in model.hash_layer.parameters():
            param.requires_grad = True

    return model
