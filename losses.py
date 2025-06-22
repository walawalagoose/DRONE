import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseClassificationLoss(nn.Module):
    def __init__(self):
        super(BaseClassificationLoss, self).__init__()
        self.losses = {}

    def forward(self, logits, code_logits, labels, onehot=True):
        raise NotImplementedError

def get_imbalance_mask(sigmoid_logits, labels, nclass, threshold=0.7, imbalance_scale=-1):
    if imbalance_scale == -1:
        imbalance_scale = 1 / nclass

    mask = torch.ones_like(sigmoid_logits) * imbalance_scale

    # wan to activate the output
    mask[labels == 1] = 1

    # if predicted wrong, and not the same as labels, minimize it
    correct = (sigmoid_logits >= threshold) == (labels == 1)
    mask[~correct] = 1

    multiclass_acc = correct.float().mean()

    # the rest maintain "imbalance_scale"
    return mask, multiclass_acc


class OrthoHashLoss(BaseClassificationLoss):
    """
    OrthoHash的统一损失函数
    包含分类损失（带余弦间隔）和量化损失
    """
    def __init__(self,
                 ce_weight=1.0,
                 temperature=8.0,
                 margin=0.2,
                 margin_type='arc',  # cos/arc
                 if_multiclass=True,
                 quan_weight=0.1,
                 quan_type='cs',
                 multiclass_loss='label_smoothing',
                 **kwargs):
        super(OrthoHashLoss, self).__init__()

        self.ce_weight = ce_weight # 分类损失权重
        self.temperature = temperature  # 温度系数，用于缩放logits
        self.margin = margin # 间隔大小
        self.margin_type = margin_type # 间隔类型：'cos'或'arc'
        self.if_multiclass = if_multiclass # 是否多标签分类

        self.quan_weight = quan_weight # 量化损失权重
        self.quan_type = quan_type  # 量化损失类型
        self.multiclass_loss = multiclass_loss # 多标签损失类型

        assert multiclass_loss in ['bce', 'imbalance', 'label_smoothing']

        # 存储各项损失值
        self.losses = {}

    def compute_margin_logits(self, logits, labels):
        """计算带间隔的logits"""
        if self.margin_type == 'cos': # OrthoCos
            if self.if_multiclass:
                y_onehot = labels * self.margin
                margin_logits = self.temperature * (logits - y_onehot)
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.margin)
                margin_logits = self.temperature * (logits - y_onehot)
        else: # OrthoArc
            if self.if_multiclass:
                y_onehot = labels * self.margin
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.temperature * logits
            else:
                y_onehot = torch.zeros_like(logits)
                y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.margin)
                arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
                logits = torch.cos(arc_logits + y_onehot)
                margin_logits = self.temperature * logits

        return margin_logits

    def compute_quantization_loss(self, code_logits):
        """计算量化损失"""
        if self.quan_type == 'cs':
            quantization = (1. - F.cosine_similarity(
                code_logits,
                code_logits.detach().sign(),
                dim=1
            )).mean()
        elif self.quan_type == 'l1':
            quantization = torch.abs(
                code_logits - code_logits.detach().sign()
            ).mean()
        else:  # l2
            quantization = torch.pow(
                code_logits - code_logits.detach().sign(),
                2
            ).mean()

        return quantization

    def forward(self, code_logits, logits, labels, onehot=True):
        # 处理多标签分类的情况
        if self.if_multiclass:
            if not onehot:
                labels = F.one_hot(labels, logits.size(1))
            labels = labels.float()

            # 计算带间隔的logits
            margin_logits = self.compute_margin_logits(logits, labels)

            # 根据多标签损失类型计算损失
            if self.multiclass_loss in ['bce', 'imbalance']:
                loss_ce = F.binary_cross_entropy_with_logits(margin_logits, labels, reduction='none')
                if self.multiclass_loss == 'imbalance':
                    imbalance_mask, multiclass_acc = get_imbalance_mask(torch.sigmoid(margin_logits), labels,
                                                                        labels.size(1))
                    loss_ce = loss_ce * imbalance_mask
                    loss_ce = loss_ce.sum() / (imbalance_mask.sum() + 1e-7)
                    self.losses['multiclass_acc'] = multiclass_acc
                else:
                    loss_ce = loss_ce.mean()
            elif self.multiclass_loss in ['label_smoothing']: # label_smoothing
                log_logits = F.log_softmax(margin_logits, dim=1)
                labels_scaled = labels / labels.sum(dim=1, keepdim=True)
                loss_ce = - (labels_scaled * log_logits).sum(dim=1).mean()
            else:
                raise NotImplementedError(f'unknown method: {self.multiclass_loss}')

        # 处理单标签分类的情况
        else:
            if onehot:
                labels = labels.argmax(1)

            margin_logits = self.compute_margin_logits(logits, labels)
            loss_ce = F.cross_entropy(margin_logits, labels)

        # 计算量化损失
        if self.quan_weight != 0:
            loss_quan = self.compute_quantization_loss(code_logits)
        else:  # l2
            loss_quan = torch.tensor(0.).to(code_logits.device)

        # 记录loss
        self.losses['ce'] = loss_ce.item()
        self.losses['quan'] = loss_quan.item()

        # 总loss
        total_loss = self.ce_weight * loss_ce + self.quan_weight * loss_quan
        return total_loss


# supervised contrastive loss
class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=1):
        super().__init__()
        self.logit_scale = logit_scale
        # self.logit_scale = nn.Parameter(torch.tensor(logit_scale))
        # self.temperature = temperature
        
    def forward(self, image_features, text_features, targets):
        """
        Args:
            image_features: (batch_size, feature_dim)
            text_features: (batch_size, n_cls, feature_dim)
            targets: (batch_size,) - 目标类别的索引
        """
        # 确保targets是长整型
        targets = targets.long()
        
        # 当backbone_type为'cocoop_vp'或'vp'时，将缓存复制batch_size份
        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(0).expand(targets.size(0), -1, -1).cuda()
        
        # 归一化特征
        image_features = F.normalize(image_features, dim=-1) # (batch_size, feature_dim)
        text_features = F.normalize(text_features, dim=-1) # (batch_size, n_cls, feature_dim)
        
        # 计算相似度矩阵
        temperature = self.logit_scale.exp()
        logits_per_image = torch.einsum('bd,bcd->bc', image_features, text_features) # (batch_size, n_cls)
        logits_per_image *= temperature
        
        targets_non_onehot = torch.argmax(targets, dim=-1)
        
        return F.cross_entropy(logits_per_image, targets_non_onehot)

    
class FairSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, method='FSCL'):
        super(FairSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.method = method

    
    def forward(self, features, labels, sensitive_labels, group_norm=0, mask=None):
        """
        Args:
            features: 特征向量 [N, feature_dim]
            labels: 目标类别 [N,]
            sensitive_labels: 域标签 [N,]
        """
        device = features.device
        features = F.normalize(features, dim=-1)
        
        # 新增代码，将one-hot的labels转换为索引
        if labels is not None and len(labels.shape) > 1:
            labels = torch.argmax(labels, dim=-1)
        
        batch_size = features.shape[0]

        # 生成mask
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            sensitive_labels = sensitive_labels.contiguous().view(-1, 1)
            # 类别mask
            mask = torch.eq(labels, labels.T).float().to(device)
            # 域mask
            sensitive_mask = torch.eq(sensitive_labels, sensitive_labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
            sensitive_labels = sensitive_labels.contiguous().view(-1, 1)
            sensitive_mask = torch.eq(sensitive_labels, sensitive_labels.T).float().to(device)

        # 直接使用特征计算相似度
        anchor_feature = features
        contrast_feature = features

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability，减去每行最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # sensitive_mask = sensitive_mask.repeat(anchor_count, contrast_count)
        # n_sensitive_mask=(~sensitive_mask.bool()).float()
        
        # mask-out self-contrast cases
        logits_mask = torch.ones_like(mask, device=device)
        logits_mask.fill_diagonal_(0)

        # compute log_prob
        if self.method=="FSCL":
            positive_mask = mask * logits_mask
            fair_mask = logits_mask * (~positive_mask.bool()).float() * sensitive_mask
            
            exp_logits_fair = torch.exp(logits) * fair_mask
            
            exp_logits_sum=exp_logits_fair.sum(1, keepdim=True)
            exp_logits_sum = torch.clamp(exp_logits_sum, min=1e-12) # for numerical stability
            log_prob = logits - torch.log(exp_logits_sum)

        elif self.method=="SupCon":
            positive_mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            exp_logits_sum = exp_logits.sum(1, keepdim=True)
            exp_logits_sum = torch.clamp(exp_logits_sum, min=1e-12)
            log_prob = logits - torch.log(exp_logits_sum)
            
        else:
            raise ValueError(f"Unsupported methos: {self.method}")
           
        # compute mean of log-likelihood over positive
        # apply group normalization
        if group_norm == 1:
            # 应用组归一化：根据相同类别和相同敏感属性的样本数量归一化
            pos_per_sample = (positive_mask * sensitive_mask).sum(1)
            pos_per_sample = torch.clamp(pos_per_sample, min=1)
            mean_log_prob_pos = ((positive_mask * log_prob) / pos_per_sample.unsqueeze(1)).sum(1)
        else:
            # 标准归一化：根据相同类别的样本数量归一化
            pos_per_sample = positive_mask.sum(1)
            pos_per_sample = torch.clamp(pos_per_sample, min=1)
            mean_log_prob_pos = (positive_mask * log_prob).sum(1) / pos_per_sample
    
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        #apply group normalization
        if group_norm == 1:
            # C = loss.size(0)/8
            # norm = 1/(((mask*sensitive_mask).sum(1)+1)).float()
            # loss = (loss*norm)*C
            norm_weights = 1.0 / torch.clamp((positive_mask * sensitive_mask).sum(1) + 1, min=1.0)
            loss = (loss * norm_weights) * batch_size
            
        loss = loss.mean()

        return loss
