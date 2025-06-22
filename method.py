'''
    训练、测试的函数
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

import torchvision.transforms as transforms
from loguru import logger
from torch.nn.functional import cross_entropy
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import torchinfo

from data.data_helper import Dataset
from losses import OrthoHashLoss, CLIPLoss, FairSupConLoss, MutualDistillationLoss
from model.aug_domain import EFDMix, DSU, MixStyle, MixHistogram
from model.rand_conv import RandConvModule

from model.model_loader import load_model
from metrics import mean_average_precision
from utils_old import *
import random
from PIL import ImageFilter, Image
from collections import OrderedDict
from tqdm import tqdm

# # 加速训练
# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True
# torch.backends.cuda.max_memory_allocated = lambda: torch.cuda.max_memory_allocated() / 1024 / 1024

torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# from signal import signal, SIGPIPE, SIG_DFL
# #Ignore SIG_PIPE and don't throw exceptions on it... (http://docs.python.org/library/signal.html)
# signal(SIGPIPE,SIG_DFL)

def reset_sigpipe_handling():
    """Restore the default `SIGPIPE` handler on supporting platforms.

    Python's `signal` library traps the SIGPIPE signal and translates it
    into an IOError exception, forcing the caller to handle it explicitly.

    Simpler applications that would rather silently die can revert to the
    default handler. See https://stackoverflow.com/a/30091579/1026 for details.
    """
    try:
        from signal import signal, SIGPIPE, SIG_DFL
        signal(SIGPIPE, SIG_DFL)
    except ImportError:  # If SIGPIPE is not available (win32),
        pass  # we don't have to do anything to ignore it.

reset_sigpipe_handling()

def train(train_dataloader,
          query_dataloader,
          retrieval_dataloader,
          code_length,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          num_classes,
          evaluate_interval,
          dataset,
          batch_size,
          class_names,
          aug,
          knn,
          alpha,
          use_pseudo_labels=True,
        #   auger,
          **kwargs):
    """训练深度哈希检索模型。

        Args:
            train_dataloader (DataLoader): 训练数据集的数据加载器
            query_dataloader (DataLoader): 查询数据集的数据加载器，用于评估
            retrieval_dataloader (DataLoader): 检索数据集的数据加载器，用于评估
            code_length (int): 哈希码长度
            max_iter (int): 最大训练轮数
            arch (str): 模型架构名称
            lr (float): 学习率
            device (torch.device): 运行设备(CPU/GPU)
            verbose (bool): 是否输出详细信息
            topk (int): 评估时考虑的近邻数量
            num_classes (int): 数据集的类别数量
            evaluate_interval (int): 评估间隔，每隔多少轮进行一次评估
            tag (str): 实验标识符，用于记录和区分不同实验
            batch_size (int): 训练的批次大小

        Returns:
            float: 最终的平均精度均值(mAP)
        """
    
    # 初始化模型和优化器
    model = load_model(arch, code_length, num_classes, pretrained=True, class_names=class_names,knn=knn)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if aug:
        if dataset == 'visda':
            num_aug = 2
            alpha_list = [0.1, 1.0]
            beta = 0.1
        else:
            num_aug = 1
            alpha_list = [0.1]
            # alpha_list = [alpha] # testing alpha!
            beta = 1.0
        aug_loader = [EFDMix(p=1.0, alpha=alpha).to(device) for alpha in alpha_list]
    
    # 训练循环
    for epoch in range(max_iter):
        model.train()
        # 进度条
        train_bar = tqdm(train_dataloader, 
                    desc=f'Epoch [{epoch+1}/{max_iter}]',
                    ncols=80,
                    position=0,
                    leave=True)
        
        for batch_idx, (data, _, target, domain_target, _) in enumerate(train_bar):
            if data.size(0) == 1:  # 跳过 batch_size = 1 的情况，A->W和A->D出现问题
                continue
            
            real_batch_size = data.size(0)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            # forward
            if aug:
                # augmentation of virtual domain
                domain_target = domain_target.to(device)
                # 多次增强的逻辑
                data_aug = []
                domain_target_aug = []
                # 应用所有增强器并为每个增强器分配不同的域标签
                for aug_idx, augmenter in enumerate(aug_loader):
                    data_aug_tmp = augmenter(data)
                    data_aug.append(data_aug_tmp)
                    # 为每个增强器创建不同的域标签（1, 2, 3, ...）
                    domain_target_tmp = torch.ones_like(domain_target) * (aug_idx + 1)
                    domain_target_aug.append(domain_target_tmp)
                # 列表->Tensor
                data_aug = torch.cat(data_aug, dim=0)
                domain_target_aug = torch.cat(domain_target_aug, dim=0)
                
                domain_target = torch.cat([domain_target, domain_target_aug], dim=0)
                target = torch.cat([target] * (len(aug_loader) + 1), dim=0)
                
                # 拓展为cocoop_covp和cocoop_vp
                if 'cocoop_covp' in arch or 'cocoop_vp' in arch:
                    code_logits, logits, (image_features, text_features) = model(data, data_aug)
                else:
                    code_logits, logits, (image_features,text_features) = model(data)
                    code_logits_aug, logits_aug, (image_features_aug,text_features_aug) = model(data_aug)
                    # feature combination
                    code_logits = torch.cat([code_logits, code_logits_aug], dim=0)
                    logits = torch.cat([logits, logits_aug],dim=0)
                    image_features = torch.cat([image_features, image_features_aug], dim=0)
                    text_features = torch.cat([text_features, text_features_aug], dim=0)
            else:
                code_logits, logits, (image_features,text_features) = model(data)
                
            # 新增：生成伪标签
            if use_pseudo_labels:
                from utils import generate_pseudo_labels_from_features
                # 使用模型输出的特征来生成伪标签
                target = generate_pseudo_labels_from_features(
                    image_features[:real_batch_size],  # 只对原始数据生成伪标签
                    text_features[:real_batch_size], 
                    num_classes, 
                    device
                )
                # 如果有数据增强，需要复制标签
                if aug:
                    target = torch.cat([target] * (len(aug_loader) + 1), dim=0)
            
            # 计算loss
            hash_loss_fn = OrthoHashLoss(code_length, num_classes) # 哈希损失
            hash_loss = hash_loss_fn(code_logits, logits, target)
            clip_loss_fn = CLIPLoss(logit_scale=model.logit_scale) # 图像和文本的对比损失
            clip_loss = clip_loss_fn(image_features, text_features, target)
            
            if aug:
                # SupCon or FSCL
                con_loss_fn = FairSupConLoss(method="FSCL") # 图像间对比损失
                con_loss = con_loss_fn(image_features, target, domain_target)
                loss = hash_loss + clip_loss + con_loss
            else:
                loss = hash_loss + clip_loss
            
            # backward
            loss.backward()
            optimizer.step()

        logger.info(f'[Epoch:{epoch+1}/{max_iter}][Train loss:{loss.item()/batch_size:.4f}]')

        # 定期评估
        if (epoch+1) % evaluate_interval == 0:
            mAP = evaluate(
                model,
                query_dataloader,
                retrieval_dataloader,
                code_length,
                device,
                topk,
                save_results=False,
            )
            logger.info(f'[Epoch:{epoch+1}/{max_iter}][mAP:{mAP:.4f}]')

    # 最终评估
    mAP = evaluate(model, query_dataloader, retrieval_dataloader,
                  code_length, device, topk, save_results=False)
    
    # optional: 保存模型
    # save_path = os.path.join('checkpoints', f'{tag}_{arch}_code{code_length}_mAP_{mAP:.4f}.pt')
    # torch.save({
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'mAP': mAP,
    # }, save_path)
    

    logger.info(f'Training finished, [iteration:{epoch+1}][mAP:{mAP:.4f}]')
    return mAP


def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, save_results=False):
    """评估哈希检索模型的性能。

       Args:
           model: 待评估的模型
           query_dataloader: 查询数据集的数据加载器
           retrieval_dataloader: 检索数据集的数据加载器
           code_length: 哈希码长度
           device: 运行设备
           topk: 评估时考虑的近邻数量
           save_results: 是否保存结果

       Returns:
           float: 平均精度均值(mAP)
       """
    model.eval()
    
    with torch.no_grad(): # 添加无梯度上下文，提高效率
        # with torch.cuda.amp.autocast():
        # 生成哈希码
        query_codes = generate_codes(model, query_dataloader, code_length, device)
        retrieval_codes = generate_codes(model, retrieval_dataloader, code_length, device)
        
        # 获取目标标签
        query_targets = query_dataloader.dataset.get_targets().to(device)
        retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)

        # 计算mAP
        mAP = mean_average_precision(
            query_codes,
            retrieval_codes,
            query_targets,
            retrieval_targets,
            device,
            topk,
        )

        # 保存结果
        if save_results:
            results = { # 使用字典统一管理待保存数据
                'query_codes': query_codes,
                'retrieval_codes': retrieval_codes,
                'query_targets': query_targets,
                'retrieval_targets': retrieval_targets,
            }
            # 确保保存目录存在
            save_dir = 'results'
            os.makedirs(save_dir, exist_ok=True)
            
            # 批量保存数据
            for name, data in results.items():
                np.save(
                    os.path.join(save_dir, f'{name}_code{code_length}_mAP_{mAP:.4f}.npy'),
                    data.cpu().numpy()
                )

    return mAP


def generate_codes(model, dataloader, code_length, device):
    """生成哈希码。

    该函数对输入数据集中的所有样本生成二值哈希码，用于后续的相似性检索。

    Args:
        model (torch.nn.Module): 已训练的深度哈希模型。
        dataloader (torch.utils.data.DataLoader): 数据加载器。
        code_length (int): 哈希码长度。(用于验证)
        device (torch.device): 计算设备。

    Returns:
        torch.Tensor: 生成的哈希码矩阵。
        形状为 [N, code_length]，其中N是数据集的样本数量。
            矩阵中的元素为{-1, 1}，表示二值哈希码。
    """
    # 避免计算梯度
    with torch.no_grad():
        # with torch.cuda.amp.autocast():
        N = len(dataloader.dataset)
        codes = []
        
        # 遍历所有batch
        for batch in dataloader:
            data = batch[0]
            if isinstance(data, torch.Tensor):
                data = data.to(device)
                
                outputs = model(data)
                code_logits, _, _ = outputs # 连续哈希code
                # 检查维度是否正确
                assert code_logits.size(1) == code_length, f"Code length mismatch: expected {code_length}, got {code_logits.size(1)}"
                
                codes.append(torch.sign(code_logits)) # 二值哈希code
            
        return torch.cat(codes, dim=0)










    


    
