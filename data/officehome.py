from re import A
from PIL import Image, ImageFile
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import torchvision.transforms as transforms
import random
from PIL import ImageFilter
from loguru import logger


class GaussianBlur:
    """Gaussian blur augmentation from SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

  
class BaseDataset(Dataset):
    """Base dataset class supporting domain adaptation tasks"""
    def __init__(self, mode, feature_transform=None, target_transform=None, num_classes=65):
        self.feature_transform = feature_transform
        self.target_transform = target_transform
        self.aug_transform = get_feature_transform('train_aug')
        self.mode = mode
        self.num_classes = num_classes
        
        # Data and targets will be set by child classes
        self.data = None
        self.targets = None

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert('RGB')
        img_aug = self.aug_transform(img)
        
        if self.feature_transform:
            img = self.feature_transform(img)
            
        target = self.target_transform(self.targets[index], self.num_classes)
        return img, img_aug, target, index

    def __len__(self):
        return len(self.data)

    def get_targets(self):
        targets = torch.zeros((len(self.targets), self.num_classes))
        for i, target in enumerate(self.targets):
            targets[i] = self.target_transform(target, self.num_classes)
        return targets


def get_feature_transform(transform_type):
    """Get data transformation pipeline"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if transform_type == 'train':
        return transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            # test，去掉增强，TODO
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    elif transform_type == 'train_aug':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    elif transform_type == 'test':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


class Onehot(object):
    def __call__(self, sample, num_class=65):
        target_onehot = torch.zeros(num_class)
        target_onehot[sample] = 1

        return target_onehot


class DomainDataset(BaseDataset):
    """Dataset class for domain adaptation tasks"""
    def __init__(self, mode, feature_transform=None, target_transform=None, num_classes=65):
        super().__init__(mode, feature_transform, target_transform, num_classes)
        self.domain_targets = None  # 添加domain_targets属性
        
    def __getitem__(self, index):
        # 获取父类返回的基本数据
        img, img_aug, target, idx = super().__getitem__(index)
        
        # 添加domain_target
        domain_target = self.domain_targets[index] if self.domain_targets is not None else None
        
        return img, img_aug, target, domain_target, idx
        
    @staticmethod
    def load_domain_data(source_path, target_path, task='cross'):
        """Load and split domain data"""
        def read_list(path):
            data, labels = [], []
            domain_labels = []  # 新增domain_labels列表
            with open(path, 'r') as f:
                for line in f:
                    parts = line.split()
                    data.append('data/hash_data/' + parts[0])
                    labels.append(int(parts[1]))
                    # 如果有domain标签，则添加
                    if len(parts) > 2:
                        domain_labels.append(int(parts[2]))
                    else:
                        domain_labels.append(0)
            # if domain_labels:
            return np.array(data), np.array(labels), np.array(domain_labels)

        source_data, source_labels, source_domains = read_list(source_path)
        target_data, target_labels, target_domains = read_list(target_path) # target_domains为None

        # Split data based on task type
        query_size = int(0.1 * len(target_data))
        perm_idx = np.random.permutation(len(target_data))
        query_idx = perm_idx[:query_size]

        query_data = target_data[query_idx]
        query_labels = target_labels[query_idx]
        query_domains = target_domains[query_idx] if target_domains is not None else None
        
        if task == 'cross':
            retrieval_data = source_data
            retrieval_labels = source_labels
            retrieval_domains = source_domains
        else:
            database_idx = perm_idx[query_size:]
            retrieval_data = target_data[database_idx]
            retrieval_labels = target_labels[database_idx]
            retrieval_domains = target_domains[database_idx] if target_domains is not None else None

        return {
            'query': (query_data, query_labels, query_domains),
            'train': (source_data, source_labels, source_domains),
            'retrieval': (retrieval_data, retrieval_labels, retrieval_domains)
        }
    

def get_classnames(dataset_name):
    if dataset_name == 'office_home':
        classnames = [
            "alarm clock", "backpack", "batteries", "bed", "bike", "bottle", "bucket", "calculator",
            "calendar", "candles", "chair", "clipboards", "computer", "couch", "curtains", "desk lamp",
            "drill", "eraser", "exit sign", "fan", "file cabinet", "flipflops", "flowers", "folder",
            "fork", "glasses", "hammer", "helmet", "kettle", "keyboard", "knives", "lamp shade",
            "laptop", "marker", "monitor", "mop", "mouse", "mug", "notebook", "oven", "pan", "paper clip",
            "pen", "pencil", "postit notes", "printer", "push pin", "radio", "refrigerator", "ruler",
            "scissors", "screwdriver", "shelf", "sink", "sneakers", "soda", "speaker", "spoon",
            "table", "telephone", "toothbrush", "toys", "trash can", "tv", "webcam"
        ]
    elif dataset_name == 'office_31':
        classnames = [
            "back_pack", "bike", "bike helmet", "bookcase", "bottle", "calculator", "desktop computer",
            "desk chair", "desk lamp", "desktop computer", "file cabinet", "headphones", "keyboard",
            "laptop computer", "letter tray", "mobile phone", "monitor", "mouse", "mug", "paper notebook",
            "pen", "phone", "printer", "projector", "punchers", "ring binder", "ruler", "scissors",
            "speaker", "stapler", "tape dispenser", "trash can"
        ]
    elif dataset_name == 'visda':
        classnames = [
            "aeroplane", "bicycle", "bus", "car", "horse", "knife", "motorcycle", "person", "plant", "skateboard", "train", "truck"
        ]
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented yet")
    
    return classnames


def get_dataloader(dataset_name, source_path, target_path, batch_size, num_workers, task='cross', num_classes=65):
    """Get data loaders for training and evaluation"""
    data = DomainDataset.load_domain_data(source_path, target_path, task)
    class_names = get_classnames(dataset_name)
    
    transforms = {
        'train': get_feature_transform('train'),
        'test': get_feature_transform('test')
    }
    
    datasets = {
        'query': DomainDataset('query', transforms['test'], target_transform=Onehot(), num_classes=num_classes),
        'train': DomainDataset('train', transforms['train'], target_transform=Onehot(), num_classes=num_classes),
        'retrieval': DomainDataset('retrieval', transforms['test'], target_transform=Onehot(), num_classes=num_classes)
    }
    
    # Set data, targets and domain_targets for each dataset
    for mode in datasets:
        datasets[mode].data, datasets[mode].targets, datasets[mode].domain_targets = data[mode]
    
    loaders = {
        'query': DataLoader(datasets['query'], batch_size=batch_size, num_workers=num_workers, collate_fn=None),
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=None),
        'retrieval': DataLoader(datasets['retrieval'], batch_size=batch_size, num_workers=num_workers, collate_fn=None)
    }
    
    logger.info(f"Task: {task}-retrieval")
    logger.info(f"Query size: {len(datasets['query'])}")
    logger.info(f"Train size: {len(datasets['train'])}")
    logger.info(f"Retrieval size: {len(datasets['retrieval'])}")
    
    return loaders['query'], loaders['train'], loaders['retrieval'], class_names

if __name__ == '__main__':

    # Test the dataloader
    source_path = 'hash_data/office_home/art.txt'
    target_path = 'hash_data/office_home/real_world.txt'
    batch_size = 32
    num_workers = 4
    task = 'cross'
    num_classes = 65

    query_loader, train_loader, retrieval_loader, _ = get_dataloader(
        'office_home',
        source_path,
        target_path,
        batch_size,
        num_workers,
        task,
        num_classes
    )

    for data, _, target, _, _ in query_loader:
        print(data.shape, target.shape)
        break