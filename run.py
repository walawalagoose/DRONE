import torch
import argparse
import os
import numpy as np
import random
from loguru import logger
import method
from model.model_loader import load_model
from data.officehome import get_dataloader

def seed_torch(seed=2022):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='DG_PyTorch')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='office_home',help="Dataset name")
    
    # Task parameters
    parser.add_argument('--source', type=str, default='product', required=False, help="Source domain")
    parser.add_argument('--target', type=str, default='real_world', required=False, help="Target domain")
    parser.add_argument('--setting', type=str, default='cross', help="cross or single")
    parser.add_argument('-c', '--code-length', default=64, type=int,
                        help='Binary hash code length.(default: 64)')
    
    # Model parameters
    parser.add_argument('-a', '--arch', default='clip', type=str,
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model')
    parser.add_argument('-n', '--kn', default=20, type=int,
                        help='Knn.(default: 20)')
    parser.add_argument('--temperature', default=0.5, type=float,
                     help='Temperature parameter.(default: 0.5)')
    parser.add_argument('--aug', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='The hyper-parameter for EFDMix.(default: 0.1)')
    
    # # temp
    # parser.add_argument('--auger', type=str, default='mixstyle', help="temp") # temp
    parser.add_argument('--knn', type=int, default=8, help="serving as rank") # temp
   
   # Training parameters
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                      help='Batch size.(default: 16)')
    parser.add_argument('-l', '--lr', default=1e-3, type=float,
                      help='Learning rate.(default: 1e-3)')
    parser.add_argument('-T', '--max-iter', type=int,
                      help='Number of iterations.(default: 50)')
    parser.add_argument('-e', '--evaluate-interval', default=5, type=int,
                      help='Interval of evaluation.(default: 5)')
    
    # Evaluation parameters
    parser.add_argument('-k', '--topk', default=-1, type=int,
                      help='Calculate mAP of top k.(default: -1)')

    # System parameters
    parser.add_argument('-w', '--num-workers', default=4, type=int,
                      help='Number of loading data threads.(default: 4)')
    parser.add_argument('-g', '--gpu', default=0, type=int,
                      help='GPU id to use.(default: 0)')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Print log.')

    # Run mode
    parser.add_argument('--mode', type=str, default='train',
                      choices=['train', 'evaluate'],
                      help='Running mode.')

    args = parser.parse_args()

    # Set device
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
    
    # 没有输入max_iter时，根据不同数据集设置不同的默认值
    if args.max_iter is None:
        if args.dataset == 'office_home':
            args.max_iter = 20
        elif args.dataset == 'office_31':
            args.max_iter = 20
        elif args.dataset == 'visda':
            args.max_iter = 5
        else:
            raise ValueError(f"Dataset {args.dataset} not implemented yet")
        
    # different datasets, diffrerent data information
    if args.dataset == 'office_home':
        args.data_path = 'data/hash_data/office_home/'
        args.num_classes = 65
    elif args.dataset == 'office_31':
        args.data_path = 'data/hash_data/office_31/'
        args.num_classes = 31
    elif args.dataset == 'visda':
        args.data_path = 'data/hash_data/visda/'
        args.num_classes = 12
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented yet")

    return args


def run():
    """Main running function."""
    # Load configuration
    seed_torch()
    args = load_config()
    
    
    # Logging setup
    log_path = os.path.join('logs', str(args.setting), str(args.dataset), str(args.arch), str(args.code_length))
    os.makedirs(log_path, exist_ok=True)
    logger.add(os.path.join(log_path, f'{args.source}->{args.target}_{args.batch_size}_{{time}}.log'), rotation="500 MB", level="INFO")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Architecture: {args.arch}")
    logger.info(f"Code length: {args.code_length}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Batch size: {args.batch_size}")

    # Set domain data path
    args.source = os.path.join(args.data_path, f"{args.source}.txt")
    args.target = os.path.join(args.data_path, f"{args.target}.txt")
    
    
    # Load dataset
    query_loader, train_loader, retrieval_loader, class_names = get_dataloader(
        args.dataset,
        args.source,
        args.target,
        args.batch_size,
        args.num_workers,
        args.setting,
        args.num_classes
    )

    if args.mode == 'train':
        method.train(
            train_loader,  # Changed from train_s_dataloader
            query_loader,
            retrieval_loader,
            args.code_length,
            args.max_iter,
            args.arch,
            args.lr,
            args.device,
            args.verbose,
            args.topk,
            args.num_classes,  # Changed from num_class
            args.evaluate_interval,
            args.dataset,      # Changed from tag
            args.batch_size,
            class_names,
            args.aug,
            args.knn, 
            args.alpha,
            use_pseudo_labels=False,
        )
        
    else:  # evaluate
        model = load_model(args.arch, args.code_length, args.num_classes, class_names=class_names)
        checkpoint = torch.load(
            os.path.join(args.save_path, f'model_{args.code_length}.pt'),
            map_location=args.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        mAP = method.evaluate(
            model,
            query_loader,
            retrieval_loader,
            args.code_length,
            args.device,
            args.topk,
        )
        logger.info(f"mAP: {mAP:.4f}")




if __name__ == '__main__':
    # import sys
    # sys.argv = ['script_name.py', 'arg1', 'arg2', 'arg3']
    run()
