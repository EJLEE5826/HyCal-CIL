import os
import torch
import random
from typing import Dict, Optional
from PIL import Image
from importlib import import_module

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from task_datasets.utils import set_seed

# Import CoOp datasets
from task_datasets.oxford_pets import OxfordPets
from task_datasets.oxford_flowers import OxfordFlowers
from task_datasets.fgvc import FGVCAircraft
from task_datasets.dtd import DescribableTextures
from task_datasets.eurosat import EuroSAT
from task_datasets.stanford_cars import StanfordCars
from task_datasets.food101 import Food101
from task_datasets.sun397 import SUN397
from task_datasets.caltech101 import Caltech101
from task_datasets.ucf101 import UCF101
from task_datasets.organamnist import OrganAMNIST
from task_datasets.galaxy10_decals import GALAXY10Decals
from task_datasets.collections import (
    MNIST,
    ARTBENCH10_256
)

# Map dataset names to dataset classes
DATASETS = {
    "oxford_pets": OxfordPets,
    "eurosat": EuroSAT,
    "ucf101": UCF101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "dtd": DescribableTextures,
    "aircraft": FGVCAircraft,
    "food101": Food101,
    "oxford_flowers": OxfordFlowers,
    "stanford_cars": StanfordCars,
    "mnist": MNIST,
    "artbench10-256": ARTBENCH10_256,
    "organamnist": OrganAMNIST,
    "galaxy10-decals": GALAXY10Decals,
}

def get_dataset_class(dataset_name: str):
    """Get dataset class by name, handling special cases to avoid circular imports"""
    if dataset_name == "imagenet":
        # Dynamically import ImageNet to avoid circular import
        imagenet_module = import_module("task_datasets.imagenet")
        return imagenet_module.ImageNet
    return DATASETS.get(dataset_name)


def read_image(path):
    """Read image from path using ``PIL.Image``.
    Args:
        path (str): path to an image.
    Returns:
        PIL image
    """
    if not os.path.exists(path):
        raise IOError('No file exists at {}'.format(path))
    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )

class BaseDatasetWrapper(Dataset):
    """Wrapper for handling data from base datasets"""
    def __init__(self, data_source, input_size, transform=None, classnames=None, is_train=False,
                 return_img0=False, augmentation_time=3):
    # def __init__(self, data, transform=None, classnames=None):
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        self.return_img0 = return_img0
        self.classnames = classnames

        self.augmentation_time = augmentation_time if is_train else 1
        self.augmentation = T.Compose([
            T.RandomHorizontalFlip(p=0.8),
            T.RandomVerticalFlip(p=0.8),
            T.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        # Build transform that doesn't apply any data augmentation
        self.to_tensor = T.Compose([
            T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                        std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.num_data_source = len(self.data_source)

        try:
            self.samples = [(item.impath, item.classname) for item in data_source]
        except: # Handle case where data_source is a list of tuples
            if hasattr(data_source, 'indices'):
                self.samples = [(data_source.full_dataset.samples[indice][0], classnames[data_source.full_dataset.samples[indice][1]]) for indice in data_source.indices]
            else:
                self.samples = [(item[0], classnames[item[1]]) for item in data_source.samples]
            # self.samples = [(item[0], item[1]) for item in data_source]

    
    def __len__(self):
        # return len(self.samples)
        return self.num_data_source * self.augmentation_time
    
    def __getitem__(self, idx):
        # 인덱스 계산 최적화
        original_idx = idx % self.num_data_source
        augmentation_idx = idx // self.num_data_source
        
        img_path, class_name = self.samples[original_idx]
        img0 = read_image(img_path)
        
        if self.is_train and augmentation_idx > 0: # 안들어감
            img0 = self.augmentation(img0)
        
        # To tensor & normalization by input transform
        output = {}
        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    output[f'img{i+1}' if i > 0 else 'img'] = tfm(img0)
            else:
                output['img'] = self.transform(img0)

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        return output['img'], class_name

class CustomDataloader:
    """
    Factory for creating dataloaders using CoOp datasets
    """
    @staticmethod 
    def create_dataloaders(cfg: Dict, few_shot_cfg: Optional[Dict] = None, input_size=224) -> Dict[str, Dict[str, DataLoader]]:
        """
        Create dataloaders for multiple CoOp datasets based on configuration
        
        Args:
            config: Task and dataset configuration
            few_shot_cfg: Few-shot settings (samples per class, etc.)
            
        Returns:
            Dict[task_name -> Dict[split -> DataLoader]]
        """
        dataloaders = {}
        total_classes = []
        
        # 각 데이터셋별 실제 사용된 num_shots 값을 저장할 딕셔너리
        dataset_shots = {}
        
        for task_name in cfg.datasets:
            if task_name not in DATASETS:
                raise ValueError(
                    f"Unsupported dataset '{task_name}'. Supported datasets: {', '.join(DATASETS)}"
                )
            dataset_class = get_dataset_class(task_name)

            # Define known dataset class counts to avoid initialization
            DATASET_CLASS_COUNTS = {
                'oxford_pets': 37,
                'eurosat': 10,
                'ucf101': 101,
                'sun397': 397,
                'caltech101': 101,
                'dtd': 47,
                'aircraft': 100,
                'food101': 101,
                'oxford_flowers': 102,
                'stanford_cars': 196,
                'mnist': 10,
                'artbench10-256': 10,
                'organamnist': 11,
                'galaxy10-decals': 10,
                'imagenet': 1000
            }

            # Check few-shot settings
            num_shots = None
            if hasattr(cfg, 'num_shots_each_range') and cfg.num_shots_each_range is not None:
                # Get number of classes without initializing the dataset
                num_classes = DATASET_CLASS_COUNTS.get(task_name)
                
                if num_classes is None:
                    # If dataset not in known list, try to get class count from the dataset
                    try:
                        if task_name in ["mnist", "artbench10-256"]:
                            dataset = dataset_class(num_shots=None, preprocess=None, location=cfg.data_root)
                        else:
                            dataset = dataset_class(root=cfg.data_root, num_shots=None)
                        num_classes = len(dataset.classnames) if hasattr(dataset, 'classnames') else 10
                    except Exception as e:
                        print(f"Error initializing {task_name} for class count: {str(e)}")
                        num_classes = 10  # Default fallback
                
                # Generate random shots per class
                min_shots, max_shots = cfg.num_shots_each_range
                num_shots = [random.randint(min_shots, max_shots) for _ in range(num_classes)]
                print(f"{task_name}: Random shots per class - {num_shots}")
            
            # Preserve the original num_shots if num_shots_each_range is not specified
            if num_shots is None and hasattr(cfg, 'num_shots') and cfg.num_shots is not None:
                num_shots = cfg.num_shots
            elif cfg.num_shots_each_range is None and few_shot_cfg and task_name in few_shot_cfg:
                num_shots = few_shot_cfg[task_name].get('train', None)
            
            # Store the shot configuration
            dataset_shots[task_name] = num_shots
            
            if task_name in ["mnist", "artbench10-256"]:
                base_dataset = dataset_class(num_shots=num_shots, preprocess=cfg.train_transform, val_transform=cfg.val_transform,
                                batch_size=cfg.batch_size, location=cfg.data_root)
            else:
                base_dataset = dataset_class(root=cfg.data_root, num_shots=num_shots)
            
            # Initialize task's dataloader dict
            dataloaders[task_name] = {}

            if task_name in ["mnist", "artbench10-256"]:
                split_data = {
                    'train': base_dataset.train_dataset,
                    'test': base_dataset.test_dataset
                }
                total_classes.extend(base_dataset.classnames)
            else:
                # Create datasets and dataloaders for each split using the same base_dataset
                split_data = {
                    'train': base_dataset.train_x,
                    'test': base_dataset.test,
                }

                total_classes.extend(base_dataset.classnames)
                
            for split, data in split_data.items():
                if split == 'train':
                    transform = cfg.train_transform
                elif split == 'test':
                    transform = cfg.val_transform
                    
                if data:  # Check if split exists
                    # Create dataset with appropriate split data
                    if task_name == "mnist":
                        dataset = data
                    else:
                        dataset = BaseDatasetWrapper(
                            data_source=data,
                            input_size=input_size,
                            transform=transform,
                            is_train=True if split == 'train' else False,
                            augmentation_time=cfg.augmentation_time,
                            classnames=base_dataset.classnames
                        )
                    if task_name == "mnist":
                        if split == 'train':
                            dataloader = base_dataset.train_loader
                        else:
                            dataloader = base_dataset.test_loader
                        if hasattr(data, 'indices'):
                            targets = (data.targets[index] for index in data.indices)
                        else:
                            targets = data.targets
                        dataloader.dataset.samples = [
                            (None, base_dataset.classnames[target]) for target in targets
                        ]
                        dataloader.dataset.classnames = base_dataset.classnames
                    else:
                        # Create dataloader
                        dataloader = DataLoader(
                            dataset,
                            batch_size=cfg.batch_size,
                            shuffle=(split == 'train'),
                            num_workers=cfg.num_workers,
                            pin_memory=torch.cuda.is_available() and split == 'test',
                            prefetch_factor=2 if cfg.num_workers > 0 else None,
                            drop_last=False
                        )
                    dataloaders[task_name][split] = dataloader
                    print(f"Created dataloader for {task_name} ({split}): {len(dataset)} samples")
                else:
                    print(f"No data available for {task_name} ({split})")
                    dataloaders[task_name][split] = None
                        
        # 데이터로더, 전체 클래스 목록, 각 데이터셋별 샷 수를 반환
        return dataloaders, total_classes, dataset_shots


def get_available_datasets():
    """Return list of available datasets"""
    return list(DATASETS.keys())
