import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger("DataLoader")

def get_data_stats(dataloaders: Dict[str, Dict[str, DataLoader]]) -> Dict:
    """
    Extract dataset statistics from dataloaders
    
    Args:
        dataloaders: Task-wise and split-wise dataloaders
    
    Returns:
        Dataset statistics (class counts, sample counts, etc. per task)
    """
    stats = {}
    
    for task_name, splits in dataloaders.items():
        stats[task_name] = {}
        
        for split_name, loader in splits.items():
            dataset = loader.dataset
            class_counts = {}
            
            # Calculate samples per class
            for _, class_name in dataset.samples:
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                class_counts[class_name] += 1
            
            stats[task_name][split_name] = {
                'total_samples': len(dataset),
                'num_classes': len(dataset.classnames),
                'classes': dataset.classnames,
                'class_counts': class_counts
            }
    return stats

def save_data_stats(stats: Dict, path: str):
    """
    Save dataset statistics in JSON format
    
    Args:
        stats: Dataset statistics
        path: Save path
    """
    import json
    
    # Create directory
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Handle non-serializable objects
    serializable_stats = {}
    for task_name, task_stats in stats.items():
        serializable_stats[task_name] = {}
        for split_name, split_stats in task_stats.items():
            serializable_stats[task_name][split_name] = {
                'total_samples': split_stats['total_samples'],
                'num_classes': split_stats['num_classes'],
                'classes': split_stats['classes'],
                'class_counts': {k: int(v) for k, v in split_stats['class_counts'].items()}
            }

    # 만약 path 파일이 이미 존재하면, 파일명 뒤에 숫자를 붙여서 저장
    if os.path.exists(path):
        base, ext = os.path.splitext(path)
        i = 1
        new_path = f"{base}_{i}{ext}"
        while os.path.exists(new_path):
            i += 1
            new_path = f"{base}_{i}{ext}"
        path = new_path
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dataset statistics saved: {path}")
