import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np

# 절대 경로로 임포트 변경
try:
    from .artbench10_256 import ARTBENCH10_256 as artbench10_256
except ImportError:
    # 직접 실행 시 절대 경로 임포트
    from task_datasets.artbench10_256 import ARTBENCH10_256 as artbench10_256

def underline_to_space(s):
    return s.replace("_", " ")


class FewShotDataset(Dataset):
    def __init__(self, full_dataset, n_shot, transform, class_to_idx=None):
        self.full_dataset = full_dataset
        self.transform = transform
        self.class_to_idx = class_to_idx

        # Get targets/labels from the dataset
        try:
            if hasattr(full_dataset, 'targets'):
                self.targets = full_dataset.targets
            elif hasattr(full_dataset, 'labels'):
                self.targets = full_dataset.labels
            elif hasattr(full_dataset, 'dataset') and hasattr(full_dataset.dataset, 'targets'):
                # Handle Subset or other wrapper datasets
                self.targets = full_dataset.dataset.targets
                if hasattr(full_dataset, 'indices'):
                    self.targets = [self.targets[i] for i in full_dataset.indices]
            else:
                # Try to extract targets from dataset items
                self.targets = [full_dataset[i][1] for i in range(len(full_dataset))]
        except Exception as e:
            # Fallback for other dataset types
            try:
                self.targets = [label for _, label in full_dataset]
            except:
                raise ValueError("Could not extract targets from the dataset")
        
        self.targets = np.array(self.targets)
        self.classes = np.unique(self.targets)
        
        # Handle n_shot parameter (can be int or list)
        if isinstance(n_shot, int):
            self.shots_per_class = [n_shot] * len(self.classes)
        elif isinstance(n_shot, (list, tuple, np.ndarray)):
            if len(n_shot) != len(self.classes):
                # If number of shots doesn't match number of classes, use the first n shots
                # or pad with the last shot count
                if len(n_shot) < len(self.classes):
                    self.shots_per_class = list(n_shot) + [n_shot[-1]] * (len(self.classes) - len(n_shot))
                else:
                    self.shots_per_class = n_shot[:len(self.classes)]
            else:
                self.shots_per_class = list(n_shot)
        else:
            raise ValueError("n_shot must be an integer or a list of integers")
            
        # Extract indices for few-shot learning
        self.indices = self._extract_n_shot_indices()

    def _extract_n_shot_indices(self):
        indices = []
        
        for i, cls in enumerate(self.classes):
            n_shot = self.shots_per_class[i]
            if n_shot <= 0:
                continue
                
            # Find all indices for the current class
            cls_indices = np.where(self.targets == cls)[0]
            available_samples = len(cls_indices)
            
            if available_samples == 0:
                continue
                
            # Sample n_shot examples from this class
            if available_samples < n_shot:
                # If not enough examples, sample with replacement
                sampled = np.random.choice(cls_indices, size=available_samples, replace=False)
                remaining = np.random.choice(cls_indices, size=n_shot-available_samples, replace=True)
                selected = np.concatenate([sampled, remaining])
            else:
                # Otherwise, sample without replacement
                selected = np.random.choice(cls_indices, size=n_shot, replace=False)
                
            indices.extend(selected.tolist())
            
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Redirect index to selected subset
        real_idx = self.indices[idx]
        return self.full_dataset[real_idx]

# MNIST #
class ClassificationDataset:
    def __init__(
            self,
            preprocess,
            val_transform,
            location='./data',
            batch_size=128,
            batch_size_eval=None,
            num_workers=8,
            append_dataset_name_to_template=True,
            num_shots=16,
    ) -> None:
        self.name = "classification_dataset"
        self.preprocess = preprocess
        self.val_transform = val_transform
        self.location = location
        self.batch_size = batch_size
        if batch_size_eval is None:
            self.batch_size_eval = batch_size
        else:
            self.batch_size_eval = batch_size_eval
        self.num_workers = num_workers
        self.append_dataset_name_to_template = append_dataset_name_to_template

        self.train_dataset = self.fs_train_dataset = self.test_dataset = None
        self.train_loader = self.test_loader = None
        self.classnames = None
        self.templates = None
        self.num_shots = num_shots

    def build_fewshotdataset(self):
        if self.num_shots != -1:
            # Get class_to_idx if available
            class_to_idx = None
            if hasattr(self.train_dataset, 'class_to_idx'):
                class_to_idx = self.train_dataset.class_to_idx
            elif hasattr(self.train_dataset, 'dataset') and hasattr(self.train_dataset.dataset, 'class_to_idx'):
                class_to_idx = self.train_dataset.dataset.class_to_idx
                
            self.fs_train_dataset = FewShotDataset(
                full_dataset=self.train_dataset, 
                transform=self.preprocess,
                n_shot=self.num_shots,
                class_to_idx=class_to_idx
            )
            self.train_dataset = self.fs_train_dataset

    def build_dataloader(self):
        if self.num_shots == -1:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                pin_memory=False,
                drop_last=False
            )
        else:
            # Get class_to_idx if available
            class_to_idx = None
            if hasattr(self.train_dataset, 'class_to_idx'):
                class_to_idx = self.train_dataset.class_to_idx
            elif hasattr(self.train_dataset, 'dataset') and hasattr(self.train_dataset.dataset, 'class_to_idx'):
                class_to_idx = self.train_dataset.dataset.class_to_idx
                
            self.fs_train_dataset = FewShotDataset(
                full_dataset=self.train_dataset, 
                transform=self.preprocess,
                n_shot=self.num_shots,
                class_to_idx=class_to_idx
            )
            self.train_dataset = self.fs_train_dataset
            self.train_loader = torch.utils.data.DataLoader(
                self.fs_train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                pin_memory=False,
                drop_last=False
            )
            
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_eval,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
            drop_last=False
        )

    def stats(self):
        if self.num_shots == -1:
            L_train = len(self.train_dataset)
        else:
            L_train = len(self.fs_train_dataset)
        L_test = len(self.test_dataset)
        N_class = len(self.classnames)
        return L_train, L_test, N_class

    @property
    def template(self):
        if self.append_dataset_name_to_template:
            return lambda x: self.templates[0](x)[:-1] + f", from dataset {self.name}]."
        return self.templates[0]

    def process_labels(self):
        self.classnames = [underline_to_space(x) for x in self.classnames]

    def split_dataset(self, dataset, ratio=0.8):
        train_size = int(ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        return train_dataset, test_dataset

    @property
    def class_to_idx(self):
        return {v: k for k, v in enumerate(self.classnames)}


class MNIST(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "mnist"
        self.train_dataset = datasets.MNIST(
            self.location, train=True, download=True, transform=self.preprocess
        )
        self.test_dataset = datasets.MNIST(
            self.location, train=False, download=True, transform=self.val_transform
        )
        self.build_dataloader()
        # self.build_fewshotdataset()
        # self.fs_train_dataset.class_to_idx = self.train_dataset.class_to_idx
        self.classnames = self.test_dataset.classes
        self.classnames = ['number: "{}"'.format(classname) for classname in self.classnames] # To keep consistent with prompt
        self.process_labels()
        self.templates = [
            'a photo of the number: "{}".',
        ]
        self.train_dataset.classnames = self.classnames


class ARTBENCH10_256(ClassificationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "artbench10-256"
        self.location = os.path.join(self.location, 'ArtBench10/artbench-10-imagefolder-split') # 루트 경로만 사용 (ARTBENCH10_256 내부에서 경로 설정)
        dataset = artbench10_256(preprocess=self.preprocess, 
                                 val_transform=self.val_transform, 
                                 location=self.location)

        self.train_dataset = dataset.train_dataset
        self.test_dataset = dataset.test_dataset
        # self.build_dataloader()
        self.build_fewshotdataset()
        self.classnames = dataset.classes
        
        # 필요시 클래스명 표준화
        for i, name in enumerate(self.classnames):
            if name == 'ukiyo_e':
                self.classnames[i] = 'ukiyo-e'
            elif name == 'post_impressionism':
                self.classnames[i] = 'post-impressionism'
        
        self.classnames = [f"painting in the {classname} style" for classname in self.classnames] # To keep consistent with prompt
        
        self.process_labels()
        self.templates = dataset.template
