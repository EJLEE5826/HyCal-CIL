import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image
# import albumentations as A
# from torchvision.transforms.functional import to_pil_image
# from task_datasets.dataset import set_seed


# def split_trainval(trainval, p_val=0.2):
#     """Split trainval data into train and validation sets.
    
#     Args:
#         trainval: list of Datum objects
#         p_val: validation set proportion (default: 0.2)
#     """
#     p_trn = 1 - p_val
#     print(f'Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val')
    
#     # Group indices by label using defaultdict
#     tracker = defaultdict(list)
#     for idx, item in enumerate(trainval):
#         tracker[item.label].append(idx)
    
#     train, val = [], []
#     for idxs in tracker.values():
#         n_val = round(len(idxs) * p_val)
#         assert n_val > 0
#         random.shuffle(idxs)
#         val.extend(trainval[idx] for idx in idxs[:n_val])
#         train.extend(trainval[idx] for idx in idxs[n_val:])

#     return train, val


def save_split(train, val, test, filepath, path_prefix):
    def _extract(items):
        out = []
        for item in items:
            impath = item.impath
            label = item.label
            classname = item.classname
            impath = impath.replace(path_prefix, '')
            if impath.startswith('/'):
                impath = impath[1:]
            out.append((impath, label, classname))
        return out

    train = _extract(train)
    val = _extract(val)
    test = _extract(test)

    split = {
        'train': train,
        'val': val,
        'test': test
    }

    write_json(split, filepath)
    print(f'Saved split to {filepath}')


def read_split(filepath, path_prefix):
    def _convert(items):
        return [
            Datum(
                impath=os.path.join(path_prefix, impath),
                label=int(label),
                classname=classname
            )
            for impath, label, classname in items
        ]

    print(f'Reading split from {filepath}')
    split = read_json(filepath)
    train = _convert(split['train'])
    val = _convert(split['val'])
    test = _convert(split['test'])

    return train, val, test


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
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


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
        path (str): directory path.
        sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    def update_label(self, base_cls_num):
        self._label += base_cls_num

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname

    def update_classname(self, new_classname):
        self._classname = new_classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = ''  # the directory where the dataset is stored
    domains = []  # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x  # labeled training data
        self._train_u = train_u  # unlabeled training data (optional)
        self._val = val  # validation data (optional)
        self._test = test  # test data

        # if not hasattr(self, '_num_classes'): # defined at 'generate_fewshot_dataset'
            # 한 번의 순회로 모든 정보를 수집
        self._process_dataset(train_x)
        #     self._num_classes = self.get_num_classes(train_x)
        # self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    def update_classnames(self, new_classnames):
        self._classnames = new_classnames

    @property
    def num_classes(self):
        return self._num_classes
                
    def _process_dataset(self, data_source):
        """Process dataset in a single pass to get all required information.
        
        Args:
            data_source (list): a list of Datum objects.
        """
        if len(data_source) == 0:
            self._num_classes = 0
            self._lab2cname = None
            self._classnames = None
            self._label_to_items = defaultdict(list)
            print('Dataset is empty. No information will be collected.')
            return
            
        # 한 번의 순회로 모든 정보 수집
        label_to_items = defaultdict(list)
        label_to_classname = {}
        
        for item in data_source:
            label_to_items[item.label].append(item)
            label_to_classname[item.label] = item.classname
            
        # 필요한 모든 정보 설정
        self._label_to_items = label_to_items
        self._num_classes = max(label_to_items.keys()) + 1
        
        # lab2cname과 classnames 설정
        labels = sorted(label_to_classname.keys())
        self._lab2cname = label_to_classname
        self._classnames = [label_to_classname[label] for label in labels]

    def merge_dataset(self, data_source):
        self._train_x += data_source.train_x
        self._val += data_source.val
        self._test += data_source.test

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        if len(data_source) == 0:
            return 0
        if hasattr(self, '_num_classes'):
            return self._num_classes
        return max(item.label for item in data_source) + 1

    def update_num_classes(self, data_source):
        self._num_classes = self.get_num_classes(data_source)
        self._lab2cname, self._classnames = self.get_lab2cname(data_source)

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        if len(data_source) == 0:
            return None, None
        if hasattr(self, '_lab2cname'):
            return self._lab2cname, self._classnames
            
        mapping = {
            item.label: item.classname 
            for item in data_source
        }
        labels = sorted(mapping.keys())
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))

    def generate_fewshot_dataset(
            self, *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int or list): number of instances per class to sample.
                                   If int, same number of shots for all classes.
                                   If list, should have length equal to number of classes.
            repeat (bool): repeat images if needed.
        Return:
            data_sources: each individual is a list containing Datum objects. (Sampled)
        """
        if isinstance(num_shots, int) and num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            
            # Convert num_shots to a list if it's an int
            if isinstance(num_shots, int):
                shots_per_class = [num_shots] * len(tracker)
                print(f'Creating a {num_shots}-shot dataset')
            else:
                shots_per_class = num_shots
                print(f'Creating a dataset with per-class shots: {shots_per_class}')
            
            for idx, (label, items) in enumerate(tracker.items()):
                if idx >= len(shots_per_class):
                    # If we run out of shot counts, use the last one
                    n = shots_per_class[-1]
                else:
                    n = shots_per_class[idx]
                
                assert n > 0, f"Shot count must be positive, but got {n}"
                
                if len(items) >= n:
                    sampled_items = random.sample(items, n)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=n)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            self._num_classes = len(tracker)
            return output[0]
        self._num_classes = len(tracker)
        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        if hasattr(self, '_label_to_items'):
            return self._label_to_items
            
        result = defaultdict(list)
        for item in data_source:
            result[item.label].append(item)
        return result

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        from itertools import groupby
        return defaultdict(list, {
            domain: list(items)
            for domain, items in groupby(
                sorted(data_source, key=lambda x: x.domain),
                key=lambda x: x.domain
            )
        })


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None, is_train=False,
                 return_img0=False, k_tfm=1, augmentation_time=3):
        self.data_source = data_source
        self.transform = transform  # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0
        self.augmentation_time = augmentation_time if is_train else 1  # dataset augmentation
        self.augmentation = T.Compose([
            T.RandomHorizontalFlip(p=0.8),
            T.RandomVerticalFlip(p=0.8),
            T.ColorJitter(brightness=0.5, contrast=0.5)
        ])

        # self.cache = {}  # ONLY FOR SPEEDUP RAIL

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source) * self.augmentation_time

    def __getitem__(self, idx):
        # if self.is_train!=True and idx in self.cache:
        #     return self.cache[idx]
        
        # 인덱스 계산 최적화
        original_idx = idx % len(self.data_source)
        augmentation_idx = idx // len(self.data_source)
        
        item = self.data_source[original_idx]
        img0 = read_image(item.impath)
        
        # Augmentation 적용
        if augmentation_idx > 0:
            img0 = self.augmentation(img0)
            # if original_idx == 1 and self.is_train:
            #     # img_pil = to_pil_image(img0)
            #     plt.imshow(img0)
            #     plt.axis('off')
            #     plt.show()

        # To tensor & normalization by input transform
        output = {}
        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    output[f'img{i+1}' if i > 0 else 'img'] = self._transform_image(tfm, img0)
            else:
                output['img'] = self._transform_image(self.transform, img0)
        # else:
        #     output = {'img': self.to_tensor(img0)}
            
        if self.return_img0:
            output['img0'] = self.to_tensor(img0)
        
        # self.cache[idx] = (output['img'], item.label)
        return output['img'], item.label

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


def build_data_loader(
        data_source=None,
        batch_size=64,
        input_size=224,
        tfm=None,
        is_train=True,
        shuffle=False,
        dataset_wrapper=None,
        augmentation_time=1,
        num_workers=8
):
    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper
        
    # 데이터셋 인스턴스 생성
    dataset = dataset_wrapper(
        data_source,
        input_size=input_size,
        transform=tfm,
        is_train=is_train,
        augmentation_time=augmentation_time
    )
    
    # DataLoader 생성 시 성능 최적화 옵션 적용
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None

    )
    
    assert len(data_loader) > 0, "Data loader must not be empty"
    return data_loader


def set_seed(seed: int):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)