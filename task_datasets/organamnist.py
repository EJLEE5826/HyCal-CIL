import os
import pandas as pd
from .utils import DatasetBase, read_split, Datum


print('preparing organAMNIST dataset')

template = ['a photo of a {} organ.']

class OrganAMNIST(DatasetBase):
    dataset_dir = 'MedMNIST/OrganAMNIST'
    
    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'organamnist_224')
        self.split_path = os.path.join(self.dataset_dir, 'organamnist_224.csv')
        # self.classnames = train.classnames

        self.template = template

        train, val, test = split_train_val_test(self.split_path, self.image_dir)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        super().__init__(train_x=train, val=val, test=test)
    
def split_train_val_test(split_filepath, path_prefix):
    def _convert(items):
        return[
            Datum(
                impath = os.path.join(path_prefix, impath),
                label=int(label),
                classname=classname
            )
            for impath, label, classname in items
        ]
    df = pd.read_csv(split_filepath, header=None)
    
    if len(df.columns) == 3:
        df.columns = ['split', 'filename', 'class_num']
        
    class_names = { # class name 맞는지 확인 필요함
        0: 'bladder',
        1: 'left femur',
        2: 'right femur',
        3: 'heart',
        4: 'left kidney',
        5: 'right kidney',
        6: 'liver',
        7: 'left lung',
        8: 'right lung',
        9: 'pancreas',
        10: 'spleen'
    }

    for i, classname in class_names.items():
        class_names[i] = f"a photo of a CT scan of {classname}" 
    
    result = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for _, row in df.iterrows():
        split_key = row[0].lower()
        filename = row[1]
        class_num = int(row[2])
        class_name = class_names.get(class_num, f"class_{class_num}")
        
        # if split_type == 'train':
        #     split_key = 'train'
        if split_key == 'validation':
            split_key = 'val'
        # else: 
        #     split_key = 'test'
            
        result[split_key].append([filename, class_num, class_name])
        
    train =_convert(result['train'])
    val = _convert(result['val'])
    test =_convert(result['test'])
        
    return train, val, test
        