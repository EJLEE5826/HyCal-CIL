import os
import pandas as pd
from .utils import DatasetBase, read_split, Datum


print('preparing galaxy10_decals dataset')

template = ['a photo of a {}.']

class GALAXY10Decals(DatasetBase):
    dataset_dir = 'Galaxy10_DECals/_galaxy_temp'
    
    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.split_path = os.path.join(self.dataset_dir, 'galaxy10_split.csv')
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
    df = pd.read_csv(split_filepath)
    
    # if len(df.columns) == 3:
    #     df.columns = ['split', 'filename', 'class_num']
        
    class_names = { # class name 맞는지 확인 필요함
        0: 'Disturbed Galaxies',
        1: 'Merging Galaxies',
        2: 'Round Smooth Galaxies',
        3: 'In-between Round Smooth Galaxies',
        4: 'Cigar Shaped Smooth Galaxies',
        5: 'Barred Spiral Galaxies',
        6: 'Unbarred Tight Spiral Galaxies',
        7: 'Unbarred Loose Spiral Galaxies',
        8: 'Edge-on Galaxies without Bulge',
        9: 'Edge-on Galaxies with Bulge'
    }
    
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
            
        result[split_key].append([filename, class_num, class_name])
        
    train =_convert(result['train'])
    val = _convert(result['val'])
    test =_convert(result['test'])
        
    return train, val, test
        