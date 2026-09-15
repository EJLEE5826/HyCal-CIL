import os
import pandas as pd

from .utils import Datum, DatasetBase, read_json

template = ["a histology patch of {}."]

print('preparing PathMNIST dataset')

NEW_CNAMES = {
    'ADI': 'Adipose tissue',
    'BACK': 'Background',
    'DEB': 'Debris',
    'LYM': 'Lymphocytes cells', 
    'MUC': 'Mucus',
    'MUS': 'Smooth muscle tissue',
    'NORM': 'Normal colon mucosa',
    'STR': 'Cancer-associated stromal tissue',
    'TUM': 'Colorectal adenocarcinoma epithelium'
}

class PathMNIST(DatasetBase):
    """PathMNIST dataset"""
    
    dataset_dir = os.path.join('MedMNIST', 'PathMNIST')

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.template = template
        self.split_file = os.path.join(self.dataset_dir, 'pathmnist_split_info.csv')
        
        # Read train/val/test splits
        train, val, test = self.read_data()
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)
        
        super().__init__(train_x=train, val=val, test=test)
        
    def read_data(self):
        """Read the dataset splits from CSV file"""
        classnames = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 
                     'NORM', 'STR', 'TUM'] # The 9 classes in PathMNIST
        
        cname2lab = {c: i for i, c in enumerate(classnames)}
        
        df = pd.read_csv(self.split_file)
        grouped_splits = df.groupby('split')
        
        def create_datum_object(row):
            image_class = row['image_id'].split('-')[0]
            if image_class not in cname2lab:
                return None
                
            return Datum(
                impath=os.path.join(self.dataset_dir, row['split'], f"{row['image_id']}.tif"),
                label=cname2lab[image_class],
                classname=NEW_CNAMES[image_class]
            )
        
        # 각 split에 대해 데이터 생성
        splits_data = {}
        for split, group in grouped_splits:
            split_data = []
            for _, row in group.iterrows():
                datum = create_datum_object(row)
                if datum:
                    split_data.append(datum)
            splits_data[split] = split_data
        
        return splits_data['train'], splits_data['val'], splits_data['test']