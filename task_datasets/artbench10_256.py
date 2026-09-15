import torch
from torchvision.datasets import ImageFolder
import os

class ArtBench10_256(ImageFolder):
    """ArtBench10 데이터셋 클래스.
    
    ImageFolder 구조로 준비된 ArtBench10 데이터셋을 로딩합니다.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None):
        # 제공된 경로 사용 
        root_dir = root
        
        super().__init__(
            root=root_dir,
            transform=transform,
            target_transform=target_transform
        )
        
        # ImageFolder에서 자동으로 생성된 classes와 class_to_idx 활용
        # 폴더명 기반 클래스명 목록
        self.classes = self.classes  # ImageFolder에서 자동 생성된 classes
        self.class_to_idx = self.class_to_idx  # ImageFolder에서 자동 생성된 class_to_idx
        self.classnames = self.classes  # classes와 동일한 정보를 classnames에도 저장


class ARTBENCH10_256:
    
    def __init__(
            self,
            preprocess,
            val_transform,
            location="./data",
            batch_size=128,
            num_workers=8,
            classnames=None,
    ):  
        
        self.train_dataset_dir = os.path.join(location, 'train')
        self.test_dataset_dir = os.path.join(location, 'test')
        # train_dataset
        self.train_dataset = ArtBench10_256(
            root=self.train_dataset_dir, transform=preprocess
        )
        # test_dataset
        self.test_dataset = ArtBench10_256(
            root=self.test_dataset_dir, transform=val_transform
        )

        # ['impressionism', 'realism', 'romanticism', 'expressionism', 'baroque', 'post_impressionism', 'art_nouveau', 'surrealism', 'ukiyo_e', 'renaissance']
        self.classes = self.test_dataset.classes.copy()
        self.class_to_idx = self.test_dataset.class_to_idx.copy()
        self.classnames = self.classes  # classes와 동일한 정보를 classnames에도 저장
        self.template = ['a photo of a {} style painting.']  # 예술 스타일에 맞게 템플릿 수정
        
        # # 데이터로더 설정 (필요시)
        # self.train_loader = torch.utils.data.DataLoader(
        #     self.train_dataset,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     num_workers=num_workers
        # )
        
        # self.test_loader = torch.utils.data.DataLoader(
        #     self.test_dataset,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=num_workers
        # )