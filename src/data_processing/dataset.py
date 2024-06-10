import os
import cv2 as cv
import pandas as pd
from .vocabulary import Vocabulary
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

TRANSFORMS = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class FlickrDataset(Dataset):
    def __init__(self, images_path: str, df: pd.DataFrame, vocab: Vocabulary, sample, random_state=42, transforms=TRANSFORMS):
        self.images_path = images_path
        self.df = df.copy()
        self.vocab = vocab
        
        train, valid = train_test_split(list(self.df.index), test_size=0.1, random_state=random_state)
        valid, test = train_test_split(list(valid), train_size=0.5, random_state=random_state)
        if sample == 'train':
            self.df = self.df[self.df.index.isin(train)].reset_index()
        elif sample == 'valid':
            self.df = self.df[self.df.index.isin(valid)].reset_index()
        elif sample == 'test':
            self.df = self.df[self.df.index.isin(test)].reset_index()
        self.captions = self.df[' comment'].astype(str)
        self.imgs = self.df['image_name']
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        target = self.vocab.numericalize(caption)
        image_name = self.imgs[idx]
        image = cv.cvtColor(cv.imread(os.path.join(self.images_path, image_name)), cv.COLOR_BGR2RGB)
        img_tensor = self.transforms(image)
        return img_tensor, target
    
class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        imgs = [item[0][None, ...] for item in batch]
        imgs = torch.cat(imgs)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, padding_value=self.pad_idx, batch_first=True)
        return imgs, targets