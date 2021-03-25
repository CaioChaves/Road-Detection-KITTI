import os
from imageio import imread
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
from PIL import Image

class KittiDataset(Dataset):
    def __init__(self, rootDir,split = 'train',target_type = 'semantic_binary_ground', transform=None, balanced=None, colormodel='RGB'):
        self.rootDir = rootDir 
        self.split = split
        if balanced:
            self.trainingImagesPath = os.path.join(self.rootDir,'data_semantics/training/image_2_balanced')  
            self.trainingSegmentationBinaryPath = os.path.join(self.rootDir,'data_semantics/training/semantic_binary_ground_balanced')
        else:
            self.trainingImagesPath = os.path.join(self.rootDir,'data_semantics/training/image_2')
            self.trainingSegmentationBinaryPath = os.path.join(self.rootDir,'data_semantics/training/semantics_binary_ground')
        self.trainingSegmentationPath = os.path.join(self.rootDir,'data_semantics/training/semantic_rgb')
        self.testingImagesPath = os.path.join(self.rootDir,'data_semantics/testing/image_2')
        self.target_type = target_type
        self.transform = transform 
        if colormodel == 'RGB':
            self.colormodel='RGB'
        elif colormodel == 'grayscale':
            self.colormodel='L'
        self.images = []
        self.semantics = []
        	
        if not self.target_type in ['semantic', 'semantic_binary_ground']:
            raise ValueError('Invalid value for "target_type"! Valid values are: "  semantic", "semantic_binary_ground"')

        if not os.path.isdir(self.trainingSegmentationPath) or not os.path.isdir(self.trainingImagesPath):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders are inside the "root" directory')
        
        
        for filename in os.listdir(self.trainingImagesPath):
            self.images.append(os.path.join(self.trainingImagesPath,filename))
            if target_type == 'semantic':
                self.semantics.append(os.path.join(self.trainingSegmentationPath,filename))
            elif target_type == 'semantic_binary_ground':
                self.semantics.append(os.path.join(self.trainingSegmentationBinaryPath,filename))
                
        
    def __getitem__(self,idx):
        if self.split == 'train':
            image = Image.open(self.images[idx]).convert(self.colormodel)
            segmentation = Image.open(self.semantics[idx])
            
            if self.transform:
                image = self.transform(image)
                segmentation = self.transform(segmentation) 
    
            return image,segmentation
        elif self.split == 'test':
            image = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
    
    
    
    def __len__(self):
        return len(self.images)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        original, segmentation = sample['original'], sample['segmentation']
        return {'original': torch.from_numpy(original),
                'segmentation': torch.from_numpy(segmentation)}
        
class RandomCrop(object):
    """
        Crop randomly the image in a sample.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        original, segmentation = sample['original'], sample['segmentation']

        h, w = original.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        # Make sure that the crop is performed is the same area of the image 
        original = original[top: top + new_h,
                            left: left + new_w]
        segmentation = segmentation[top: top + new_h,
                                    left: left + new_w]
        return {'original': original, 'segmentation': segmentation}

"""class MakePatchs(object):
    def __call__(self, sample, ppi):
        for i in range(ppi):

        
        
        return {'patch':patch,'row':row,'column':column,'probability':P}"""

