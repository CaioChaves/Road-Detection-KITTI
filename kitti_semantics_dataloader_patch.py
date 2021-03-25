import os
from imageio import imread
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import random
import torchvision

class KittiDatasetPatch(Dataset):
    def __init__(self, rootDir, ppi, patch_size, target_type = 'semantic_binary_ground', transform=None, totensor=None):
        self.rootDir = rootDir
        self.ppi = ppi
        self.patch_size = patch_size
        self.trainingImagesPath = os.path.join(self.rootDir,'data_semantics/training/image_2')  
        self.trainingSegmentationBinaryPath = os.path.join(self.rootDir,'data_semantics/training/semantics_binary_ground')
        self.trainingSegmentationPath = os.path.join(self.rootDir,'data_semantics/training/semantic_rgb')
        self.testingImagesPath = os.path.join(self.rootDir,'data_semantics/testing/image_2')
        self.target_type = target_type
        self.transform = transform
        self.totensortransf = torchvision.transforms.ToTensor() if totensor else {}
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
        image = Image.open(self.images[idx]).convert('RGB')
        segmentation = Image.open(self.semantics[idx])
        if self.transform:
                image = self.transform(image)
                segmentation = self.transform(segmentation) 
        if 0 <= idx <= 9:
                origin_image_name = '00000'+str(idx)+'_10.png'
        elif 10 <= idx <= 99:
                origin_image_name = '0000'+str(idx)+'_10.png'
        elif 100 <= idx <=199:
                origin_image_name = '000'+str(idx)+'_10.png'
        row = random.randint(0,image.shape[1]-self.patch_size)
        column = random.randint(0,image.shape[2]-self.patch_size)
        patch = image[:,row:row+self.patch_size,column:column+self.patch_size]
        segmentation_patch = segmentation[:,row:row+self.patch_size,column:column+self.patch_size]
        P = segmentation_patch.mean()
        D = {'patch':patch,'row':row,'column':column,'probability':P,'origin_image':origin_image_name}
        return D
    
        
    def __len__(self):
        return len(self.images)
    
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

