#importing libraries

import torch
import torch.utils.data as data
import os
import random
from PIL import Image


class CreateDataset(data.Dataset):
    
    def __init__(self , imagedir , subfolder='train' , direction = 'AtoB' , flip = False , transform = None ,resize_scale = None , crop_size = None):
        
        super(CreateDataset , self).__init__()
        self.images_path = os.path.join(imagedir , subfolder)
        self.image_filenames = [name for name in sorted(os.listdir(self.images_path))]
        self.flip = flip
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.direction = direction
        
    
    def __getitem__(self , index):
        
        image_path = os.path.join(self.images_path , self.image_filenames[index])
        img = Image.open(image_path)
        
        if self.direction == 'AtoB':
            
            inp_img = img.crop((0,0,img.width//2 , img.height))
            target_img = img.crop((img.width//2 , 0 , img.width , img.height))
            
        elif self.direction == 'BtoA':
            
            inp_img = img.crop((img.width//2 , 0 , img.width , img.height))
            target_img = img.crop((0,0,img.width//2 , img.height))
            
        if self.resize_scale:
            
            inp_img = inp_img.resize((self.resize_scale , self.resize_scale) , Image.BILINEAR)
            target_img = target_img.resize((self.resize_scale , self.resize_scale) , Image.BILINEAR)
        
        if self.crop_size:
            
            x = random.randint(0 , self.resize_scale - self.crop_size + 1)
            y = random.randint(0 , self.resize_scale - self.crop_size + 1)
            
            inp_img = inp_img.crop((x , y , x + self.crop_size , y + self.crop_size))
            target_img = target_img.crop((x , y , x + self.crop_size , y + self.crop_size))
            
        if self.flip:
            
            if random.random() < 0.5:
                
                inp_img  = inp_img.transpose(Image.FLIP_LEFT_RIGHT)
                target_img = target_img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.transform is not None:
            
            inp_img = self.transform(inp_img)
            target_img = self.transform(target_img)
            
        return inp_img , target_img


    def __len__(self):
        return len(self.image_filenames)             
                