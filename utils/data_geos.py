from torch.utils.data import Dataset
import numpy as np
import glob
import torch
import math

class geos_dataset(Dataset):
    
    def __init__(self, ftr_path, tgt_path, lev=0, split=False):
        self.ftr_path = ftr_path
        self.tgt_path = tgt_path
        
        self.split = split # whether break input image to tiles
        
        self.transform = None
        
        self.nlev = 181
        self.nx = 360
        self.ny = 181
        
        self.level = lev
        self.features = sorted(glob.glob(f'{self.ftr_path}/features/*'))
        self.targets = sorted(glob.glob(f'{self.tgt_path}/target/*'))
        self.n_imgs = len(self.features)
        assert len(self.features) == len(self.targets)
    
    @staticmethod    
    def split_tensor(tensor, tile_size=256, xoffset=256):
        tiles = []
        # padding along X
        tensor = torch.cat((tensor, tensor[..., 0:90]), dim=-1)
        
        h, w = tensor.size(-2), tensor.size(-1)
        for x in range(int(math.ceil(w/xoffset))-1):
            tiles.append(tensor[..., xoffset*x:min(xoffset*x+tile_size, w)])
        #if tensor.is_cuda:
        #     base_tensor = torch.zeros(tensor.size(), device=tensor.get_device())
        #else: 
        #     base_tensor = torch.zeros(tensor.size())
        return tiles #, base_tensor
        
    def _get_np_array(self, fn):
        tmp = np.load(fn)
        if tmp.ndim == 1:
            tmp = tmp[:, None]
        img4d = tmp.reshape((self.nlev, self.ny, self.nx, -1)) # zHWC
        img = img4d[self.level, ...].squeeze()    # HWC
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))        # CHW
            
        return img
    
    def _get_target_array(self, fn):
        tmp = np.load(fn)
        img = tmp.reshape((self.nlev, self.ny, self.nx))
        return img
    
    def __getitem__(self, index):
        #print(self.targets[index])
        input_img = torch.Tensor(self._get_np_array(self.features[index]))
        if self.transform is not None:
            input_img = self.transform(input_img)
        

        
        target_img = torch.Tensor(self._get_np_array(self.targets[index]))
        if self.transform is not None:
            target_img = self.transform(target_img)
        
        if self.split:
            tlist = geos_dataset.split_tensor(input_img, tile_size=180, xoffset=90)
            input_img = torch.stack(tlist, dim=0)
            tlist = geos_dataset.split_tensor(target_img, tile_size=180, xoffset=90)
            target_img = torch.stack(tlist, dim=0)
        
        return input_img, target_img

    def __len__(self):
        return self.n_imgs
        
        