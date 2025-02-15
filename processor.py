import os 
import torch
import argparse
import numpy as np
from utils.data_geos import geos_dataset
from models.SmaAt_UNet import SmaAt_UNet

class Processor:
    def __init__(self, ckpt_root_path: str = None, *arrays: np.ndarray):
        self.levs= [2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,60,64,69,74,80,85,90,95,100,105,110,115,120,125,130,135,140,145,152,160,170,181]
        # Define the required keys
        self.keys = ['U', 'V', 'T', 'QV', 'QI', 'QL', 'QG', 'QR', 'QS', 'PS']
        # Check that the number of arrays matches the number of keys
        if len(arrays) != len(self.keys):
            raise ValueError(f"Expected {len(self.keys)} arrays, but got {len(arrays)}.")

        self.arrays = dict(zip(self.keys, arrays))
        self.ckpt_root_path = ckpt_root_path

    
    def prog_scale(self):
        # Scale the prognostic variables
        maxs = {
            'U': 193.542038,
            'V': 126.025597,
            'T': 340.213074,
            'QV': 0.024481887,
            'QI': 0.000640826765,
            'QL': 0.0037784921,
            'QG': 0.000279535569,
            'QR': 0.000570683682,
            'QS': 0.000795233995,
            'PS': 105569.586
            }
        mins = {
            'U': -139.770218,
            'V': -150.331345,
            'T': 178.431381,
            'QV': 1.58426859e-17,
            'QI': 0.0,
            'QL': 0.0,
            'QG': 0.0,
            'QR': 0.0,
            'QS': 0.0,
            'PS': 51069.2383,
            }
        for key in self.keys:
            self.arrays[key] = (self.arrays[key] - mins[key]) / (maxs[key] - mins[key])
        return self.arrays
    
    def re_build_pred(self, arrs):
        offset = 90
        xsize = 180
        base = np.zeros((181,450))
        cnt = np.zeros((181,450))
    
        for i in range(len(arrs)):
            base[:,i*offset:(i*offset+xsize)] += arrs[i]
            cnt[:,i*offset:(i*offset+xsize)] += 1.
        base[:,:offset] += base[:, -90:]
        cnt[:, :offset] += 1
        return base[:,:360]/cnt[:, :360]
    
    def get_xy(self):
        # Get the x and y coordinates
        lat = np.linspace(-90, 90, 181)
        lon = np.linspace(-180, 179, 360)
    
        lons, lats = np.meshgrid(lon, lat)
    
        lons_norm = (lons+180.0)/(179.0+180.0)
        lats_norm = (lats+90.0)/180.0
        return lons_norm, lats_norm
    
    def get_levs(self, nlev):
        # Get the model level indices
        levs = np.ones((181, 360))*nlev
        levs_norm = (levs-1.0)/180.0
        return levs_norm
    
    def pred_one_lev(self, nlev, input_img):
        # Predict the state bias at one level

        # Load the model
        ckpt_name = f'model_{nlev}.pt'
        ckpt_path = os.path.join(self.ckpt_root_path, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} not found")
        checkpoint = torch.load(ckpt_path, map_location='cpu')


        model = SmaAt_UNet(n_channels=13, n_classes=1)

        model.load_state_dict(state_dict=checkpoint['state_dict'], strict=False)
        model.to('cpu')

        model.eval()
        
        y_preds = []   
        with torch.no_grad():
            for k in range(input_img.size(0)):
                y_pred = model(input_img[k].unsqueeze(0))
                y_preds.append(y_pred)
        
        y_hat = self.re_build_pred(y_preds)
        return y_hat
    
    def predict(self):
        # Min-max scale the prognostic variables
        prog_scaled = self.prog_scale()
        lon_scaled, lat_scaled = self.get_xy()
        ps_scaled = prog_scaled['PS'].to_numpy().squeeze()

        out_arrs = []
        for nz in self.levs:
            lev_scaled = self.get_levs(nz)
            arrays = [prog_scaled[key][nz] for key in self.keys]
            arrays.extend([ps_scaled, lat_scaled, lon_scaled, lev_scaled])
            vx = np.stack(arrays, axis=0)
            input_img = torch.Tensor(vx)
            tlist = geos_dataset.split_tensor(input_img, tile_size=180, xoffset=90)
            input_img = torch.stack(tlist, dim=0)
            y_hat = self.pred_one_lev(nz, input_img)
            out_arrs.append(y_hat)
        return out_arrs
