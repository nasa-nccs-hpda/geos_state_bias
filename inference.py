import xarray as xr
import numpy as np
import torch
import os
from models.SmaAt_UNet import SmaAt_UNet
from utils.data_geos import geos_dataset
import argparse
import time

def re_build_pred(arrs):
    offset = 90
    xsize = 180
    base = np.zeros((181,450)) # HxW_extend
    cnt = np.zeros((181,450))
    
    for i in range(len(arrs)):
        base[:,i*offset:(i*offset+xsize)] += arrs[i]
        cnt[:,i*offset:(i*offset+xsize)] += 1.
    base[:,:offset] += base[:, -90:]
    cnt[:, :offset] += 1
    return base[:,:360]/cnt[:, :360]

def create_ds(current_time, levels):
    time = current_time + np.timedelta64(3, 'h')
    print(time)
    lat = np.linspace(-90, 90, 181)  # 181 steps
    lon = np.linspace(-180, 179, 360)  # 360 steps
    lev = np.array(levels)

    # Create the data array filled with zeros
    dtdtml = np.zeros((1, len(lev), len(lat), len(lon)))

    # Create the xarray dataset
    ds = xr.Dataset(
        {
            "DTDTML": (["time", "lev", "lat", "lon"], dtdtml)
        },
        coords={
	    "time": [time],
            "lat": lat,
            "lon": lon,
            "lev": lev
        }
    )
    return ds

def norm_prog(prog_file: str) -> xr.Dataset:
    prog_vars = ['U', 'V', 'T', 'QV', 'QI', 'QL', 'QG', 'QR', 'QS', 'PS']
    prog_data = xr.open_dataset(prog_file)
    prog_data = prog_data[prog_vars]
    maxs = prog_data.map(np.max)
    mins = prog_data.map(np.min)
    
    prog_scaled = xr.Dataset()
    for var in prog_data.data_vars:
        prog_scaled[var] = (prog_data[var]-mins[var])/(maxs[var]-mins[var])
    return prog_scaled

def scale_iau(predicts, min=-0.00300, max=0.00240):
    # min=-0.0036053888, max=0.0027486791  ## Global min/max of 2000-2002 DJF
    return predicts*(max-min)+min

def get_xys() -> np.ndarray:
    """
    return:
        np.ndarray
    """
    lat = np.linspace(-90, 90, 181)
    lon = np.linspace(-180, 179, 360)
    
    lons, lats = np.meshgrid(lon, lat)
    
    lons_norm = (lons+180.0)/(179.0+180.0)
    lats_norm = (lats+90.0)/180.0
    return lons_norm, lats_norm

def get_levs(nlev) -> np.ndarray:
    levs = np.ones((181, 360))*nlev
    levs_norm = (levs-1.0)/180.0
    return levs_norm

def gen_filename(current_time, prog_file):
    base_name = os.path.basename(prog_file)
    base_name = base_name.replace("prog", "iau_ml")
    ct_stamp = current_time.astype("datetime64[us]").item().strftime("%Y%m%d_%H%Mz")
    nt_stamp = (current_time + np.timedelta64(3, 'h')).astype("datetime64[us]").item().strftime("%Y%m%d_%H%Mz")
    base_name = base_name.replace(ct_stamp, nt_stamp)
    return base_name

def pred(nlev, X):
    root = "./checkpoints/batch_2"
    ckpt_name = f"model_{nlev}.pt"
    ckpt_path = os.path.join(root, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model = SmaAt_UNet(
        n_channels=13,
        n_classes=1)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict=state_dict, strict=False)
    model.to("cpu")
    model.eval()

    y_preds = []
    """
    for k in range(len(X)):
        tx = torch.unsqueeze(X[k,...],0)
        ty = model(tx).squeeze()
        y_preds.append(ty.detach().numpy())
    y_hat = re_build_pred(y_preds)
    """
    for k in [0, 2]:
        tx = torch.unsqueeze(X[k,...],0)
        ty = model(tx).squeeze()
        y_preds.append(ty.detach().numpy())
    y_hat = np.vstack(y_preds)
    return y_hat

def main(prog_file, output_path):
    levs = [2,5,8,11,14,17,20,23,26,29,32,35,38,41,44,47,50,53,56,60,64,69,74,80,85,90,95,100,105,110,115,120,125,130,135,140,145,152,160,170,181]


    # Load min-max scaled prog variables and min/max values
    prog_vars = ['U', 'V', 'T', 'QV', 'QI', 'QL', 'QG', 'QR', 'QS']
    prog_scaled = norm_prog(prog_file)
    lon_scaled, lat_scaled = get_xys()
    ps_scaled = prog_scaled['PS'].to_numpy().squeeze() 

    # Init output xr.dataset
    ct = prog_scaled.time.values[0]
    ds_out = create_ds(ct, levs)
    out_fn = gen_filename(ct, prog_file)
    out_file = os.path.join(output_path, out_fn)
    for nz in levs:
        lev_scaled = get_levs(nz)
        arrays = [prog_scaled[v].isel(lev=(nz-1)).to_numpy().squeeze() for v in prog_vars]
        arrays.extend([ps_scaled, lat_scaled, lon_scaled, lev_scaled])
        vx = np.stack(arrays, axis=0)
        input_img = torch.Tensor(vx)
        tlist = geos_dataset.split_tensor(input_img, tile_size=180, xoffset=90)
        input_img = torch.stack(tlist, dim=0)
        y_hat = pred(nz, input_img)
        y_hat = scale_iau(y_hat)
        
        ds_out['DTDTML'].loc[dict(lev=nz)] = np.expand_dims(y_hat, axis=0)
    ds_out.to_netcdf(path=out_file) 
    return

if __name__ == "__main__":
    start_time = time.time()
    # Set up argument prser
    parser = argparse.ArgumentParser(description='Predict State Bias')
    parser.add_argument(
        '--prog_file',
        type=str,
        default=None,
        help='Absolute Path to prognostic file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./',
        help = 'Path to save output result'
    )
    # Parse arguments
    args = parser.parse_args()

    main(args.prog_file, args.output_path)
    print("Single Prog prediction in %s mins"%((time.time()-start_time)/60.0))
