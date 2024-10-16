import xarray as xr
import numpy as np
import torch
import os
from models.SmaAt_UNet import SmaAt_UNet
import argparse

def main(prog_file, output_path):
    

    return

if __name__ == "__main__":
    # Set up argument prser
    parser = argparse.ArgumentParser(description='Predict State Bias')
    parser.add_argument(
        '--prog_file',
        type=str,
        default=None,
        help='Absolute Path to prognosis file'
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