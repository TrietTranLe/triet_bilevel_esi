"""
evaluate results
"""

import csv
import os
from argparse import ArgumentParser
from collections import namedtuple
from datetime import datetime
from functools import partial
from pathlib import Path

import einops
import hydra
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch import nn
from tqdm import tqdm
import sys 

import contrib.eeg.utils_eeg as utl
from contrib.eeg import metrics as met
from contrib.eeg.data import EsiDatamodule
from contrib.eeg.model_short import (ConvAEPrior, EsiBaseObsCost,
                                     EsiGradSolver, ESILitModule,
                                     RearrangedConvLstmGradModel,
                                     optim_adam_gradphi)
from contrib.eeg.models import Cosine, CosineReshape
from contrib.eeg.models_directinv import HeckerLSTM, HeckerLSTMpl
from contrib.eeg.utils_eeg import (load_fwd, load_mne_info,
                                   load_model_from_conf, plot_source_estimate,
                                   plot_src_from_imgs)
from contrib.eeg.utils_eeg import signal_to_windows, windows_to_signal
from contrib.eeg.evaluations import eval_fn

parser = ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("-od", "--output_dir", type=str, help="name of the output directory", required=True)
parser.add_argument("-add_ovr", "--add_overrides", nargs="*", help="additional overrides")
parser.add_argument("-bsl", "--baselines", nargs="+", help="baselines neural network method to use", default=["1dcnn", "lstm"])

parser.add_argument("-test_ovr", "--test_overrides", nargs="*", help="test dataset overrides")
parser.add_argument("-i", "--eval_idx", type=int, help="index of data for visualisation", default=2)
parser.add_argument("-sv", "--surfer_view", type=str, default="lat", help="surfer view if different from the one in the default file")
parser.add_argument("-sh", "--show", action="store_true")
parser.add_argument("-tdc", "--test_data_config", type=str, help="test dataset config file", default='test_dataset.yaml')
parser.add_argument("-ott", "--on_train", action="store_true", help="on train dataset")
parser.add_argument("-tw", "--time_window", action="store_true", help="time window")
parser.add_argument("-ovlp", "--overlap", type=int, default=7, help="overlap for the time window")
parser.add_argument("-wl", "--window_length", type=int, default=7, help="length of time window")
parser.add_argument("-nf", "--noise_fixed", action="store_true", help="use a dataset with fixed noise")
parser.add_argument("-m", "--method", type=str, help="method to evaluate", default="4dvar")
parser.add_argument("-sbset", "--subset_name", type=str, help="subset name", default="left_back")
args = parser.parse_args()


pl.seed_everything(333)
device = torch.device("cpu")
TrainingItem = namedtuple('TrainingItem', ['input', 'tgt']) # batch format for 4DVARNET

# Load the data
# if time windows: overlap and window length
overlap = args.overlap
window_length = args.window_length
# test dataset configuration file:
test_config = OmegaConf.load( str(Path('config', 'dataset', args.test_data_config)) )
# override some parameters in command line
if args.test_overrides :
    for override in args.test_overrides : 
        arg_path, value = override.split("=") 
        arg_path_list = arg_path.split('.')
        
        current_config = test_config
        for arg in arg_path_list : 
            current_config = getattr(current_config, arg)
        value_type = type(current_config)
        OmegaConf.update( test_config, arg_path, value_type(value) )
# create test datamodule configuration
datamodule_conf = {
    "dataset_kw": 
    {
        "datafolder":test_config.datafolder, 
        "simu_name": test_config.simu_name,
        "subject_name": test_config.subject_name,
        "source_sampling": test_config.source_sampling,
        "electrode_montage": test_config.electrode_montage,
        "orientation": test_config.orientation,
        "to_load": test_config.to_load,
        "snr_db": 5,
        "noise_type": {"white":1.},
        "scaler_type": 'linear_bis',
        "replace_root": True
    },
    "subset_name": args.subset_name,
    "dl_kw":{
        "batch_size": 16
    },
    "noise_fixed": args.noise_fixed
}


## load the config to load the model (and load data if evaluation on train dataset)
output_dir = args.output_dir
config_path = Path( f"{args.output_dir}", ".hydra" )

# init hydra config
with initialize(config_path=str(config_path), version_base=None):
    cfg = compose(config_name="config.yaml")#, overrides=overr)

if args.on_train : 
    dm = hydra.utils.call(cfg.datamodule)
else : 
    dm = EsiDatamodule(**datamodule_conf)
     
dm.setup("test") # always loading the dataset without shuffling data - important to get the correct "metadata" information of the data (seed, act_sources...)
test_dl = dm.test_dataloader()
# additional data : forward object (leadfield), mne_info
fwd = hydra.utils.call(cfg.fwd)
leadfield = torch.from_numpy(fwd['sol']['data']).float()
# source space information : positions, mesh triangles -> neighbors
spos = torch.from_numpy(fwd['source_rr'])
neighbors = utl.get_neighbors(
    [fwd["src"][0]["use_tris"], fwd["src"][1]["use_tris"]],
    [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]],
)
# temporal information
mne_info = hydra.utils.call(cfg.mne_info)
n_times = next(iter(test_dl))[0].shape[2]
fs = np.floor(mne_info['sfreq'])
t_vec = np.arange(0, n_times / fs, 1 / fs)


########### Load the model(s) ############################
# 4dvarnet
model_path = Path( output_dir, "lightning_logs", "checkpoints", "best_ckpt.ckpt" )
if os.path.isfile(model_path): 
    print("Model exists \U0001F44D")
else: 
    print("try other model path")
    print(model_path)
    sys.exit()

litmodel = hydra.utils.call(cfg.litmodel)

loaded_mod = torch.load(model_path, map_location=torch.device('cpu'))
litmodel.load_state_dict( loaded_mod['state_dict'] )
litmodel.eval()

### --- LOAD BASELINES --- ###
baselines = args.baselines
baseline_nets = dict(zip(baselines, []*len(baselines)))
baseline_config = OmegaConf.load(str(Path("baselines", "baselines_ses_retrained.yaml")))
for bsl in baselines: 
    baseline_nets[bsl] = load_model_from_conf(bsl, baseline_config)

## create saving directory
if args.on_train :
    figs_path =  Path(output_dir, "metrics", "train")
else : 
    figs_path =  Path(output_dir, "metrics", "test")

os.makedirs( figs_path , exist_ok=True)


# Evaluation
##########################################################################################
loc_error_dict, auc_dict, nmse_dict, psnr_dict, time_error_dict = eval_fn(
    dataset=dm.test_ds, fwd=fwd, mne_info=mne_info, 
    lin_methods=['MNE', 'sLORETA'], nn_methods={'4dvar':litmodel, '1dcnn':baseline_nets['1dcnn'], 'lstm': baseline_nets['lstm']}, 
)
# Save the results ##################################################################################


os.makedirs(Path(output_dir, "evals", "test", args.test_data_config), exist_ok=True)
## first : save every value, for each metric and each method : 
metrics = {
    "LE":loc_error_dict, 
    "nMSE":nmse_dict, 
    "AUC":auc_dict, 
    "PSNR":psnr_dict, 
    "TE":time_error_dict
    }
for me in metrics :
    list_of_arrays = [metrics[me][m].squeeze() for m in metrics[me].keys()]
    df = pd.DataFrame(data = list_of_arrays).T
    df.columns = list(metrics[me].keys())

    df.to_csv(Path(output_dir, "evals", "test", args.test_data_config,  f"{me.upper()}.csv"), index=False)

 
# save mean and std value
data = {
    'metric': list(metrics.keys())
    }
for method in methods: 
    data.update( 
        {method: [metrics[metric][method].mean() for metric in metrics.keys()]}
)
df = pd.DataFrame(data=data)
df = df.set_index('metric')
df.to_csv(Path(output_dir, "evals", "test",args.test_data_config, "MEANS.csv"), float_format='%.4f')
## std
data = {
    'metric': list(metrics.keys())
    }
for method in methods: 
    data.update( 
        {method: [metrics[metric][method].std() for metric in metrics.keys()]}
)
df = pd.DataFrame(data=data)
df = df.set_index('metric')
df.to_csv(Path(output_dir, "evals", "test", args.test_data_config, "STDS.csv"), float_format='%.4f')

for method in methods:
    print(f" >>>>>>>>>>>>>>> Results method {method} <<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"mean localisation error: {loc_error_dict[method].mean()} [mm]")
    print(f"auc: {auc_dict[method].mean():.4f}")
    print(f"mean nmse at instant of max activity: {nmse_dict[method].mean():.4f}")
    print(f"psnr for total source distrib: {psnr_dict[method].mean():.4f} [dB]")
    print(f"mean time error: {time_error_dict[method].mean()*1e3} [ms]")
    