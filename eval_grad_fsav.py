"""
Same as the notebook "visu_results.ipynb" but in a python script.
-> visualize results given an output folder
## with modification to use fsav994 head model
"""
import sys
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
from torch import nn

from contrib.eeg.data import EsiDatamodule, EsiDataset
from contrib.eeg.prior import ConvAEPrior
from contrib.eeg.grad_models import RearrangedConvLstmGradModel_n
from contrib.eeg.solvers import EsiBaseObsCost, EsiGradSolver_n, EsiLitModule
from contrib.eeg.optimizers import optim_adam_gradphi
from contrib.eeg.cost_funcs import Cosine, CosineReshape
from contrib.eeg.models_directinv import HeckerLSTM, HeckerLSTMpl
from contrib.eeg.utils_eeg import (load_fwd, load_mne_info,
                                   load_model_from_conf, plot_source_estimate,
                                   plot_src_from_imgs)
from contrib.eeg.utils_eeg import signal_to_windows, windows_to_signal, windows_to_signal_center
from scipy.io import loadmat

home = os.path.expanduser('~')
anatomy_folder = Path(f"{home}/Documents/DATA/fsaverage/constrained/standard_1020/fsav_994/model/anatomy")


parser = ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("-od", "--output_dir", type=str, help="name of the output directory", required=True)
parser.add_argument("-add_ovr", "--add_overrides", nargs="*", help="additional overrides")
parser.add_argument("-bsl", "--baselines", nargs="+", help="baselines neural network method to use", default=["1dcnn", "lstm"])
parser.add_argument("-bsl_conf", "--baseline_config", type=str, default="baselines_fsav.yaml")
parser.add_argument("-mw", "--model_weight", type=str, help="model weight name", default='best_ckpt.ckpt')


parser.add_argument("-test_ovr", "--test_overrides", nargs="*", help="test dataset overrides")
parser.add_argument("-i", "--eval_idx", type=int, help="index of data for visualisation", default=2)
parser.add_argument("-sv", "--surfer_view", type=str, default="lat", help="surfer view if different from the one in the default file")
parser.add_argument("-sh", "--show", action="store_true")
parser.add_argument("-tdc", "--test_data_config", type=str, help="test dataset config file", default='test_ses_sereega_fsav994_125ms.yaml')
parser.add_argument("-tw", "--time_window", action="store_true", help="time window")
parser.add_argument("-ovlp", "--overlap", type=int, default=7, help="overlap for the time window")
parser.add_argument("-wl", "--window_length", type=int, default=7, help="length of time window")
parser.add_argument("-ott", "--on_train", action="store_true", help="on train dataset")
parser.add_argument("-sub", "--subset_name", type=str, default="left_back", help="name of the training subset to use")
args = parser.parse_args()

#-----
hemi = "lh" if args.subset_name.split('_')[0]=="left" else "rh" 
pl.seed_everything(333)
device = torch.device("cpu")

output_dir = args.output_dir
config_path = Path( f"{args.output_dir}", ".hydra" )
#overrides_name = Path("overrides.yaml")
## load the overrides file
#overr = OmegaConf.load(Path(config_path, overrides_name))
#if args.add_overrides : 
#    overr = overr + args.add_overrides

# init hydra config
with initialize(config_path=str(config_path), version_base=None):
    cfg = compose(config_name="config.yaml")#, overrides=overr)

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

test_data_config = str(Path('config', 'dataset', args.test_data_config))
test_config = OmegaConf.load(test_data_config)
if args.test_overrides :
    for override in args.test_overrides : 
        arg_path, value = override.split("=") 
        arg_path_list = arg_path.split('.')
        
        current_config = test_config
        for arg in arg_path_list : 
            current_config = getattr(current_config, arg)
        value_type = type(current_config)
        OmegaConf.update( test_config, arg_path, value_type(value) )

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
    "per_valid": 1,
    "dl_kw":{
        "batch_size": 32, #16
    },
    # "time_window":args.time_window
}
cfg.datamodule = datamodule_conf
cfg.fwd._target_ = "contrib.eeg.utils_eeg.load_fwd_fsav"
if args.on_train : 
    dm = hydra.utils.call(cfg.datamodule)
else : 
    dm = EsiDatamodule(**datamodule_conf)
     
dm.setup("test")
test_dl = dm.test_dataloader()
#----
fwd = hydra.utils.call(cfg.fwd)
# mne_info = hydra.utils.call(cfg.mne_info)
leadfield = torch.from_numpy(fwd['sol']['data']).float()

### load vertices data to view source ###
### load the 2 source spaces and region mapping
model_folder = Path( f"{test_config.datafolder}/{test_config.subject_name}/{test_config.orientation}/{test_config.electrode_montage}/{test_config.source_sampling}/model" )
fwd_vertices = mne.read_forward_solution(
    f"{model_folder}/fwd_verticesfsav_994-fwd.fif"
)
fwd_vertices = mne.convert_forward_solution(
    fwd_vertices, surf_ori=True, force_fixed=True, use_cps=True, verbose=0
)
fwd_regions = mne.read_forward_solution(f"{model_folder}/fwd_fsav_994-fwd.fif")
fwd_regions = mne.convert_forward_solution(
    fwd_regions, surf_ori=True, force_fixed=True, use_cps=True, verbose=0
)

## !! assign fwd_region the proper leadfield matrix values (summed version)
fwd_regions["sol"]["data"] = leadfield.numpy()

region_mapping = loadmat(f"{model_folder}/fs_cortex_20k_region_mapping.mat")[
    "rm"
][0]

n_vertices = fwd_vertices["nsource"]
n_regs = len(np.unique(region_mapping))
#-----
### --- LOAD MODEL --- ###
import sys

model_path = Path( output_dir, "lightning_logs", "checkpoints", args.model_weight )
if os.path.isfile(model_path): 
    print("Model exists \U0001F44D")
else: 
    print("try other model path")
    print(model_path)
    sys.exit()

litmodel = hydra.utils.call(cfg.litmodel)
if litmodel.solver.init_type == "direct":
    init_model = hydra.utils.call(cfg.init_model.model)
    litmodel.solver.load_init_model(init_model)
loaded_mod = torch.load(model_path, map_location=torch.device('cpu'))
litmodel.load_state_dict( loaded_mod['state_dict'] )
litmodel.eval()

#---
## visu results ##
on_tw = "tw" if args.time_window else ""
if args.on_train :
    figs_path =  Path(output_dir, "figs", on_tw, "train")
else : 
    figs_path =  Path(output_dir, "figs", on_tw, "test")

os.makedirs( figs_path , exist_ok=True)

iter_data = iter(test_dl)
batch = next(iter_data)
idx = args.eval_idx
idx_v = idx
# if idx >= batch.input.shape[0]:
#     for k in range(idx // batch.input.shape[0]): 
#         batch = next(iter_data)
#     idx_v = idx % batch.input.shape[0]

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt']) # batch format for 4DVARNET
eeg, src = dm.test_ds[idx_v]
batch = TrainingItem(input=eeg.unsqueeze(0), tgt=src.unsqueeze(0))
idx_v = 0

#---
overlap=args.overlap
window_length = args.window_length
to_keep = 4
print(overlap)
with torch.no_grad():
    if args.time_window: 
        windows_input = signal_to_windows(batch.input, window_length=window_length, overlap=overlap, pad=True) 
        windows_tgt = signal_to_windows(batch.tgt, window_length=window_length, overlap=overlap, pad=True) 
        windows = TrainingItem(input=windows_input.squeeze(), tgt=windows_tgt.squeeze())
        output = litmodel(windows)
        # output = windows.tgt
        output_ae = litmodel.solver.prior_cost.forward_ae( litmodel(windows) ) # check the output of the prior model
        output = torch.from_numpy( windows_to_signal(output.unsqueeze(1), overlap=overlap, n_times=batch.input.shape[-1]) )
        output_ae = torch.from_numpy( windows_to_signal(output_ae.unsqueeze(1), overlap=overlap, n_times=batch.input.shape[-1]) )
    else :    
        output = litmodel(batch).detach()
        output_ae = litmodel.solver.prior_cost.forward_ae( litmodel(batch) ) # check the output of the prior model
        gt_ae = litmodel.solver.prior_cost.forward_ae( batch.tgt ) 



## eval time / visu time : max activity
t_max = np.argmax( batch.tgt[idx_v,:,:].squeeze().abs().sum(0).numpy() )

######
# 24 oct
normalised_output = output.squeeze()[:,t_max] / output.abs().squeeze()[:,t_max].max()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(fwd['source_rr'][:,0], fwd['source_rr'][:,1], fwd['source_rr'][:,2], c=normalised_output, marker='o')
plt.show(block=False)

#####
# compute metrics on the given data
from contrib.eeg import utils_eeg as utl
from contrib.eeg import metrics as met
seeds = dm.test_ds.md[idx]["seeds"]
if type(seeds) is int:
    seeds = [seeds]
s = seeds[0]

def reg_to_verts_src(src_reg, n_vertices, region_mapping, fwd_vertices, fs=512): 
    n_regs = np.unique(region_mapping).shape[0]
    src_vertices = np.zeros((n_vertices, src_reg.shape[-1]))
    for r in range(n_regs):
        src_vertices[np.where(region_mapping == r)[0], :] = src_reg[r, :]
    stc = mne.SourceEstimate(
        data=src_vertices,
        vertices=[fwd_vertices["src"][0]["vertno"], fwd_vertices["src"][1]["vertno"]],
        tmin=0.0,
        tstep=1 / fs,
        subject="fsaverage",
    )
    return stc, src_vertices

###
mne_info = load_mne_info( electrode_montage = test_config.electrode_montage, sampling_freq=512 )
fs = mne_info['sfreq']
_, gt_verts = reg_to_verts_src(batch.tgt[idx_v,:,:].detach().squeeze().numpy(), n_vertices, region_mapping, fwd_vertices, fs)
img_gt = plot_source_estimate(src=gt_verts, t_max=t_max, fwd=fwd_vertices,fs=512, surfer_view=args.surfer_view)
_, dvar_verts = reg_to_verts_src(output[idx_v,:,:].detach().squeeze().numpy(), n_vertices, region_mapping, fwd_vertices, fs)
img_4dvar = plot_source_estimate(src=dvar_verts, t_max=t_max, fwd=fwd_vertices,fs=512, surfer_view=args.surfer_view)

gradients = litmodel.solver.gradients
gradients = gradients[len(gradients) - 50:]
moment = dict()
i_moment = 0
second_moment = dict()
i_second_moment = 0
lr = dict()
i_lr = 0
grad = dict()
i_grad = 0
lr_grad = dict()
i_lr_grad = 0
for i in range(len(gradients)):
    _, dvar_verts = reg_to_verts_src(gradients[i][:,:], n_vertices, region_mapping, fwd_vertices, fs)
    img_4dvar = plot_source_estimate(src=dvar_verts, t_max=t_max, fwd=fwd_vertices,fs=512, surfer_view=args.surfer_view)

    if i % 5 == 0:
        moment[f"step: {i_moment}"] = img_4dvar
        i_moment += 1
    elif i % 5 == 1:
        second_moment[f"step: {i_second_moment}"] = img_4dvar
        i_second_moment += 1
    elif i % 5 == 2:
        lr[f"step: {i_lr}"] = img_4dvar
        i_lr += 1
    elif i % 5 == 3:
        grad[f"step: {i_grad}"] = img_4dvar
        i_grad += 1
    else:
        lr_grad[f"step: {i_lr_grad}"] = img_4dvar
        i_lr_grad += 1
        
plot_src_from_imgs(moment, list(moment.keys()))
plt.savefig(Path(figs_path, f"cortex_moment_idx_{idx}.png"))
if args.show:
    plt.show(block=False)
else :
    plt.close()
    
plot_src_from_imgs(second_moment, list(second_moment.keys()))
plt.savefig(Path(figs_path, f"cortex_2_moment_idx_{idx}.png"))
if args.show:
    plt.show(block=False)
else :
    plt.close() 
        
plot_src_from_imgs(lr, list(lr.keys()))
plt.savefig(Path(figs_path, f"cortex_lr_idx_{idx}.png"))
if args.show:
    plt.show(block=False)
else :
    plt.close()

plot_src_from_imgs(grad, list(grad.keys()))
plt.savefig(Path(figs_path, f"cortex_grad_idx_{idx}.png"))
if args.show:
    plt.show(block=False)
else :
    plt.close()

plot_src_from_imgs(lr_grad, list(lr_grad.keys()))
plt.savefig(Path(figs_path, f"cortex_lr_grad_idx_{idx}.png"))
if args.show:
    plt.show(block=False)
else :
    plt.close()