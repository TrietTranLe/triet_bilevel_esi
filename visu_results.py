"""
Same as the notebook "visu_results.ipynb" but in a python script.
-> visualize results given an output folder
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

parser = ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("-od", "--output_dir", type=str, help="name of the output directory", required=True)
parser.add_argument("-add_ovr", "--add_overrides", nargs="*", help="additional overrides")
parser.add_argument("-bsl", "--baselines", nargs="+", help="baselines neural network method to use", default=["1dcnn", "lstm"])
parser.add_argument("-mw", "--model_weight", type=str, help="model weight name", default='best_ckpt.ckpt')

parser.add_argument("-test_ovr", "--test_overrides", nargs="*", help="test dataset overrides")
parser.add_argument("-i", "--eval_idx", type=int, help="index of data for visualisation", default=2)
parser.add_argument("-sv", "--surfer_view", type=str, default="lat", help="surfer view if different from the one in the default file")
parser.add_argument("-sh", "--show", action="store_true")
parser.add_argument("-tdc", "--test_data_config", type=str, help="test dataset config file", default='test_dataset.yaml')
parser.add_argument("-tw", "--time_window", action="store_true", help="time window")
parser.add_argument("-ovlp", "--overlap", type=int, default=7, help="overlap for the time window")
parser.add_argument("-wl", "--window_length", type=int, default=7, help="length of time window")
parser.add_argument("-ott", "--on_train", action="store_true", help="on train dataset")
args = parser.parse_args()


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
    "subset_name": "left_back",
    "per_valid": 1,
    "dl_kw":{
        "batch_size": 1, #16
    },
    # "time_window":args.time_window
}

if args.on_train : 
    dm = hydra.utils.call(cfg.datamodule)
else : 
    dm = EsiDatamodule(**datamodule_conf)
     
dm.setup("test")
test_dl = dm.test_dataloader()

fwd = hydra.utils.call(cfg.fwd)
# mne_info = hydra.utils.call(cfg.mne_info)
leadfield = torch.from_numpy(fwd['sol']['data']).float()

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

loaded_mod = torch.load(model_path, map_location=torch.device('cpu'))
litmodel.load_state_dict( loaded_mod['state_dict'] )
litmodel.eval()

### --- LOAD BASELINES --- ###
baselines = args.baselines
baseline_nets = dict(zip(baselines, []*len(baselines)))
baseline_config = OmegaConf.load(str(Path("baselines", "baselines_ses_125ms_cosine.yaml")))
# baseline_config =  OmegaConf.load(args.baseline_config)
for bsl in baselines: 
    baseline_nets[bsl] = load_model_from_conf(bsl, baseline_config)


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
if idx >= batch.input.shape[0]:
    for k in range(idx // batch.input.shape[0]): 
        batch = next(iter_data)
    idx_v = idx % batch.input.shape[0]

## Visu the GT data
plt.figure()
plt.subplot(121)
plt.imshow(batch.input[idx_v,:,:].squeeze().numpy())
plt.colorbar()
plt.axis('off')
plt.title('Input:EEG data')
plt.subplot(122)
plt.imshow(batch.tgt[idx_v,:,:].squeeze().numpy())
plt.axis('off')
plt.colorbar()
plt.title('Output: Source activity')
plt.savefig(Path(figs_path, f"gt_data_{idx}.png"))
if args.show :
    plt.show(block=False)    
else : 
    plt.close()

plt.figure(figsize=(10,5))
plt.subplot(121)
for e in range(test_config.n_electrodes):
    plt.plot(batch.input[idx_v,e,:].squeeze().numpy())
plt.xlabel('Time points')
plt.ylabel('Amplitude')
plt.title('EEG data')
plt.subplot(122)
for s in range(test_config.n_sources): 
    plt.plot(batch.tgt[idx_v,s,:].squeeze().numpy())
plt.xlabel('Time points')
plt.ylabel('Amplitude')
plt.title('Source activity')
plt.tight_layout()
plt.savefig(Path(figs_path, f"gt_data_waveform_{idx}.png"))
if args.show :
    plt.show(block=False)    
else : 
    plt.close()

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
    else:
        output = litmodel(batch).detach()
        output_ae = litmodel.solver.prior_cost.forward_ae( litmodel(batch) ) # check the output of the prior model

# with torch.no_grad():
#     if args.time_window: 
#         windows_input = signal_to_windows(batch.input, window_length=window_length, overlap=overlap, pad=True) 
#         windows_tgt = signal_to_windows(batch.tgt, window_length=window_length, overlap=overlap, pad=True) 
#         windows = TrainingItem(input=windows_input.squeeze(), tgt=windows_tgt.squeeze())
#         output = litmodel(windows)
#         # output = windows.tgt
#         output_ae = litmodel.solver.prior_cost.forward_ae( litmodel(windows) ) # check the output of the prior model
#         output = torch.from_numpy( windows_to_signal_center(output.unsqueeze(1), to_keep=to_keep, overlap=overlap, n_times=batch.input.shape[-1]) )
#         output_ae = torch.from_numpy( windows_to_signal_center(output_ae.unsqueeze(1), to_keep=to_keep, overlap=overlap, n_times=batch.input.shape[-1]) )
#     else :    
#         output = litmodel(batch).detach()
#         output_ae = litmodel.solver.prior_cost.forward_ae( litmodel(batch) ) # check the output of the prior model

plt.figure(figsize=(10,5))
plt.subplot(121)
for s in range(test_config.n_sources):
    plt.plot(batch.tgt[idx_v,s,:].squeeze().numpy())
plt.xlabel('Time points')
plt.ylabel('Amplitude')
plt.title('source GT data')
plt.subplot(122)
for s in range(test_config.n_sources): 
    plt.plot(output[idx_v,s,:].squeeze().numpy())
plt.xlabel('Time points')
plt.ylabel('Amplitude')
plt.title('Source estimated')
plt.tight_layout()
plt.savefig(Path(figs_path, f"gt_and_hat_source_data_waveform_{idx}.png"))
if args.show :
    plt.show(block=False)    
else : 
    plt.close()
# sys.exit()

with torch.no_grad(): 
    lstm_output = baseline_nets["lstm"](batch.input)
    cnn_output = baseline_nets["1dcnn"](batch.input)

## MNE and sLORETA, using mne-python implementation
mne_info = load_mne_info( electrode_montage = test_config.electrode_montage, sampling_freq=512 )
raw_eeg = mne.io.RawArray(batch.input[idx_v,:,:].squeeze().numpy(), mne_info)
raw_eeg.set_eeg_reference(projection=True, verbose=False)
noise_eeg = mne.io.RawArray(
        np.random.randn(batch.input[idx_v,:,:].squeeze().numpy().shape[0], 500), mne_info, verbose=False
    )
noise_cov = mne.compute_raw_covariance(noise_eeg, verbose=False)
lambda2 = 1.0 / (test_config.snr_db**2) ## !! this could be tuned to improve the results


inv_op = mne.minimum_norm.make_inverse_operator(
    info=raw_eeg.info,
    forward=fwd,
    noise_cov=noise_cov,
    loose=0,
    depth=0,
    )

m = "MNE"
stc_mne = mne.minimum_norm.apply_inverse_raw(
    raw=raw_eeg, inverse_operator=inv_op, lambda2=lambda2, method=m
)

m = "sLORETA"
stc_slo = mne.minimum_norm.apply_inverse_raw(
    raw=raw_eeg, inverse_operator=inv_op, lambda2=lambda2, method=m
)

mne_output = stc_mne.data
slo_output = stc_slo.data
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

img_gt = plot_source_estimate(src=batch.tgt[idx_v,:,:].detach().numpy(), t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view)
img_4dvar = plot_source_estimate(src=output[idx_v,:,:].detach().numpy(), t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view)
img_4dvar_ae = plot_source_estimate(src=output_ae[idx_v,:,:].detach().numpy(), t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view)

img_lstm = plot_source_estimate(src=lstm_output[idx_v,:,:].numpy(), t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view)
img_cnn = plot_source_estimate(src=cnn_output[idx_v,:,:].numpy(), t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view)

img_mne = plot_source_estimate(src=mne_output, t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view)
img_slo = plot_source_estimate(src=slo_output, t_max=t_max, fwd=fwd,fs=512, surfer_view=args.surfer_view)

# visu GT, output from 4dvarnet and phi(output) : check if the prior autoencoder is working / learned
plot_src_from_imgs({"GT":img_gt, "4DVar": img_4dvar, "phi(4DVar)": img_4dvar_ae}, ["GT","4DVar", "phi(4DVar)"])
plt.savefig(Path(figs_path, f"cortex_check_ae_idx_{idx}.png"))
if args.show: 
    plt.show(block=False)
else :
    plt.close()

plot_src_from_imgs({"GT":img_gt, "4DVar": img_4dvar, "LSTM":img_lstm, "1DCNN":img_cnn}, ["GT","4DVar", "LSTM", "1DCNN"])
plt.savefig(Path(figs_path, f"cortex_learning_based_idx_{idx}.png"))
if args.show: 
    plt.show(block=False)
else :
    plt.close()

plot_src_from_imgs({"GT":img_gt, "4DVar": img_4dvar, "LSTM":img_lstm, "1DCNN":img_cnn, "MNE": img_mne, "sLORETA":img_slo}, ["GT","4DVar", "LSTM", "1DCNN", "MNE", "sLORETA"])
plt.savefig(Path(figs_path, f"cortex_all_methods_{idx}.png"))
if args.show: 
    plt.show(block=False)
else :
    plt.close()

plot_src_from_imgs({"MNE": img_mne, "sLORETA":img_slo}, ["MNE", "sLORETA"])
plt.savefig(Path(figs_path, f"cortex_nl_based_idx_{idx}.png"))
if args.show: 
    plt.show(block=False)
else :
    plt.close()

## Check the whole data as an image 
plt.figure(figsize=(15,6)) 
plt.subplot(161)
plt.imshow(batch.tgt[idx_v,:,:].squeeze().numpy()) 
plt.axis('off')
plt.colorbar()
plt.title('GT')
plt.subplot(162)
plt.imshow(output[idx_v,:,:].squeeze().detach().numpy())
plt.axis('off')
plt.colorbar()
plt.title('4DVar')
plt.subplot(163)
plt.imshow(lstm_output[idx_v,:,:].squeeze().numpy())
plt.axis('off')
plt.colorbar()
plt.title('LSTM')
plt.subplot(164)
plt.imshow(cnn_output[idx_v,:,:].squeeze().detach().numpy())
plt.axis('off')
plt.colorbar()
plt.title('1DCNN')
plt.subplot(165)
plt.imshow(mne_output.squeeze())
plt.axis('off')
plt.colorbar()
plt.title('MNE')
plt.subplot(166)
plt.imshow(slo_output.squeeze())
plt.axis('off')
plt.colorbar()
plt.title('sLORETA')
plt.savefig(Path(figs_path, f"source_2dt_idx_{idx}.png"))
if args.show: 
    plt.show(block=False)
else: 
    plt.close()


## Variational cost during the gradient descent (and obs and prior cost separately)
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(litmodel.solver.varc, '-', marker='o')
plt.xlabel('step')
plt.title("var cost")

plt.subplot(1,3,2)
plt.plot( litmodel.solver.obsc, '-', marker='o' )
plt.title("obs cost")
plt.xlabel('step')

plt.subplot(1,3,3)
plt.plot( litmodel.solver.pc , '-',  marker='o')
plt.title("prior cost")
plt.xlabel('step')
plt.savefig(Path(figs_path,f"var_cost_idx_{idx}.png"))
if args.show: 
    plt.show(block=False)
else: 
    plt.close() 


src_hat = {
    "GT": batch.tgt[idx_v,:,:].detach().numpy(),
    "4Dvar": output[idx_v,:,:].detach().numpy(),
    "LSTM": lstm_output[idx_v,:,:].numpy(),
    "1DCNN": cnn_output[idx_v,:,:].numpy(),
    "MNE": mne_output,
    "sLORETA": slo_output
}

fig, axes = plt.subplots(figsize=(16, 10), nrows=1, ncols=len(list(src_hat.keys())))
i = 0
for m in src_hat.keys(): 
    tp = t_max / mne_info['sfreq']
    eeg_hat = fwd['sol']['data'] @ src_hat[m]
    eeg_hat = eeg_hat / np.abs(eeg_hat).max()
    reproj = mne.EvokedArray( data = eeg_hat, info = mne_info, tmin=-0.)
    reproj.plot_topomap(
        times=tp,
        colorbar=False, 
        axes = axes[i]
    )
    axes[i].set_title(f"{m}")
    i += 1
plt.savefig(Path(figs_path, f"eeg_reproj_idx_{idx}.png"))

if args.show:
    plt.show(block=False)
else:
    plt.close()
