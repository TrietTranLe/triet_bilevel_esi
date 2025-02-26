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

### --- LOAD BASELINES --- ###
baselines = args.baselines
baseline_config_path=args.baseline_config
baseline_nets = dict(zip(baselines, []*len(baselines)))
baseline_config = OmegaConf.load( str(Path("baselines", baseline_config_path)) )
# baseline_config =  OmegaConf.load(args.baseline_config)
for bsl in baselines: 
    baseline_nets[bsl] = load_model_from_conf(bsl, baseline_config)

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

#----
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

#---

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
# compute metrics on the given data
from contrib.eeg import utils_eeg as utl
from contrib.eeg import metrics as met
seeds = dm.test_ds.md[idx]["seeds"]
if type(seeds) is int:
    seeds = [seeds]
s = seeds[0]

if test_config.source_sampling == "fsav_994": 
    neighbors = np.squeeze( loadmat(f"{anatomy_folder}/fs_cortex_20k_region_mapping.mat")['nbs'] )
    # reshape neighbors
    l_max           = np.max( np.array([len(l[0]) for l in neighbors]) )
    neighb_array    = np.zeros( (len(neighbors), l_max) )
    for i in range(len(neighbors) ) : 
        l = neighbors[i][0]
        neighb_array[i,:len(l)] = l -1
        if len(l)<l_max: 
            neighb_array[i,len(l):] = None 
    neighb_array = neighb_array.astype(np.int64)

    neighbors = neighb_array.copy() 
    del neighb_array

else : 
    neighbors = utl.get_neighbors(
    [fwd["src"][0]["use_tris"], fwd["src"][1]["use_tris"]],
    [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]],
)

#---
def le_and_auc(fwd, neighbors, s, src, src_hat): 
    src_hat = torch.from_numpy(src_hat)
    spos = torch.from_numpy(fwd['source_rr'])
    t_eval_gt = torch.argmax(src[s, :].abs())
    # find estimated seed, in a neighboring area
    
    eval_zone = utl.get_patch(order=7, idx=s, neighbors=neighbors)
    # find estimated seed, in a neighboring area
    s_hat = eval_zone[torch.argmax(src_hat[eval_zone, t_eval_gt].abs())]
    le = torch.sqrt(((spos[s, :] - spos[s_hat, :]) ** 2).sum())
    auc_val = met.auc_t(
        src, src_hat, t_eval_gt, thresh=True, act_thresh=0.0
        )  # probablement peut mieux faire
    return le*1e3, auc_val #le in mm

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
#---
le_4dvar, auc_4dvar = le_and_auc(fwd, neighbors, s, src, output[idx_v,:,:].detach().numpy() )
le_lstm, auc_lstm = le_and_auc(fwd, neighbors, s, src, lstm_output[idx_v,:,:].numpy() )
le_cnn, auc_cnn = le_and_auc(fwd, neighbors, s, src, cnn_output[idx_v,:,:].numpy() )
le_mne, auc_mne = le_and_auc(fwd, neighbors, s, src, mne_output )
le_slo, auc_slo = le_and_auc( fwd, neighbors, s, src, slo_output )

###
fs = mne_info['sfreq']
_, gt_verts = reg_to_verts_src(batch.tgt[idx_v,:,:].detach().squeeze().numpy(), n_vertices, region_mapping, fwd_vertices, fs)
img_gt = plot_source_estimate(src=gt_verts, t_max=t_max, fwd=fwd_vertices,fs=512, surfer_view=args.surfer_view)
_, dvar_verts = reg_to_verts_src(output[idx_v,:,:].detach().squeeze().numpy(), n_vertices, region_mapping, fwd_vertices, fs)
img_4dvar = plot_source_estimate(src=dvar_verts, t_max=t_max, fwd=fwd_vertices,fs=512, surfer_view=args.surfer_view)
_, dvar_verts_ae = reg_to_verts_src(output_ae[idx_v,:,:].detach().squeeze().numpy(), n_vertices, region_mapping, fwd_vertices, fs)
img_4dvar_ae = plot_source_estimate(src=dvar_verts_ae, t_max=t_max, fwd=fwd_vertices,fs=512, surfer_view=args.surfer_view)
_, gt_ae_verts = reg_to_verts_src(gt_ae[idx_v,:,:].detach().squeeze().numpy(), n_vertices, region_mapping, fwd_vertices, fs)
img_gt_ae = plot_source_estimate(src=gt_ae_verts, t_max=t_max, fwd=fwd_vertices,fs=512, surfer_view=args.surfer_view)



_, lstm_verts = reg_to_verts_src(lstm_output[idx_v,:,:].squeeze().numpy(), n_vertices, region_mapping, fwd_vertices, fs)
img_lstm = plot_source_estimate(src=lstm_verts, t_max=t_max, fwd=fwd_vertices,fs=512, surfer_view=args.surfer_view)
_, cnn_verts = reg_to_verts_src(cnn_output[idx_v,:,:].squeeze().numpy(), n_vertices, region_mapping, fwd_vertices, fs)
img_cnn = plot_source_estimate(src=cnn_verts, t_max=t_max, fwd=fwd_vertices,fs=512, surfer_view=args.surfer_view)

_, mne_verts = reg_to_verts_src(mne_output, n_vertices, region_mapping, fwd_vertices, fs)
_, slo_verts = reg_to_verts_src(slo_output, n_vertices, region_mapping, fwd_vertices, fs)
img_mne = plot_source_estimate(src=mne_verts, t_max=t_max, fwd=fwd_vertices,fs=512, surfer_view=args.surfer_view)
img_slo = plot_source_estimate(src=slo_verts, t_max=t_max, fwd=fwd_vertices,fs=512, surfer_view=args.surfer_view)
#---
# visu GT, output from 4dvarnet and phi(output) : check if the prior autoencoder is working / learned

plot_src_from_imgs({"GT":img_gt, "4DVar": img_4dvar, "phi(4DVar)": img_4dvar_ae, "phi(gt)": img_gt_ae}, ["GT","4DVar", "phi(4DVar)", "phi(gt)"])
plt.savefig(Path(figs_path, f"cortex_check_ae_idx_{idx}.png"))
if args.show: 
    plt.show(block=False)
else :
    plt.close()


# plot_src_from_imgs({"GT":img_gt, "Ours (A + C)": img_4dvar}, ["GT", "Ours (A + C)"])
# plt.savefig(Path(figs_path, f"cortex_check_ae_idx_{idx}.png"))
# if args.show: 
#     plt.show(block=False)
# else :
#     plt.close()

plot_src_from_imgs({"GT":img_gt, "4DVar": img_4dvar, "LSTM":img_lstm, "1DCNN":img_cnn}, ["GT","4DVar", "LSTM", "1DCNN"])
plt.savefig(Path(figs_path, f"cortex_learning_based_idx_{idx}.png"))
if args.show: 
    plt.show(block=False)
else :
    plt.close()

plot_src_from_imgs({"GT":img_gt, "4DVar": img_4dvar, "LSTM":img_lstm, "1DCNN":img_cnn, "MNE": img_mne, "sLORETA":img_slo}, 
                   ["GT","4DVar", "LSTM", "1DCNN", "MNE", "sLORETA"])
                   # subtitles=["\n0 -1"] + [f"\n{le:.3f} - {auc*100:.2f}" for le,auc in [(le_4dvar, auc_4dvar), (le_lstm, auc_lstm), (le_cnn, auc_cnn), (le_mne, auc_mne), (le_slo, auc_slo)]])
# plot_src_from_imgs({"GT":img_gt, "4DVar": img_4dvar, "MNE": img_mne, "sLORETA":img_slo}, 
#                    ["GT","4DVar", "MNE", "sLORETA"], 
#                    subtitles=["\n0 -1"] + [f"\n{le:.3f} - {auc*100:.2f}" for le,auc in [(le_4dvar, auc_4dvar), (le_mne, auc_mne), (le_slo, auc_slo)]])

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

#---
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
plt.plot( litmodel.solver.pc, '-',  marker='o')
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
#---
coss = CosineReshape()

print(f"4DVAR cossim: {coss(output.detach(), batch.tgt.detach()):.4f}")
print(f"LSTM cossim: {coss(lstm_output, batch.tgt.detach()):.4f}")
print(f"CNN cossim: {coss(cnn_output.detach(), batch.tgt.detach()):.4f}")
print(f"MNE cossim: {coss(torch.from_numpy(mne_output).unsqueeze(0), batch.tgt.detach()):.4f}")
print(f"sLO cossim: {coss(torch.from_numpy(slo_output).unsqueeze(0), batch.tgt.detach()):.4f}")

## obs cost
print(f"GT obs cost: {coss( leadfield @ batch.tgt.detach() , batch.input.detach() ) }")
print(f"4DVAR obs cost: {coss( leadfield @ output.detach() , batch.input.detach() ) }")
print(f"LSTM obs cost: {coss( leadfield @ lstm_output.detach() , batch.input.detach() ) }")
print(f"CNN obs cost: {coss( leadfield @ cnn_output.detach() , batch.input.detach() ) }")
print(f"MNE obs cost: {coss( leadfield @ torch.from_numpy(mne_output).unsqueeze(0).float() , batch.input.detach() ) }")
print(f"sLO obs cost: {coss( leadfield @ torch.from_numpy(slo_output).unsqueeze(0).float() , batch.input.detach() ) }")
######### MSE 
print(f"4DVAR nMSE: {met.nmse_fn(output.detach().squeeze(), batch.tgt.detach().squeeze())* 1e3:.4f}")
print(f"LSTM nMSE: {met.nmse_fn(lstm_output.squeeze(), batch.tgt.detach().squeeze())* 1e3:.4f}")
print(f"CNN nMSE: {met.nmse_fn(cnn_output.detach().squeeze(), batch.tgt.detach().squeeze())* 1e3:.4f}")
print(f"MNE nMSE: {met.nmse_fn(torch.from_numpy(mne_output), batch.tgt.detach().squeeze())* 1e3:.4f}")
print(f"sLO nMSE: {met.nmse_fn(torch.from_numpy(slo_output), batch.tgt.detach().squeeze())* 1e3:.4f}")

## obs cost
print(f"GT obs cost nMSE: {met.nmse_fn( leadfield @ batch.tgt.detach().squeeze() , batch.input.detach().squeeze() ) * 1e3}")
print(f"4DVAR obs cost nMSE: {met.nmse_fn( leadfield @ output.detach().squeeze() , batch.input.detach().squeeze() ) * 1e3}")
print(f"LSTM obs cost nMSE: {met.nmse_fn( leadfield @ lstm_output.detach().squeeze() , batch.input.detach().squeeze() ) * 1e3}")
print(f"CNN obs cost nMSE: {met.nmse_fn( leadfield @ cnn_output.detach().squeeze() , batch.input.detach().squeeze() ) * 1e3}")
print(f"MNE obs cost nMSE: {met.nmse_fn( leadfield @ torch.from_numpy(mne_output).float() , batch.input.detach().squeeze() ) * 1e3}")
print(f"sLO obs cost nMSE: {met.nmse_fn( leadfield @ torch.from_numpy(slo_output).float() , batch.input.detach().squeeze() ) * 1e3}")

#---

def dice_metric(x, x_hat, threshold): 
    # suppose x is a binary array
    # binarize x_hat
    x_hat_bin = torch.zeros_like(x_hat)
    x_hat_bin[x_hat.abs()>threshold*x_hat.abs().max()] = 1

    tp = ( x_hat_bin * x ).sum() 
    fp = ( x_hat_bin *(1-x) ).sum()
    fn = ( (1-x_hat_bin) * x ).sum()

    return 2*tp / (2*tp + fp + fn)
    # return tp, fp, fn
src_bin = torch.zeros_like(src) 
src_bin[batch.tgt.detach().squeeze() > 1e-6] = 1
t_eval = src.abs().sum(0).argmax()
src_bin = src_bin[:, t_eval.item()]
print(f"4DVAR dice: {dice_metric(src_bin, output.detach().squeeze()[:,t_eval], 0.2):.4f}")
print(f"LSTM dice: {dice_metric(src_bin, lstm_output.squeeze()[:,t_eval], 0.2):.4f}")
print(f"CNN dice: {dice_metric(src_bin, cnn_output.detach().squeeze()[:,t_eval], 0.2):.4f}")
print(f"MNE dice: {dice_metric(src_bin, torch.from_numpy(mne_output).squeeze()[:,t_eval], 0.2):.4f}")
print(f"sLORETA dice: {dice_metric(src_bin, torch.from_numpy(slo_output).squeeze()[:,t_eval], 0.2):.4f}")


#---