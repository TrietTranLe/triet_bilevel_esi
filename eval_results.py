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
from contrib.eeg.prior import ConvAEPrior
from contrib.eeg.grad_models import RearrangedConvLstmGradModel_n, RearrangedConvLstmGradModel_x
from contrib.eeg.solvers import EsiBaseObsCost, EsiGradSolver_n, EsiLitModule
from contrib.eeg.optimizers import optim_adam_gradphi
from contrib.eeg.cost_funcs import Cosine, CosineReshape
from contrib.eeg.models_directinv import HeckerLSTM, HeckerLSTMpl
from contrib.eeg.utils_eeg import (load_fwd, load_mne_info,
                                   load_model_from_conf, plot_source_estimate,
                                   plot_src_from_imgs)
from contrib.eeg.utils_eeg import signal_to_windows, windows_to_signal

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
parser.add_argument("-ott", "--on_train", action="store_true", help="on train dataset")
parser.add_argument("-tw", "--time_window", action="store_true", help="time window")
parser.add_argument("-ovlp", "--overlap", type=int, default=7, help="overlap for the time window")
parser.add_argument("-wl", "--window_length", type=int, default=7, help="length of time window")
parser.add_argument("-nf", "--noise_fixed", action="store_true", help="use a dataset with fixed noise")
parser.add_argument("-m", "--method", type=str, help="method to evaluate", default="4dvar")
args = parser.parse_args()

pl.seed_everything(333)
device = torch.device("cpu")

overlap=args.overlap
window_length = args.window_length

output_dir = args.output_dir
config_path = Path( f"{args.output_dir}", ".hydra" )
# overrides_name = Path("overrides.yaml")
# load the overrides file
# overr = OmegaConf.load(Path(config_path, overrides_name))
# if args.add_overrides : 
#    overr = overr + args.add_overrides

# init hydra config
with initialize(config_path=str(config_path), version_base=None):
    cfg = compose(config_name="config.yaml")#, overrides=overr)

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])


test_config = OmegaConf.load(
    str(Path('config', 'dataset', args.test_data_config))
)

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
    "subset_name": "left_back", #"none", 
    "dl_kw":{
        "batch_size": 32
    },
    "noise_fixed": args.noise_fixed
}

if args.on_train : 
    dm = hydra.utils.call(cfg.datamodule)
else : 
    dm = EsiDatamodule(**datamodule_conf)
     
dm.setup("test")
test_dl = dm.test_dataloader()
n_times = next(iter(test_dl))[0].shape[2]
fwd = hydra.utils.call(cfg.fwd)
# mne_info = hydra.utils.call(cfg.mne_info)
leadfield = torch.from_numpy(fwd['sol']['data']).float()

### --- LOAD MODEL --- ###

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



if args.on_train :
    figs_path =  Path(output_dir, "metrics", "train")
else : 
    figs_path =  Path(output_dir, "metrics", "test")

os.makedirs( figs_path , exist_ok=True)

mne_info = load_mne_info( electrode_montage = test_config.electrode_montage, sampling_freq=512 )
fs = np.floor(mne_info['sfreq'])
t_vec = np.arange(0, n_times / fs, 1 / fs)
spos = torch.from_numpy(fwd['source_rr'])


neighbors = utl.get_neighbors(
    [fwd["src"][0]["use_tris"], fwd["src"][1]["use_tris"]],
    [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]],
)
linear_methods = ["MNE", "sLORETA"]
# nn_methods = ["4dvar"] + baselines
# methods = ["gt"] + nn_methods + linear_methods 
nn_methods = baselines + ["4dvar"]
methods = ["gt"] + linear_methods + nn_methods
# methods = ["gt"] + [args.method]
##### eval #####
n_val_samples = len(dm.test_ds)
# nmse_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
# loc_error_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
# psnr_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
# time_error_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
# auc_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}

nmse_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
loc_error_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
psnr_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
time_error_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
auc_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
overlapping_regions = 0
for k in tqdm(range(n_val_samples)):
    eeg_gt, src_gt = dm.test_ds[k]
    eeg_gt, src_gt = eeg_gt.float().clone(), src_gt.float().clone()

    eeg_gt_unscaled = eeg_gt.clone() * dm.test_ds.max_eeg[k]
    src_gt_unscaled = src_gt.clone() * dm.test_ds.max_src[k]
    #####  
    # data covariance:
    # activity_thresh = 0.1
    # noise_cov, data_cov, nap = inv.mne_compute_covs(
    #    (M_unscaled).numpy(), mne_info, activity_thresh
    # )
    ### TEST BETTER NOISE COV
    raw_noise = mne.io.RawArray(
        data=np.random.randn(cfg.dataset.n_electrodes, 600),
        info=mne_info,verbose=False
    )
    noise_cov = mne.compute_raw_covariance(raw_noise, verbose=False)   

    raw_eeg = mne.io.RawArray(
        data=eeg_gt_unscaled, info=mne_info, first_samp=0.0, verbose=False
    )
    rraw_eeg = mne.set_eeg_reference(raw_eeg, "average", projection=True, verbose=False)[0]

    stc_hat = dict(zip(methods, [None] * len(methods)))
    stc_hat["gt"] = src_gt_unscaled.numpy()
    lambda2 = 1.0 / (cfg.datamodule.dataset_kw.snr_db**2)
    ###
    for m in methods:
        
        if m in linear_methods:
            inv_op = mne.minimum_norm.make_inverse_operator(
                info=raw_eeg.info,
                forward=fwd,
                noise_cov=noise_cov,
                loose=0,
                depth=0,
                verbose=False
            )
            stc_hat[m] = mne.minimum_norm.apply_inverse_raw(
                raw=raw_eeg, inverse_operator=inv_op, lambda2=lambda2, method=m, verbose=False
            )
            stc_hat[m] = stc_hat[m].data
        
        elif m in baselines : 
            with torch.no_grad():
                batch = TrainingItem(
                    input= eeg_gt.float().unsqueeze(0), 
                    tgt= src_gt.float().unsqueeze(0)) 
                stc_hat[m] = baseline_nets[m](batch.input).detach().squeeze()  

            stc_hat[m] = stc_hat[m].detach().numpy() * dm.test_ds.max_src[k].numpy()
        
        elif m == "4dvar":
            batch = TrainingItem(
                input= eeg_gt.float().unsqueeze(0).clone(), 
                tgt= src_gt.float().unsqueeze(0).clone()) 

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
                    output_ae = litmodel.solver.prior_cost.forward_ae( litmodel(batch) )
                stc_hat[m] = output.squeeze() 
            stc_hat[m] = stc_hat[m].detach().numpy() * dm.test_ds.max_src[k].numpy()

#------------------------------------------------------------------------------------------------------------------------#
        src_hat = torch.from_numpy(stc_hat[m])
        le = 0
        te = 0
        nmse = 0
        auc_val = 0
        seeds_hat = []
        ## check for overlap ------ @TODO : fix ok for 2 sources, not for more
        seeds = dm.test_ds.md[k]["seeds"]
        if type(seeds) is int:
            seeds = [seeds]

        patches = [ [] for _ in range(len(seeds)) ]
        for kk in range(len(seeds)) : 
            patches[kk] = dm.test_ds.md[k]['act_src'][f'patch_{kk+1}'] 
        if len(seeds) > 1:
            inter = list( 
                set(patches[0]).intersection(patches[1])
            )
            if len(inter)>0 : # for overlapping regions : only keep seed with max activity
                overlapping_regions += 1
                to_keep = torch.argmax( torch.Tensor([src_gt[seeds[0], :].abs().max(), src_gt[seeds[1], :].abs().max() ]) )
                seeds = [ seeds[to_keep] ]
        ## ----------------------
        act_src = [ s for l in patches for s in l ]
            # compute metrics -----------------------------------------------------------------
        for kk in range(len(seeds)) :
            s = seeds[kk]
            other_sources = np.setdiff1d(
                act_src, patches[kk]
            ) # source from other patches
            t_eval_gt = torch.argmax(src_gt[s, :].abs())
                # find estimated seed, in a neighboring area
            eval_zone = utl.get_patch(order=7, idx=s, neighbors=neighbors)
            ## remove sources from other patches of the eval zone (case of close sources regions) ##
            eval_zone = np.setdiff1d(eval_zone, other_sources)

            # find estimated seed, in a neighboring area
            # eval_zone = utl.get_patch(order=2, idx=s, neighbors=neighbors)
            s_hat = eval_zone[torch.argmax(src_hat[eval_zone, t_eval_gt].abs())]

            t_eval_pred = torch.argmax(src_hat[s_hat, :].abs())

            le += torch.sqrt(((spos[s, :] - spos[s_hat, :]) ** 2).sum())
            te += np.abs(t_vec[t_eval_gt] - t_vec[t_eval_pred])
            auc_val += met.auc_t(
                src_gt_unscaled, src_hat, t_eval_gt, thresh=True, act_thresh=0.0
            )  # probablement peut mieux faire

            #nmse += met.nmse_t_fn(j_unscaled, j_hat, t_eval_gt)
            nmse_tmp = ( (
                src_gt_unscaled[:,t_eval_gt] / src_gt_unscaled[:,t_eval_gt].abs().max() - src_hat[:,t_eval_gt] / src_hat[:,t_eval_gt].abs().max() 
                )**2 ).mean()
            nmse += nmse_tmp
                
            seeds_hat.append(s_hat)

        le = le / len(seeds)
        te = te / len(seeds)
        nmse = nmse / len(seeds)
        auc_val = auc_val / len(seeds)
        tmaxs_pred = torch.argmax(src_hat[seeds_hat, :].abs(), dim=1)
        # time error (error on the instant of the max. activity):
        time_error_dict[m][k] = te
        # print(f"time error: {time_error*1e3} [ms]")

        # localisation error
        loc_error_dict[m][k] = le*1e3
        # print(f"localisation error: {loc_error*1e3} [mm]")

        # instant nMSE:
        nmse_dict[m][k] = nmse
        # print(f"nmse at instant of max activity: {nmse_t:.4f}")

        # PSNR
        psnr_dict[m][k] = psnr(
            (src_gt_unscaled / src_gt_unscaled.abs().max()).numpy(),
            (src_hat / src_hat.abs().max()).numpy(),
            data_range=(
                (src_gt_unscaled / src_gt_unscaled.abs().max()).min()
                - (src_hat / src_hat.abs().max()).max()
            ),
        )
        # print(f"psnr for total source distrib: {psnr_val:.4f} [dB]")

        # AUC
        # act_src = esi_datamodule.val_ds.act_src[k]
        auc_dict[m][k] = auc_val
        # print(f"auc: {auc_val:.4f}")

        # change plots to visu. multiple sources
        idx_max_gt = seeds[0]
        idx_max_pred = seeds_hat[0]
#####################################################################
#############################################################################
import pandas as pd
import os

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
    
