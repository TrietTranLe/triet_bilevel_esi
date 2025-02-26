import numpy as np
import torch 
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from contrib.eeg.utils_eeg import signal_to_windows, windows_to_signal
from collections import namedtuple
from contrib.eeg import utils_eeg as utl
from contrib.eeg import metrics as met

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt']) # batch format for 4DVARNET

def eval_fn(dataset, fwd, mne_info, lin_methods, nn_methods, snr_db=5, time_window=False):  
    methods = ['gt'] + lin_methods + list( nn_methods.keys() )
    n_val_samples = len(dataset)
    n_electrodes = dataset[0][0].shape[0]
    snr_db = snr_db
    window_length = 16 
    overlap = 15

    neighbors = utl.get_neighbors(
        [fwd["src"][0]["use_tris"], fwd["src"][1]["use_tris"]],
        [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]],
    )
    spos = torch.from_numpy(fwd['source_rr'])
    fs = np.floor(mne_info['sfreq'])
    n_times = dataset[0][0].shape[1]
    t_vec = np.arange(0, n_times / fs, 1 / fs)
    
    nmse_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
    loc_error_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
    psnr_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
    time_error_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
    auc_dict = {method: np.empty((n_val_samples, 1)) for method in methods}
    overlapping_regions = 0
    
    for k in tqdm(range(n_val_samples)):
        eeg_gt, src_gt = dataset[k]
        eeg_gt, src_gt = eeg_gt.float().clone(), src_gt.float().clone()

        eeg_gt_unscaled = eeg_gt.clone() * dataset.max_eeg[k]
        src_gt_unscaled = src_gt.clone() * dataset.max_src[k]
        #####  
        # data covariance:
        # activity_thresh = 0.1
        # noise_cov, data_cov, nap = inv.mne_compute_covs(
        #    (M_unscaled).numpy(), mne_info, activity_thresh
        # )
        ### TEST BETTER NOISE COV
        random_state = np.random.get_state()
        raw_noise = mne.io.RawArray(
            data=np.random.randn(n_electrodes, 600),
            info=mne_info,verbose=False
        )
        np.random.set_state(random_state)
        noise_cov = mne.compute_raw_covariance(raw_noise, verbose=False)   

        raw_eeg = mne.io.RawArray(
            data=eeg_gt_unscaled, info=mne_info, first_samp=0.0, verbose=False
        )
        raw_eeg = mne.set_eeg_reference(raw_eeg, "average", projection=True, verbose=False)[0]

        stc_hat = dict(zip(methods, [None] * len(methods)))
        # stc_hat["gt"] = src_gt_unscaled.numpy()
        lambda2 = 1.0 / (snr_db**2)
        ###
        for m in methods:
            if m == "gt":
                stc_hat[m] = src_gt_unscaled.numpy()
            elif m in lin_methods:
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

            elif m in nn_methods :
                batch = TrainingItem(
                        input= eeg_gt.float().unsqueeze(0).clone(), 
                        tgt= src_gt.float().unsqueeze(0).clone())
                if m == "4dvar": 
                    with torch.no_grad():
                        if time_window: 
                            windows_input = signal_to_windows(batch.input, window_length=window_length, overlap=overlap, pad=True) 
                            windows_tgt = signal_to_windows(batch.tgt, window_length=window_length, overlap=overlap, pad=True) 
                            windows = TrainingItem(input=windows_input.squeeze(), tgt=windows_tgt.squeeze())
                            output = nn_methods[m](windows)
                            # output = windows.tgt
                            output_ae = nn_methods[m].solver.prior_cost.forward_ae( nn_methods[m](windows) ) # check the output of the prior model
                            output = torch.from_numpy( windows_to_signal(output.unsqueeze(1), overlap=overlap, n_times=batch.input.shape[-1]) )
                            output_ae = torch.from_numpy( windows_to_signal(output_ae.unsqueeze(1), overlap=overlap, n_times=batch.input.shape[-1]) )
                        else :    
                            output = nn_methods[m](batch).detach()
                            output_ae = nn_methods[m].solver.prior_cost.forward_ae( nn_methods[m](batch) )
                    stc_hat[m] = output.squeeze() 
                    stc_hat[m] = stc_hat[m].detach().numpy() * dataset.max_src[k].numpy()
                else :
                    with torch.no_grad():
                        stc_hat[m] = nn_methods[m](batch.input).detach().squeeze()  
                    stc_hat[m] = stc_hat[m].detach().numpy() * dataset.max_src[k].numpy()

    #------------------------------------------------------------------------------------------------------------------------#
            src_hat = torch.from_numpy(stc_hat[m])
            le = 0
            te = 0
            nmse = 0
            auc_val = 0
            seeds_hat = []
            ## check for overlap ------ @TODO : fix ok for 2 sources, not for more
            seeds = dataset.md[k]["seeds"]
            if type(seeds) is int:
                seeds = [seeds]

            patches = [ [] for _ in range(len(seeds)) ]
            for kk in range(len(seeds)) : 
                patches[kk] = dataset.md[k]['act_src'][f'patch_{kk+1}'] 
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
    return loc_error_dict, auc_dict, nmse_dict, psnr_dict, time_error_dict