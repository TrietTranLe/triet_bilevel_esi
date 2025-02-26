from contrib.eeg.cost_funcs import *
import pytorch_lightning as pl
import torch 
from torch import nn
import numpy as np

def Cosine(x, y):
    return (1 - F.cosine_similarity(einops.rearrange(x, 'b c h -> b (c h)'), einops.rearrange(y, 'b c h -> b (c h)')))


## obs cost:
class EsiBaseObsCost(nn.Module):
    """
    Observation cost : D(Y, LX) where Y is the EEG data, L is the leadfield matrix and X the source data.
    - forward_obj: forward object from mne-python for which fwd['sol']['data'] contains the leadfield matrix.
    - device : device to use
    - cost_fn : cost function to use
    """
    def __init__(self, forward_obj, cost_fn = CosineReshape(), attention_hidden_size = 128, attention_kernel_size = 3) -> None:
        super().__init__()
        self.register_buffer('leadfield', torch.from_numpy(forward_obj['sol']['data']).float(), persistent=False)
        self.cost_fn = cost_fn
        self.fwd=forward_obj
        
        # Attention
        # self.attention_hidden_size = attention_hidden_size
        # bias = True
        # self.conv = nn.Conv1d(in_channels  = attention_input_size[0],
        #                       out_channels = attention_hidden_size,
        #                       kernel_size  = attention_kernel_size,
        #                       stride       = 2,
        #                       padding      = 1,
        #                       bias         = bias)
        # self.fc = nn.Linear(in_features  = attention_hidden_size*attention_input_size[1]//2,
        #                     out_features = attention_hidden_size,
        #                     bias         = bias)
        # self.fc_v = nn.Linear(in_features  = 1,
        #                       out_features = attention_hidden_size,
        #                       bias         = bias)
        # self.fc_end = nn.Linear(in_features  = attention_hidden_size,
        #                         out_features = 1,
        #                         bias         = bias)
    
    def forward(self, state, inp):
        """
        batch.input = EEG data
        state = estimated source data
        """
        # x = torch.matmul(self.leadfield, state)
        # y = inp
        # return self._attention(x, y, Cosine(x, y))
        # return self.cost_fn(torch.matmul(self.leadfield, state), inp)
        return self.cost_fn(torch.matmul(self.leadfield, state), inp)
    
    def Lx(self, state):
        return torch.matmul(self.leadfield, state)
    
    # def _attention(self, x, y, v):
    #     x = self.fc(F.relu(torch.flatten(self.conv(x), 1, -1))).unsqueeze(-1)
    #     y = self.fc(F.relu(torch.flatten(self.conv(y), 1, -1))).unsqueeze(-2)
    #     v = self.fc_v(v.unsqueeze(-1)).unsqueeze(-1)
    #     return self.fc_end((F.softmax((x @ y)/(self.attention_hidden_size ** 0.5), dim=-1) @ v).squeeze(-1)).abs().mean()


### solver
class EsiGradSolver(nn.Module) : 
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad, fwd, noise_ampl=1e-3, prior_ponde=1, obs_ponde=1, init_type = "noise", mne_info=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_grad = lr_grad
        self.fwd = fwd
        self.mne_info = mne_info
        self.inv_op = None 
        self.init_type = init_type
        
        self.reg = 1/25
        self.prior_ponde = prior_ponde
        self.obs_ponde = obs_ponde
        self.noise_ampl = noise_ampl

        self._grad_norm = None
        
        self.gradients = []

    def forward(self, batch):
        self.varc = []
        self.pc = []
        self.obsc = []
        self.pc_grad = []
        self.obsc_grad = []
        self.varc_grad_mod = []
        
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.tgt)
            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            if not self.training:
                state = self.prior_cost.forward_ae(state)
            return state

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init
        else:
            if self.init_type.upper() == "MNE":
                state_0 = state_0 = self.mnep_init(batch.input)
            elif self.init_type.upper() == "NOISE":
                state_0 = self.noise_init(batch.tgt)
            elif self.init_type.upper() == "ZEROS":
                state_0 = self.zeros_init(batch.tgt)
            elif self.init_type.upper() == "DIRECT":
                state_0 = self.direct_init(batch.input)
            else:
                raise Exception(f"{self.init_type=} unknown state init type")
            return state_0.detach().requires_grad_(True)

    def load_init_model(self, init_model):
        if self.init_type.upper() == "DIRECT":      
            self.init_model = init_model
            self.init_model.eval()
            self.init_model.to(device = next(self.parameters()).device)
            for p in self.init_model.parameters():
                p.requires_grad = False
        else:
            print(f"Current init state mode: {self.init_type.upper()}. Please change init state mode: DIRECT")

    def direct_init(self, y):
        with torch.no_grad(): 
            direct_sol = self.init_model( y )
        return direct_sol

    def mnep_invop(self, mne_info, fwd, method="MNE"): 
        """
        Compute the inverse operator for the minimum norm solution *method* (e.g MNE) based on the mne-python algorithms.
        intput : 
        - mne_info : mne-python *info* object associated with the eeg data
        - fwd : mne-python forward operator linked with the simulated data (head model)
        - method : method to use (MNE, sLORETA, dSPM, eLORETA c.f mne-python documentation on minimum-norm inverse solutions)
        output : 
        - K : inverse operator (torch.tensor)
        """
        import mne
        from mne.minimum_norm.inverse import (_assemble_kernel,
                                              _check_or_prepare)
        ## compute a "fake" noise covariance
        random_state = np.random.get_state() # avoid changing all random number generation when using MNE init
        noise_eeg = mne.io.RawArray(
                np.random.randn(len(mne_info['chs']), 600), mne_info, verbose=False
            )
        np.random.set_state(random_state)
        noise_cov = mne.compute_raw_covariance(noise_eeg, verbose=False)
        ## compute the inverse operator (K)
        inv_op = mne.minimum_norm.make_inverse_operator(
            info=mne_info,
            forward=fwd,
            noise_cov=noise_cov,
            loose=0,
            depth=0,
            verbose=False
        )

        inv = _check_or_prepare(
            inv_op, 1, self.reg, method ,None,False
        )
        
        K, _, _, _ = _assemble_kernel(
                inv, label=None, method=method, pick_ori=None, use_cps=True, verbose=False
            )
        return torch.from_numpy(K)
    
    def mnep_init(self, y): 
        """  
        Inverse problem resolution : estimates x from y, using the inverse operator K (based on mne-python algorithms). 
        input : 
        - y : eeg data (batch, channel, time)
        - method : method to use (MNE, sLORETA, dSPM, eLORETA c.f mne-python documentation on minimum-norm inverse solutions)
        """
        y = y.float()
        if self.inv_op is None: 
            self.inv_op = self.mnep_invop( self.mne_info, self.fwd.copy(), self.init_type ).float().to(device = next(self.parameters()).device)
        
        return torch.matmul(self.inv_op.to(device = next(self.parameters()).device), y)

    def noise_init(self, x):
        noise = torch.randn(*x.shape, device=x.device)
        return self.noise_ampl*(noise/noise.max())

    def zero_init(self, x):
        return torch.zeros_like(x, device=x.device)


class EsiGradSolver_l(EsiGradSolver) : 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def solver_step(self, state, batch, step):
        obs_cost = self.obs_ponde*self.obs_cost(state, batch)
        prior_cost = self.prior_ponde**2*self.prior_cost(state)
        var_cost = obs_cost + prior_cost
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
        
        self.varc.append(var_cost.item())
        self.pc.append(prior_cost.item())
        self.obsc.append(obs_cost.item())

        return self.grad_mod(state, grad)


class EsiGradSolver_m(EsiGradSolver) : 
    def __init__(self, cost_fn, **kwargs):
        super().__init__(**kwargs)
        self.cost_fn = cost_fn

    def solver_step(self, state, batch, step):
        obs_cost = self.obs_ponde*self.cost_fn.forward_y(self.obs_cost.Lx(state), batch.input)
        prior_cost = self.prior_ponde*self.cost_fn.forward_x(state, self.prior_cost.forward_ae(state))
        var_cost = obs_cost + prior_cost
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
        self.varc.append(var_cost.item())
        self.pc.append(prior_cost.item())
        self.obsc.append(obs_cost.item())

        state_update = self.grad_mod(grad)
        state_update = self.lr_grad*state_update
        return state - state_update


class EsiGradSolver_n(EsiGradSolver) : 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def solver_step(self, state, batch, step):
        obs_cost = self.obs_ponde*self.obs_cost(state, batch.input)
        prior_cost = self.prior_ponde**2*self.prior_cost(state)
        var_cost = obs_cost + prior_cost
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
        self.varc.append(var_cost.item())
        self.pc.append(prior_cost.item())
        self.obsc.append(obs_cost.item())

        # state_update, self.gradients = self.grad_mod(state, grad, self.gradients)
        state_update = self.grad_mod(grad)
        # state_update = (
        #         1 / (step + 1) * state_update
        #         + self.lr_grad * (step + 1) / self.n_step * grad
        # )
        state_update = self.lr_grad*state_update
        return state - state_update


## lightning module
class EsiLitModule(pl.LightningModule):
    """
    lightning module to put it all together and train the model
    """
    def __init__(self, solver, opt_fn, loss_fn, noise_std=None):
    #def __init__(self, solver, opt_fn, lr, loss_fn, Lambda, noise_std=None):
        super().__init__()
        self.solver = solver
        self.opt_fn = opt_fn
        #self.lr = lr

        self.loss_fn = loss_fn
        #self.Lambda = Lambda
        self.noise_std = noise_std

    def forward(self, batch):
        return self.solver(batch)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")[0]

    def step(self, batch, batch_idx, phase=""):
        out = self(batch=batch)
        if self.noise_std and phase == "val":
            out = self.solver.prior_cost.forward_ae(out)

        x = batch.tgt
        if self.noise_std and phase == "train":
            x = batch.tgt + torch.distributions.normal.Normal(0, self.noise_std).sample(batch.tgt.shape).to(batch.tgt.device)
        
        loss = self.loss_fn(batch.tgt, out) + self.loss_fn(x, self.solver.prior_cost.forward_ae(x))
        # loss = self.loss_fn(batch.tgt, out) + self.Lambda*self.loss_fn(x, self.solver.prior_cost.forward_ae(x))

        with torch.no_grad():
            self.log("model params mean", np.array([p.detach().cpu().mean() for p in self.solver.prior_cost.parameters()]).mean(), prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_data_loss", self.loss_fn( batch.tgt, out ), prog_bar=False, on_step=False, on_epoch=True)
            self.log(f"{phase}_ae_loss", self.loss_fn(out, self.solver.prior_cost.forward_ae(out)), prog_bar=False, on_step=False, on_epoch=True)

        return loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)
        # return self.opt_fn(
        # [
        #     {"params": self.solver.grad_mod.parameters(), "lr": self.lr},
        #     {"params": self.solver.prior_cost.parameters(), "lr": self.lr/2},
        # ],)