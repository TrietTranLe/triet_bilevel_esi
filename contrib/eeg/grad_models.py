import einops
import torch
from torch import nn


class RearrangedConvLstmGradModel(nn.Module):
    """
    Wrapper around the base grad model that allows for reshaping of the input batch
    Used to convert the lorenz timeseries into an "image" for reuse of conv2d layers
    """
    def __init__(self, dim_hidden, dropout=0.1, downsamp=None, rearrange_from='b c t', rearrange_to='b c t ()', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rearrange_bef = rearrange_from + ' -> ' + rearrange_to
        self.rearrange_aft = rearrange_to + ' -> ' + rearrange_from

        self.dim_hidden = dim_hidden
        self.dropout = torch.nn.Dropout(dropout)
        self._state = []
        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]


class RearrangedConvLstmGradModel_lr(RearrangedConvLstmGradModel):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None, lr_init=0.0, rearrange_from='b c t', rearrange_to='b c t ()', *args, **kwargs):
        super().__init__(dim_hidden, dropout, downsamp, rearrange_from, rearrange_to, *args, **kwargs)

        self.lr_init = lr_init
        self.encoder = torch.nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
        self.decoder = torch.nn.Conv2d(dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2)

        self.encoder_grad = torch.nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
        self.decoder_grad = torch.nn.Conv2d(dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2)

        self.gates_x = torch.nn.Conv2d(
            3*dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.gates_grad = torch.nn.Conv2d(
            2*dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        
        self._state_x = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

        self._state_grad = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x, grad_x, gradients):
        x = einops.rearrange(x, self.rearrange_bef)
        grad_x = einops.rearrange(grad_x, self.rearrange_bef)

        if self._grad_norm is None:
            self._grad_norm = (grad_x**2).mean().sqrt()
        grad_x =  grad_x / self._grad_norm

        x = self.dropout(x)
        x = self.down(x)
        x = self.encoder(x)
        grad_x = self.dropout(grad_x)
        grad_x = self.down(grad_x)
        grad_x = self.encoder_grad(grad_x)

        grad_x = self._forward_grad(grad_x)
        x = self._forward_x(x, grad_x)

        lr = self.up(self.decoder(x))
        out = self.up(self.decoder_grad(grad_x))

        out = lr*out
        return einops.rearrange(out, self.rearrange_aft), gradients

    def _forward_x(self, x, grad_x):
        hidden, cell = self._state_x
        gates = self.gates_x(torch.cat((x, grad_x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state_x = hidden, cell
        return hidden

    def _forward_grad(self, grad_x):
        hidden, cell = self._state_grad
        gates = self.gates_grad(torch.cat((grad_x, hidden), 1))
        
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state_grad = hidden, cell
        return hidden


class RearrangedConvLstmGradModel_a(RearrangedConvLstmGradModel):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None, lr_init=0.0, rearrange_from='b c t', rearrange_to='b c t ()', *args, **kwargs):
        super().__init__(dim_hidden, dropout, downsamp, rearrange_from, rearrange_to, *args, **kwargs)

        self.lr_init = lr_init
        self.encoder = torch.nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
        self.decoder = torch.nn.Conv2d(dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2)

        self.encoder_grad = torch.nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
        self.decoder_grad = torch.nn.Conv2d(dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2)

        self.gates_moment = torch.nn.Conv2d(
            2 * dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.gates_2nd_moment = torch.nn.Conv2d(
            2 * dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        
        self._grad_norm = None
        
        self._state_moment = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

        self._state_2nd_moment = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x, grad_x, gradients):
        x = einops.rearrange(x, self.rearrange_bef)
        grad_x = einops.rearrange(grad_x, self.rearrange_bef)

        if self._grad_norm is None:
            self._grad_norm = (grad_x**2).mean().sqrt()
        grad_x =  grad_x / self._grad_norm

        x = self.dropout(x)
        x = self.down(x)
        x = self.encoder(x)
        grad_x = self.dropout(grad_x)
        grad_x = self.down(grad_x)
        grad_x = self.encoder_grad(grad_x)

        moment = self._forward_moment(grad_x)
        second_moment = self._forward_2nd_moment(grad_x**2)
        out = moment/torch.sqrt(second_moment)

        return einops.rearrange(out, self.rearrange_aft), gradients

    def _forward_moment(self, grad_x):
        hidden, cell = self._state_moment
        gates = self.gates_moment(torch.cat((grad_x, hidden), 1))
        
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state_moment = hidden, cell
        return hidden

    def _forward_2nd_moment(self, grad_x):
        hidden, cell = self._state_2nd_moment
        gates = self.gates_2nd_moment(torch.cat((grad_x, hidden), 1))
        
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state_2nd_moment = hidden, cell
        return torch.sigmoid(hidden)


class RearrangedConvLstmGradModel_m(RearrangedConvLstmGradModel):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None, lr_init=0.0, rearrange_from='b c t', rearrange_to='b c t ()', *args, **kwargs):
        super().__init__(dim_hidden, dropout, downsamp, rearrange_from, rearrange_to, *args, **kwargs)

        self.lr_init = lr_init
        self.encoder_mag = nn.Sequential(
                                            nn.Flatten(),
                                            nn.Linear(in_features = 1, out_features = dim_hidden*64),
                                            nn.Unflatten(-1, (dim_hidden, 64, 1))
                                        )
        self.decoder_mag = nn.Sequential(
                                    nn.Flatten(),
                                    nn.Linear(in_features = dim_hidden*64, out_features = 1),
                                    nn.Unflatten(-1, (1, 1, 1))
                                )

        self.encoder_grad = torch.nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
        self.decoder_grad = torch.nn.Conv2d(dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2)

        self.gates_mag_grad_x = torch.nn.Conv2d(
            3 * dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.gates_moment = torch.nn.Conv2d(
            2*dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        
        # size_mag = [1, self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        
        self._state_mag_grad_x = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]
        
        self._state_moment = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x, grad_x, gradients):
        grad_x = einops.rearrange(grad_x, self.rearrange_bef)

        mag_grad_x = (grad_x**2).mean(dim=(-1, -2, -3), keepdim=True)
        grad_x = grad_x/(mag_grad_x.mean().sqrt())

        grad_x = self.dropout(grad_x)
        grad_x = self.down(grad_x)
        grad_x = self.encoder_grad(grad_x)

        mag_grad_x = self.encoder_mag(mag_grad_x.sqrt())

        grad_x = self._forward_moment(grad_x)
        mag_grad_x = self._forward_mag_grad(mag_grad_x, grad_x)

        mag_grad_x = self.decoder_mag(mag_grad_x).pow(2)
        out = self.up(self.decoder_grad(grad_x))
        
        return einops.rearrange(out*(mag_grad_x.mean().sqrt()), self.rearrange_aft), gradients

    def _forward_mag_grad(self, mag_grad_x, grad_mod):
        hidden, cell = self._state_mag_grad_x
        gates = self.gates_mag_grad_x(torch.cat((mag_grad_x, grad_mod, hidden), 1))
        
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state_mag_grad_x = hidden, cell
        return hidden

    def _forward_moment(self, grad_x):
        hidden, cell = self._state_moment
        gates = self.gates_moment(torch.cat((grad_x, hidden), 1))
        
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state_moment = hidden, cell


class RearrangedConvLstmGradModel_x(RearrangedConvLstmGradModel):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None, lr_init=0.0, rearrange_from='b c t', rearrange_to='b c t ()', *args, **kwargs):
        super().__init__(dim_hidden, dropout, downsamp, rearrange_from, rearrange_to, *args, **kwargs)

        self.lr_init = lr_init
        self.encoder = torch.nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
        self.decoder = torch.nn.Conv2d(dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2)

        self.encoder_grad = torch.nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
        self.decoder_grad = torch.nn.Conv2d(dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2)

        self.gates_lr = torch.nn.Conv2d(
            3 * dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.gates_moment = torch.nn.Conv2d(
            2 * dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.gates_2nd_moment = torch.nn.Conv2d(
            2*dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None

        self._state_lr = [
            self.down(torch.full(size, self.lr_init, device=inp.device)),
            self.down(torch.full(size, self.lr_init, device=inp.device)),
        ]
        
        self._state_moment = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

        self._state_2nd_moment = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x, grad_x, gradients):
        x = einops.rearrange(x, self.rearrange_bef)
        grad_x = einops.rearrange(grad_x, self.rearrange_bef)

        if self._grad_norm is None:
            self._grad_norm = (grad_x**2).mean().sqrt()
        grad_x =  grad_x / self._grad_norm

        x = self.dropout(x)
        x = self.down(x)
        x = self.encoder(x)
        grad_x = self.dropout(grad_x)
        grad_x = self.down(grad_x)
        grad_x = self.encoder_grad(grad_x)

        moment = self._forward_moment(grad_x)
        second_moment = self._forward_2nd_moment(grad_x**2)

        grad_x = moment/torch.sqrt(second_moment)
        lr = self._forward_lr(x, grad_x)
        
        lr = self.up(self.decoder(x))
        out = self.up(self.decoder_grad(grad_x))
        out = lr*out
        return einops.rearrange(out, self.rearrange_aft), gradients

    def _forward_lr(self, x, grad_x):
        hidden, cell = self._state_lr
        gates = self.gates_lr(torch.cat((x, grad_x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state_lr = hidden, cell
        return hidden

    def _forward_moment(self, grad_x):
        hidden, cell = self._state_moment
        gates = self.gates_moment(torch.cat((grad_x, hidden), 1))
        
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state_moment = hidden, cell
        return hidden
        
    def _forward_2nd_moment(self, grad_x):
        hidden, cell = self._state_2nd_moment
        gates = self.gates_2nd_moment(torch.cat((grad_x, hidden), 1))
        
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self.gates_2nd_moment = hidden, cell
        return torch.sigmoid(hidden)
        
