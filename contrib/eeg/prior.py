from collections import OrderedDict
from contrib.eeg.cost_funcs import *
import pytorch_lightning as pl
import torch
from torch import nn


def Cosine(x, y):
    return (1 - F.cosine_similarity(einops.rearrange(x, 'b c h -> b (c h)'), einops.rearrange(y, 'b c h -> b (c h)')))


class Prior(nn.Module):
    """
    Generic class for prior neural network.
    
    Attributes:
        cost_fn: A function to compute the cost or distance.
        pretrained_model_path: Path to the pretrained model, if any.
        fixed: Boolean indicating whether the model's weights are fixed or trainable.
    """
    def __init__(self, cost_fn=CosineReshape(), pretrained_model_path: str = None, fixed: bool = True, attention_hidden_size = 128, attention_kernel_size = 3) -> None:
        super().__init__()
        self.cost_fn = cost_fn

        if pretrained_model_path is not None: # load a pretrained model if a path is provided
            self.load_state_dict(torch.load(pretrained_model_path))

        if fixed : # freeze the model if fixed is True
            for param in self.parameters():
                param.requires_grad = False
                
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

    def forward_ae(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        return self.decoder(self.encoder(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the distance between the state and the autoencoder output."""
        y = self.forward_ae(x)
        # return self._attention(x, y, Cosine(x, y))
        return self.cost_fn(x, y)

    def forward_enc(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input x"""
        return self.encoder(x)

    # def _attention(self, x, y, v):
    #     x = self.fc(F.relu(torch.flatten(self.conv(x), 1, -1))).unsqueeze(-1)
    #     y = self.fc(F.relu(torch.flatten(self.conv(y), 1, -1))).unsqueeze(-2)
    #     v = self.fc_v(v.unsqueeze(-1)).unsqueeze(-1)
    #     return self.fc_end((F.softmax((x @ y)/(self.attention_hidden_size ** 0.5), dim=-1) @ v).squeeze(-1)).abs().mean()


class ConvAEPrior(Prior):
    """
    Convolutional autoencoder prior for source data.
    dim_in: Input dimension (number of input features).
    dim_hidden: Dimension of the hidden layer(s).
    dim_out: Output dimension (number of output features, typically equal to dim_in for reconstruction).
    bias: Whether to use bias terms in the convolutional layers.
    kernel_size: Size of the convolutional kernels (can be a single integer or a list of integers for different layers).
    """
    def __init__(self, dim_in=1284, dim_hidden=128, dim_out=1284, bias=False, kernel_size=3, cost_fn=CosineReshape(), pretrained_model_path=None, fixed=False) -> None:
        super().__init__(cost_fn, pretrained_model_path, fixed)
        if type(kernel_size) == int:
            kernel_size=[kernel_size]*3

        self.encoder = nn.Sequential(OrderedDict([
            ('conv_in', nn.Conv1d(in_channels=dim_in, out_channels=dim_hidden, kernel_size=kernel_size[0], bias=bias, padding="same")),
            ('relu_in', nn.ReLU()),
            ('conv_hidden', nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=kernel_size[1], bias=bias, padding="same")),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('relu_hidden', nn.ReLU()),
            ("conv_out", nn.Conv1d(in_channels=dim_hidden, out_channels=dim_out, kernel_size=kernel_size[2], bias=bias, padding="same"))
        ]))


class LstmAEPrior(Prior):
    """
    LSTM autoencoder prior for source data.
    dim_in: Input dimension (number of input features).
    dim_hidden: Dimension of the hidden layer(s).
    dim_out: Output dimension (number of output features, typically equal to dim_in for reconstruction).
    bias: Whether to use bias terms in the convolutional layers.
    kernel_size: Size of the convolutional kernels (can be a single integer or a list of integers for different layers).
    """
    def __init__(self, dim_in=1284, dim_hidden=128, dim_out=1284, kernel_size=1, lstm_num_layers=1, lstm_dropout=0, batch_first=True, bidirectional=True, bias=False, cost_fn=CosineReshape(), pretrained_model_path=None, fixed=False) -> None:
        super().__init__(cost_fn, pretrained_model_path, fixed)
        if type(kernel_size) == int:
            kernel_size=[kernel_size]*2

        self.conv_encoder = nn.Sequential(OrderedDict([
            ('conv_in', nn.Conv1d(in_channels=dim_in, out_channels=dim_hidden, kernel_size=kernel_size[0], bias=bias, padding="same")),
            ('relu_in', nn.ReLU()),
        ]))

        D = 2 if bidirectional else 1
        self.lstm_hidden = nn.LSTM(input_size = dim_hidden,
                           hidden_size   = dim_hidden,
                           num_layers    = lstm_num_layers,
                           bias          = bias,
                           dropout       = lstm_dropout,
                           batch_first   = batch_first,
                           bidirectional = bidirectional)

        self.decoder = nn.Sequential(OrderedDict([
            ('relu_hidden', nn.ReLU()),
            ("conv_out", nn.Conv1d(in_channels=D*dim_hidden, out_channels=dim_out, kernel_size=kernel_size[1], bias=bias, padding="same"))
        ]))

    def encoder(self, x):
        x = self.conv_encoder(x)
        x = x.transpose(-1, -2)
        x, (h, c) = self.lstm_hidden(x)
        return x.transpose(-1, -2)
        

class PriorPl(pl.LightningModule):
    def __init__(self, model, criterion=CosineReshape(), optimizer=torch.optim.Adam, lr=1e-3, **kwargs) -> None:
        """
        model : nn Module initialized with the prior architecture
        criterion : loss function
        optimizer : optimizer
        lr : learning rate
        """
        super().__init__(**kwargs)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer(self.model.parameters(), lr)

    def forward(self, x):
        return self.model.forward_ae(x)

    def base_step(self, batch, batch_idx):
        x_in, x_tgt = batch
        out = self.model.forward_ae(x_in)
        loss = self.criterion(out, x_tgt)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.base_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx): 
        loss = self.base_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer