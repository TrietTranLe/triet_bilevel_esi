# from Triet 
import pytorch_lightning as pl
import torch
from torch import nn


class autoencoder(nn.Module):
    def __init__(self, model_size_dict: dict, bias = False, hidden_activation_function = "ReLU", group_conv = 2, skip_connect = True, encoder_template_name = "encode", decoder_template_name = "decode", mode = "train", **kwargs):
        super().__init__()
        self.model_size_dict = model_size_dict
        self.bias = bias
        self.hidden_activation_function = hidden_activation_function
        self.group_conv = [group_conv] if str(type(group_conv)) == str(int) else group_conv
        self.skip_connect = skip_connect
        self.encoder_template_name = encoder_template_name
        self.decoder_template_name = decoder_template_name
        self.mode = mode
        self.kwargs = kwargs
        self._create_layers()

    def _create_layers(self):
        self.layers = []
        for layer_name, [in_channels, out_channels, kernel_size] in self.model_size_dict.items():
            if self.encoder_template_name in layer_name.lower() or self.decoder_template_name in layer_name.lower():
                sublayers = ["Conv", self.hidden_activation_function]
            else:
                sublayers = ["Group"]
                if len(self.group_conv) > 1:
                    sublayer_config = list(zip([in_channels]*len(self.group_conv), [out_channels]*len(self.group_conv), self.group_conv))
                else:
                    sublayer_config = list(zip([in_channels]*self.group_conv[0], [out_channels]*self.group_conv[0], [kernel_size]*self.group_conv[0]))
                for i in range(len(sublayer_config)):
                    sub_in_channels, sub_out_channels, sub_kernel_size = sublayer_config[i]
                    sublayers += [f"Conv{sub_kernel_size}_{i}"]
                sublayers += ["Ungroup"]

            sublayer_ith = 0
            for sublayer in sublayers:
                sublayer_name = layer_name + "_" + sublayer
                self.layers.append(sublayer_name)
                if sublayer not in ["Group", "Ungroup", "Subgroup", "Unsubgroup"]:
                    if sublayer not in self.hidden_activation_function:
                        if self.encoder_template_name in layer_name.lower() or self.decoder_template_name in layer_name.lower():
                            sub_in_channels = in_channels
                            sub_out_channels = out_channels
                            sub_kernel_size = kernel_size
                        else:
                            sub_in_channels, sub_out_channels, sub_kernel_size = sublayer_config[sublayer_ith]
                            sublayer_ith += 1
                            if sublayer_ith == len(sublayer_config):
                                sublayer_ith = 0
                        setattr(self, sublayer_name, nn.Conv1d(
                            in_channels   = sub_in_channels,
                            out_channels  = sub_out_channels,
                            kernel_size   = sub_kernel_size, 
                            padding       = 'same',
                            bias          = self.bias,
                            **self.kwargs
                        ))
                    else:
                        setattr(self, sublayer_name, getattr(nn, sublayer)())

    def _group_conv(self, layer_name, x):
        if "Group" in layer_name:
            if self.skip_connect:
                self.group_output = x
            else:
                self.group_output = torch.zeros_like(x)
            return x
        elif "Ungroup" in layer_name:
            return self.group_output
        else:
            res = getattr(self, layer_name)(x)
            self.group_output = torch.add(self.group_output, res)
            return x

    def forward(self, x):
        for layer_name in self.layers:
            if self.encoder_template_name in layer_name.lower() or self.decoder_template_name in layer_name.lower():
                x = getattr(self, layer_name)(x)
            else:
                x = self._group_conv(layer_name, x)
        return x


## Lightning Module
class autoencoder_pl(pl.LightningModule ): 
    def __init__(self, model_size_dict, bias=False, hidden_activation_function="ReLU",
                    group_conv=2, skip_connect=True,
                    encoder_template_name="encode", decoder_template_name="decode", 
                    optimizer = torch.optim.Adam, lr = 0.001, 
                    criterion = torch.nn.MSELoss(), mode="train", **kwargs) -> None:
        super().__init__()
        self.mode = mode
        
        # Load state dict (encode/eval mode)
        if mode.lower() in ["encode", "eval"]:
            if "state_dict" in kwargs:
                state_dict = kwargs["state_dict"]
                kwargs.pop("state_dict")
            else:
                raise Exception("state_dict not found.")

        # Remove decoder layers (encode mode)
        if mode.lower() == "encode":
            removal_layers_list = []
            for layer_name in model_size_dict:
                if decoder_template_name in layer_name.lower():
                    removal_layers_list.append(layer_name)
            for layer_name in removal_layers_list:
                for weight_name in state_dict:
                    if layer_name in weight_name:
                        state_dict.pop(weight_name)
                model_size_dict.pop(layer_name)

        self.model = autoencoder(model_size_dict, bias, hidden_activation_function, group_conv, skip_connect,
                                    encoder_template_name, decoder_template_name, mode, **kwargs)
        if mode.lower() in ["encode", "eval"]:
            self.load_state_dict(state_dict)
            self.model.eval()

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.kwargs = kwargs

    def forward(self, x):
        if self.mode.lower() == 'train':
            return self.model(x)
        
        # Don't need to add a new dimension or use torch.no_grad() outside
        elif self.mode.lower() in ["encode", "eval"]:
            x_dim = 3
            if self.mode.lower() == 'eval' and len(list(x.size())) == 2:
                x = x.unsqueeze(0)
                x_dim = 2
            with torch.no_grad():
                result = self.model(x)
                if self.mode.lower() == 'eval' and x_dim == 2:
                    result = result.squeeze(0)
                return result
        else:
            raise Exception(f"Mode {self.mode} not found.")
    
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        eeg, src = batch
        eeg, src = eeg.float(), src.float()
        src_hat = self.forward(src)

        # compute loss
        loss_train = self.criterion(src_hat, src)
        self.log("train_loss", loss_train, prog_bar=True, on_step=False, on_epoch=True)
        # loss = loss_func()
        # self.log_dict({"train_loss": loss_train, "train_mse": loss.mse(src_hat, src)}, prog_bar=True, on_step=False, on_epoch=True)
        return loss_train
    
    def validation_step(self, batch, batch_idx):
        eeg, src = batch
        eeg, src = eeg.float(), src.float()
        src_hat = self.forward(src)

        # compute loss
        loss_val = self.criterion(src_hat, src)
        self.log("val_loss", loss_val, prog_bar=True, on_step=False, on_epoch=True)
        # loss = loss_func()
        # self.log_dict({"val_loss": loss_val, "val_mse": loss.mse(src_hat, src)}, prog_bar=True, on_step=False, on_epoch=True)
        return loss_val


class autoencoder_lstm(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, cnn_kernel_size=1, lstm_num_layers=1, lstm_dropout=0, batch_first=True, bidirectional=True, bias=False):
        super().__init__()
        D = 2 if bidirectional else 1
        
        self.s_encode = nn.Conv1d(in_channels = in_size,
                                out_channels  = hidden_size,
                                kernel_size   = cnn_kernel_size, 
                                padding       = 'same',
                                bias          = bias)
        self.t_code = nn.LSTM(input_size = hidden_size,
                           hidden_size   = hidden_size,
                           num_layers    = lstm_num_layers,
                           bias          = bias,
                           dropout       = lstm_dropout,
                           batch_first   = batch_first,
                           bidirectional = bidirectional)
        self.s_decode = nn.Conv1d(in_channels = D*hidden_size,
                                out_channels  = out_size,
                                kernel_size   = cnn_kernel_size,
                                padding       = 'same',
                                bias          = bias)
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        x_dim = 3
        if len(list(x.size())) == 2:
            x = x.unsqueeze(0)
            x_dim = 2

        x = self.s_encode(x)
        x = self.ReLU(x)
        x = x.transpose(-1, -2)
        x, (h, c) = self.t_code(x)
        x = self.ReLU(x)
        x = x.transpose(-1, -2)
        x = self.s_decode(x)
        x = self.ReLU(x)
        
        if x_dim == 2:
            return x.squeeze(0)
        else:
            return x
