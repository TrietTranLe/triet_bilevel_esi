from torch import nn


class SiameseMetric(nn.Module):
    def __init__(self, dim_in_x = 994, dim_in_y = 90, dim_in_t = 64, dim_hidden=64, kernel_size = 3, bias=True) -> None:
        super().__init__()

        # if pretrained_model_path is not None: # load a pretrained model if a path is provided
        #     self.load_state_dict(torch.load(pretrained_model_path))

        # if fixed : # freeze the model if fixed is True
        #     for param in self.parameters():
        #         param.requires_grad = False

        self.encoder_x = nn.Sequential(
            nn.Conv1d(in_channels=dim_in_x, out_channels=dim_hidden, kernel_size=kernel_size, bias=bias, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=dim_hidden*dim_in_t, out_features=dim_hidden)
        )
        
        self.encoder_y = nn.Sequential(
            nn.Conv1d(in_channels=dim_in_y, out_channels=dim_hidden, kernel_size=kernel_size, bias=bias, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=dim_hidden*dim_in_t, out_features=dim_hidden)
        )
        
        self.metric = nn.Bilinear(in1_features=dim_hidden, in2_features=dim_hidden, out_features=1)

    def forward_x(self, x1, x2):
        x1 = self._remove_mag(x1)
        x2 = self._remove_mag(x2)
        
        x1 = self.encoder_x(x1)
        x2 = self.encoder_x(x2)
        
        # x = x1 - x2
        return self._norm_cosine(self.metric(x1, x2)).mean()

    def forward_y(self, y1, y2):
        y1 = self._remove_mag(y1)
        y2 = self._remove_mag(y2)
        
        y1 = self.encoder_y(y1)
        y2 = self.encoder_y(y2)
        
        # y = y1 - y2
        return self._norm_cosine(self.metric(y1, y2)).mean()
    
    def _remove_mag(self, x):
        return x/((x**2).sum((-1, -2), keepdim=True).sqrt())

    def _norm_cosine(self, x):
        x = x - x.min()
        return 2*x/x.max()
    
    
    