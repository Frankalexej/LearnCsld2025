import torch
import torch.nn as nn

class SimpleNonlinearFCAutoencoder(nn.Module): 
    """
    This model is just a two-layer autoencoder with an additional ReLU. It is expected to perform similarly to the theoretical linear AE. 
    """
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hid_features), 
            # nn.ReLU()
        )
        self.decoder = nn.Linear(hid_features, out_features)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
    
    def encode(self, x): 
        return self.encoder(x)

class SimpleNonlinearFCClassifier(nn.Module): 
    """
    This model is just a two-layer autoencoder with an additional ReLU. It is expected to perform similarly to the theoretical linear AE. 
    """
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hid_features), 
            nn.ReLU()
        )
        self.decoder = nn.Linear(hid_features, out_features)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
    
    def encode(self, x): 
        return self.encoder(x)


class SimpleResNet1D(nn.Module):
    def __init__(self, hid_features=16, out_features=4):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(16, hid_features)
        self.fccl = nn.Linear(hid_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return self.fccl(x)

    def encode(self, x): 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.squeeze(-1)
        return self.fc(x)

class SimpleResNet1DClass(nn.Module):
    def __init__(self, hid_features=16, out_features=4):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(16, hid_features)
        self.fccl = nn.Linear(hid_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return self.fccl(x)

    def encode(self, x): 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.squeeze(-1)
        return self.fc(x)


class SimpleResNet1DRecon(nn.Module):
    """
    Reconstruction model that keeps your encoder structure essentially unchanged.
    Input:  (B, 1, 51)
    Output: (B, 1, 51)

    Encoder path (unchanged):
      layer1: 51 -> 25
      layer2: 25 -> 12
      layer3: -> AdaptiveAvgPool1d(1) => length 1
      fc: 16 -> hid_features

    Decoder:
      hid_features -> (16, 12) via Linear
      12 -> 24 -> 51 via ConvTranspose1d
    """
    def __init__(self, hid_features=16):
        super().__init__()

        # -------------------
        # Encoder (same as yours, minus classifier head)
        # -------------------
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)   # keep as-is
        )
        self.fc = nn.Linear(16, hid_features)       # keep as-is

        # -------------------
        # Decoder
        # -------------------
        # We "re-inflate" time by mapping z -> (B, 16, 12).
        # Why 12? Because after two MaxPool1d(2) on length 51: 51 -> 25 -> 12.
        self.down_len = 12
        self.fc_dec = nn.Linear(hid_features, 16 * self.down_len)

        self.dec_conv = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        # 12 -> 24
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        # 24 -> 51 (critical: output_padding=1)
        self.deconv1 = nn.ConvTranspose1d(
            16, 1, kernel_size=3, stride=2, padding=0, output_padding=0
        )

    def encode(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.squeeze(-1)     # (B, 16)
        return self.fc(x)     # (B, hid_features)

    def decode(self, z):
        x = self.fc_dec(z)                               # (B, 16*12)
        x = x.view(z.size(0), 16, self.down_len)         # (B, 16, 12)
        x = self.dec_conv(x)                             # (B, 32, 12)
        x = self.deconv2(x)                              # (B, 16, 24)
        x = self.deconv1(x)                              # (B,  1, 51)
        return x

    def forward(self, x, return_latent: bool = False):
        z = self.encode(x)
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon
    
class SimpleResNet1DEncode(nn.Module):
    def __init__(self, hid_features=16):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(16, hid_features)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.squeeze(-1)
        return self.fc(x)
    


##################### Simplets Linear Model #####################
"""
Notice that there is no difference bewteen LinearFC Recon and Class, just for compatibility of codes. 
"""
class LinearFC(nn.Module): 
    def __init__(self):
        super().__init__()
    
    def forward(self, x): 
        pass

    def encode(self, x): 
        pass

    def set_freeze(self): 
        pass

    def set_unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

class LinearFCRecon(LinearFC):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.encoder = nn.Linear(in_features, hid_features)
        self.decoder = nn.Linear(hid_features, out_features)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        hid = self.encoder(x)
        out = self.decoder(hid)
        return out

    def encode(self, x):
        x = x.reshape(x.size(0), -1)
        return self.encoder(x)
    
    def set_freeze(self, freeze_encoder=False, freeze_decoder=False):
        for p in self.encoder.parameters():
            p.requires_grad = not freeze_encoder
        for p in self.decoder.parameters():
            p.requires_grad = not freeze_decoder

    def encoder_names(self): 
        return ("encoder.")
    
class LinearFCClass(LinearFC):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.encoder = nn.Linear(in_features, hid_features)
        self.predictor = nn.Linear(hid_features, out_features)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        hid = self.encoder(x)
        out = self.predictor(hid)
        return out

    def encode(self, x):
        x = x.reshape(x.size(0), -1)
        return self.encoder(x)
    
    def set_freeze(self, freeze_encoder=False, freeze_decoder=False):
        for p in self.encoder.parameters():
            p.requires_grad = not freeze_encoder
        for p in self.decoder.parameters():
            p.requires_grad = not freeze_decoder

    def encoder_names(self): 
        return ("encoder.")


class LinearFCEncode(LinearFC):
    def __init__(self, in_features, hid_features, out_features):
        super().__init__()
        self.encoder = nn.Linear(in_features, hid_features)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        hid = self.encoder(x)
        return hid