from model.KAN import KANLinear

import torch
class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.kans1 = KANLinear(in_channels, hidden_channels)
        self.kans2 = KANLinear(hidden_channels, out_channels)

    def forward(self, x):
        x1=self.kans1(x)
        x2=self.kans2(x1)
        return x1,x2

class Decoder_a(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Decoder_a, self).__init__()
        self.kans1 = KANLinear(in_channels, hidden_channels)
        self.kans2 = KANLinear(hidden_channels, out_channels)

    def forward(self, x):
        x1 = self.kans1(x)
        x2 = self.kans2(x1)
        return x1, x2

class Decoder_b(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Decoder_b, self).__init__()
        self.kans1 = KANLinear(in_channels, hidden_channels)
        self.kans2 = KANLinear(hidden_channels, out_channels)

    def forward(self, x):
        x1 = self.kans1(x)
        x2 = self.kans2(x1)
        return x1, x2

