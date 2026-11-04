"""1D Fourier Neural Operator that maps two past states to the next state.

Input channels: [u_{t-1}, u_t, x]  (x = grid coordinate). The network predicts an INCREMENT, so
u_{t+1} = u_t + f(u_{t-1}, u_t, x). No PDE identifier or coefficient is given to the model: the
two-frame history is the only information about the dynamics ("in-context" inference).
"""

import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.in_ch, self.out_ch, self.modes = in_ch, out_ch, modes
        scale = 1.0 / (in_ch * out_ch)
        # complex weights stored as real tensors (last dim = real/imag) for optimiser safety
        self.weight = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, 2))

    def forward(self, x):                                   # x: (B, C, N)
        B, _, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                    # (B, C, N//2+1)
        w = torch.view_as_complex(self.weight)              # (in, out, modes)
        out_ft = torch.zeros(B, self.out_ch, N // 2 + 1, dtype=x_ft.dtype, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :self.modes], w)
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, width=64, modes=16, layers=4, in_frames=2):
        super().__init__()
        self.lift = nn.Conv1d(in_frames + 1, width, 1)      # +1 for the grid coordinate
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(layers)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv1d(width, 128, 1), nn.GELU(), nn.Conv1d(128, 1, 1))

    def forward(self, x):                                   # x: (B, 2, N) = [u_{t-1}, u_t]
        B, _, N = x.shape
        grid = torch.linspace(0, 1, N + 1, device=x.device, dtype=x.dtype)[:-1]
        grid = grid.view(1, 1, N).expand(B, 1, N)
        h = self.lift(torch.cat([x, grid], dim=1))
        for spec, pw in zip(self.spectral, self.pointwise):
            h = nn.functional.gelu(spec(h) + pw(h))
        return x[:, 1] + self.proj(h).squeeze(1)            # residual: u_t + increment


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
