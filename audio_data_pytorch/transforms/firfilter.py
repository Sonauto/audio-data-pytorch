from torch import Tensor, nn, permute
from auraloss.perceptual import FIRFilter as Filter

class FIRFilter(nn.Module):
    """Scales waveform (change volume)"""

    def __init__(
        self
    ):
        super().__init__()
        self.fir_filter = Filter(filter_type="aw", coef=0.85)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(0)
        left, right = self.fir_filter(x[:, 0, :], x[:, 1, :])
        x[0][0] = left
        x[0][1] = right
        return x[0]
