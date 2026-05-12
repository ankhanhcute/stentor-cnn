"""
model.py
-------
A small CNN for binary classification of Stentor coeruleus contractions
events.

Input: (batch, 2, 150, 150) float32 in [0,1] - normalize
        channel 0 = pre-stimulus frame, channel 1 = post-stimulus frame
Output: (batch, 1) float32 LOGITS(NOT sigmoid-activated)
        Apply torch.sigmoid() externally to get probabilities, or use 
        BCEWithLogitsLoss for training (numerically stable)

Architecture: 4 conv block (with BatchNorm + ReLU + MaxPool) followed by 
global average pooling and a single linear layer. 

Run this file directly in the terminal:
     python stentor_cnn/model.py

"""
from __future__ import annotations 
import torch 
import torch.nn as nn
import torch.nn.functional as F 

class StentorCNN(nn.Module):
    def __init__(self, in_channels: int = 2, dropout: float = 0.3):
        super().__init__()

        self.block1 = self._conv_block(in_channels, 32)
        self.block2 = self._conv_block(32, 64)
        self.block3 = self._conv_block(64, 128)
        self.block4 = self._conv_block(128, 128)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, 1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.head(x)
        return x
def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

#-----Test if the model alive-----
def _self_test() -> None:
    print("===Model self test before training===")
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"device chose: {device}")


    model = StentorCNN().to(device)
    n = count_params(model)
    print(f"parameters: {n:,}  ({n/1e3:.1f}k)")
    #print how many trainable weights your model have 

    #Shape check 
    x = torch.randn(4, 2, 150, 150, device=device)
    y = model(x)
    print(f"input shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)} (expected (4, 1))")
    assert y.shape == (4, 1) , f"unexpected output shape {y.shape}"
    
    #Overfit a tiny batch test
    model.train()
    x_small = torch.randn(8, 2, 150, 150, device=device)
    y_small = torch.randint(0, 2, (8, 1), device=device).float()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = None
    for step in range(100):
        opt.zero_grad()
        logits = model(x_small)
        loss = loss_fn(logits, y_small)
        loss.backward()
        opt.step()
        if step==0:
            initial_loss = loss.item()
        if step % 20 == 0 or step == 99:
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y_small).float().mean().item()
            print(f"  step {step:3d}: loss={loss.item():.4f}  acc={acc:.2f}")
    final_loss = loss.item()
    assert final_loss < 0.1, (
        f"model could not overfit 8 samples (initial={initial_loss:.3f}, "
        f"final={final_loss:.3f}); something is wrong with the wiring."
    )
    print(f"\n[OK] model overfit 8 samples (loss {initial_loss:.3f} -> {final_loss:.4f})")

if __name__ == "__main__":
    _self_test()

