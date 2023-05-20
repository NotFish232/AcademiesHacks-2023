import torch as T
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self


class Generator(nn.Module):
    def __init__(self: Self, latent_dim: int) -> None:
        super().__init__()

        self.fully_connected_layers = nn.Sequential(
            [
                nn.Linear(latent_dim, 15 * 15 * 128),  # (batch_size, 15 * 15 * 128)
                nn.Dropout(),
                nn.LeakyReLU(),
                nn.Linear(15 * 15 * 128, 15 * 15 * 256),  # (batch_size, 15 * 15 * 256)
                nn.Dropout(),
                nn.LeakyReLU(),
            ]
        )
        # (batch_size, 256, 15, 15)
        self.inverse_convolutional_layers = nn.Sequential(
            [
                nn.ConvTranspose2d(256, 128, 9, stride=4),  # (batch_size, 128, 65, 65)
                nn.Dropout2d(),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 32, 8, stride=4),  # (batch_size, 32, 264, 264)
                nn.Dropout2d(),
                nn.LeakyReLU(),
            ]
        )
        self.convolutional_layers = nn.Sequential(
            [
                nn.Conv2d(32, 3, kernel_size=9),  # (batch_size, 3, 25, 256)
                nn.Dropout2d(),
                nn.LeakyReLU(),
            ]
        )

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        x = self.fully_connected_layers(x)
        x = self.inverse_convolutional_layers(x)
        x = self.convolutional_layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(self: Self) -> None:
        super().__init__()

    def forward(self: Self, x: T.Tensor) -> T.Tensor:
        pass

def main() -> None:
    pass 

if __name__ == "__main__":
    main()