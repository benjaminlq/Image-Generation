"""Building Blocks for VAE and GANs
"""
import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """Basic FC Block with BatchNorm and Activation"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        neg_slope: float = 0.1,
    ):
        """Basic FC Block with BatchNorm and Activation

        Args:
            in_features (int): No of input features
            out_features (int): No of output features
            neg_slope (float): Negative Slope for LeakyReLU unit. Default to 0.1.

        Returns:
            nn.Module: Linear Block
        """
        super(MLPBlock, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(neg_slope),
        )

    def forward(self, inputs):
        """Forward Propagation

        Args:
            inputs (torch.tensor): batch of input images

        Returns:
            torch.tensor: Output Tensor
        """
        return self.linear(inputs)


class ConvBlock(nn.Module):
    """Basic Conv Block with BatchNorm and Activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """Basic Conv Block with BatchNorm and Activation

        Args:
            in_channels (int): No of input channels
            out_channels (int): No of output channels
            kernel_size (int, optional): Kernel Size of Conv Filter. Defaults to 3.
            padding (int, optional): Padding Size. Defaults to 1.

        Returns:
            nn.Module: Conv Block
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        """Forward Propagation

        Args:
            inputs (torch.tensor): batch of input images

        Returns:
            torch.tensor: Output Tensor
        """
        return self.conv(inputs)


class DownSample(nn.Module):
    """DownSample Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """DownSampling Block.

        Args:
            in_channels (int): No of input channels
            out_channels (int): No of output channels
            kernel_size (int, optional): Kernel Size of Conv Filter. Defaults to 3.
            padding (int, optional): Padding Size. Defaults to 1.

        Returns:
            nn.Module: DownSampling Block
        """
        super(DownSample, self).__init__()
        self.upsample = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, padding),
            ConvBlock(out_channels, out_channels, kernel_size, padding),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, inputs):
        """Forward Propagation

        Args:
            inputs (torch.tensor): batch of input images

        Returns:
            torch.tensor: Output Tensor
        """
        return self.upsample(inputs)


class UpSample(nn.Module):
    """UpSampling Block"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """Up Sampling Block

        Args:
            in_channels (int): No of input channels
            out_channels (int): No of output channels
            kernel_size (int, optional): Kernel Size of Conv Filter. Defaults to 3.
            padding (int, optional): Padding Size. Defaults to 1.

        Returns:
            nn.Module: UpSampling Block
        """
        super(UpSample, self).__init__()
        self.downsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            ConvBlock(out_channels, out_channels, kernel_size, padding),
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        """Forward Propagation

        Args:
            inputs (torch.tensor): batch of input images

        Returns:
            torch.tensor: Output Tensor
        """
        return self.downsample(inputs)


class ConvTranspose2DBlock(nn.Module):
    """ConvTranspose2DBlock Module"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
    ):
        """ConvTranspose2DBlock Module for image generation. Use RELU activation with BatchNorm2D as per DCGAN paper.

        Args:
            in_channels (int): No of input channels
            out_channels (int): No of output channels
            kernel_size (int, optional): Kernel Size of Conv Filter. Defaults to 4.
            stride (int, optional): Conv Moving Stride. Defaults to 2.
            padding (int, optional): Padding Size. Defaults to 1.
            bias (bool, optional): If True, use bias for convolutional filter. Defaults to False.
        """
        super(ConvTranspose2DBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        """Forward Propagation

        Args:
            inputs (torch.tensor): Input Tensors

        Returns:
            torch.tensor: Output Tensors
        """
        return self.conv(inputs)


class Conv2DBlock(nn.Module):
    """Conv2DBlock Module"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
    ):
        """Conv2DBlock Module for image discrimination. Use LeakyReLU activation and BatchNorm2D as per DCGAN paper.

        Args:
            in_channels (int): No of input channels
            out_channels (int): No of output channels
            kernel_size (int, optional): Kernel Size of Conv Filter. Defaults to 4.
            stride (int, optional): Conv Moving Stride. Defaults to 2.
            padding (int, optional): Padding Size. Defaults to 1.
            bias (bool, optional): If True, use bias for convolutional filter. Defaults to False.
        """
        super(Conv2DBlock, self).__init__()
        assert out_channels % 2 == 0, "Number of output channels must be divisible by 2"
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size, stride, padding, bias=bias
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                out_channels // 2, out_channels, kernel_size, stride, padding, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        """Forward Propagation

        Args:
            inputs (torch.tensor): Input Tensors

        Returns:
            torch.tensor: Output Tensors
        """
        return self.conv(inputs)
