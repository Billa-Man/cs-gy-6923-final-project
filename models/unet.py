import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

config = Config()


def crop_tensor(enc_tensor, target_tensor):

    _, _, h, w = enc_tensor.size()
    _, _, target_h, target_w = target_tensor.size()

    crop_h = (h - target_h) // 2
    crop_w = (w - target_w) // 2

    return enc_tensor[:, :, crop_h:crop_h + target_h, crop_w:crop_w + target_w]


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=8):
        super(UNet, self).__init__()

        # Encoder: Downsampling path
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = self._conv_block(base_channels * 8, base_channels * 16)

        # Decoder: Upsampling path
        self.up4 = self._up_conv(base_channels * 16, base_channels * 8)
        self.dec4 = self._conv_block(base_channels * 16, base_channels * 8)

        self.up3 = self._up_conv(base_channels * 8, base_channels * 4)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)

        self.up2 = self._up_conv(base_channels * 4, base_channels * 2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)

        self.up1 = self._up_conv(base_channels * 2, base_channels)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)

        # Final output layer
        self.upsample = nn.Upsample(size=config.img_size, mode='bilinear', align_corners=False)
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(self._downsample(enc1))
        enc3 = self.enc3(self._downsample(enc2))
        enc4 = self.enc4(self._downsample(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self._downsample(enc4))

        # Decoder path with skip connections
        up4 = self.up4(bottleneck)
        dec4 = self.dec4(torch.cat([up4, crop_tensor(enc4, up4)], dim=1))

        up3 = self.up3(dec4)
        dec3 = self.dec3(torch.cat([up3, crop_tensor(enc3, up3)], dim=1))

        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, crop_tensor(enc2, up2)], dim=1))

        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, crop_tensor(enc1, up1)], dim=1))

        # Final output
        ups = self.upsample(dec1)
        out = self.final(ups)
        return out

    def _downsample(self, x):
        return nn.functional.max_pool2d(x, kernel_size=2, stride=2)
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1)
    input_tensor = torch.randn(1, 1, 420, 580)
    output = model(input_tensor)
    print(f"Output shape of UNet: {output.shape}")