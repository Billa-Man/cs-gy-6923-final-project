import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

config = Config()

class FCN(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(FCN, self).__init__()

        # Encoder blocks
        self.encoder_1 = self.encoder_block_2(in_channels, 8)  
        self.encoder_2 = self.encoder_block_2(8, 16)          
        self.encoder_3 = self.encoder_block_3(16, 32)         
        self.encoder_4 = self.encoder_block_3(32, 64)        

        # Mid block
        self.mid = self.mid_block(64, 128)                    

        # Upsampling layers
        self.conv_t_32s = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv_t_16s = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv_t_8s  = nn.ConvTranspose2d(96, 32, 4, 4)

        # 1x1 convolutions for skip connections
        self.x3_conv_1x1 = nn.Conv2d(32, 64, 1)
        self.x2_conv_1x1 = nn.Conv2d(16, 32, 1)

        # Output layer
        self.output = nn.Conv2d(32, out_channels, 1)

    def encoder_block_2(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def encoder_block_3(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def mid_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, 128, 3, 1, 'same'),
            nn.ReLU(inplace=True),
        )

    def upsample_to_input_size(self, input_tensor, output_tensor):
        return F.interpolate(output_tensor, size=(input_tensor.size(2), input_tensor.size(3)), mode='bilinear', align_corners=False)

    def forward(self, x, output_stride=None):
        # Encoder
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)
        x4 = self.encoder_4(x3)
    
        # 1x1 convolutions
        x3_1x1 = self.x3_conv_1x1(x3)
        x2_1x1 = self.x2_conv_1x1(x2)
    
        # Mid-block
        x_mid = self.mid(x4)
    
        # FCN-32s path
        x4 = self.conv_t_32s(x_mid)
        if output_stride == '32s':
            return self.upsample_to_input_size(x, x4)
    
        # Ensuring that x3_1x1 has the same spatial dimensions as x4
        x3_1x1 = F.interpolate(x3_1x1, size=(x4.size(2), x4.size(3)), mode='bilinear', align_corners=False)
        
        # FCN-16s path
        x5 = torch.cat([x4, x3_1x1], dim=1)
        x5 = self.conv_t_16s(x5)
        if output_stride == '16s':
            return self.upsample_to_input_size(x, x5)
    
        # Ensuring that x2_1x1 has the same spatial dimensions as x5
        x2_1x1 = F.interpolate(x2_1x1, size=(x5.size(2), x5.size(3)), mode='bilinear', align_corners=False)
    
        # FCN-8s path
        x6 = torch.cat([x5, x2_1x1], dim=1)
        x6 = self.conv_t_8s(x6)
    
        # Final output layer
        x_out = self.output(x6)
    
        # Upsample the final output to match the input size
        return self.upsample_to_input_size(x, x_out)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCN(in_channels=1, out_channels=1).to(device)
    input_tensor = torch.randn(1, 1, 420, 580).to(device)
    output = model(input_tensor)
    print("Output shape of FCN:", output.shape)