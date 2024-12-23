import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

config = Config()

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PSPPooling(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(size, size)), nn.Conv2d(in_channels, out_channels, kernel_size=1))
            for size in pool_sizes])
        self.bottleneck = nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        pooled = [F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        x = torch.cat([x] + pooled, dim=1)
        return self.bottleneck(x)


class ViTForSegmentation(nn.Module):
    def __init__(self, img_size=config.img_size, patch_size=config.patch_size, in_channels=config.in_channels, 
                 embed_dim=config.embed_dim, num_heads=config.num_heads, num_layers=config.num_layers, 
                 num_classes=config.num_classes, dropout=config.dropout):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, embed_dim))
        
        self.transformer = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        

        # PSPPooling
        self.upsample = nn.Sequential(PSPPooling(embed_dim, embed_dim),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(embed_dim // 2, num_classes, kernel_size=2, stride=2),
                    nn.Upsample(size=img_size, mode='bilinear', align_corners=False)
                )


    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        x = x + self.pos_embed
        
        x = self.transformer(x)
        x = self.norm(x)
        
        # Reshape and upsample
        H = self.img_size[0] // self.patch_size
        W = self.img_size[1] // self.patch_size
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.upsample(x)
        
        return x
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForSegmentation(config.img_size, config.patch_size, config.in_channels, 
                               config.embed_dim, config.num_heads, config.num_layers, 
                               config.num_classes, config.dropout).to(device)
    input_tensor = torch.randn(1, config.in_channels, config.img_size[0], config.img_size[1]).to(device)
    output = model(input_tensor)
    print(f"Output shape of ViT: {output.shape}")