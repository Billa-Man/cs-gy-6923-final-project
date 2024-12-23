import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    def __init__(self):
        super().__init__()

        # ViT
        self.img_size = (420, 580)
        self.patch_size = 16
        self.in_channels = 1
        self.embed_dim = 128
        self.num_heads = 4
        self.num_layers = 2
        self.num_classes = 1
        self.dropout = 0.3

        # Global Hyperparameters
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.num_epochs = 300

        # Data Preparation
        self.transforms = A.Compose([A.HorizontalFlip(p=0.5),
                                     A.VerticalFlip(p=0.5),
                                     A.ElasticTransform(p=0.5),
                                     A.GridDistortion(p=0.5),
                                     ToTensorV2()])



