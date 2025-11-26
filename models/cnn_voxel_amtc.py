# models/cnn_voxel_amtc.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class get_model(nn.Module):
    """
    CNN por-voxel basada en el paper:
    - Entrada esperada en forward: x con forma [B, C, N] (tu loop ya hace transpose(2,1))
    - C debe contener al menos 'input_dim' features (por defecto 5: f1..f5).
    - Salida: seg_pred con forma [B, N, num_classes], trans_feat=None (compatibilidad).
    """
    def __init__(self, num_classes=2, num_feats=5, dropout=0.5, input_dim=5, img_hw=8):
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim                 # nº de features por voxel (f1..f5)
        self.img_hw = img_hw                       # 8 -> 64 = 8x8 tras FC
        fc_out = img_hw * img_hw                   # 64

        # Mapeo tabular -> "imagen"
        self.fc1 = nn.Linear(self.input_dim, fc_out)
        self.bn_fc1 = nn.BatchNorm1d(fc_out)

        # CNN 2D: 3 bloques Conv(3x3)+ReLU+MaxPool(2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.pool  = nn.MaxPool2d(2, 2)

        # Clasificación
        # Con img_hw=8 y 3 pools (8->4->2->1) queda 1x1, así que 256*1*1
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, C, N]
        B, C, N = x.shape

        # Tomamos sólo las 'input_dim' features por voxel (primeras columnas)
        # Si C == input_dim, no corta; si C > input_dim, ignora extras.
        x = x[:, :self.input_dim, :]                  # [B, input_dim, N]
        x = x.transpose(2, 1).contiguous()            # [B, N, input_dim]
        x = x.view(B * N, self.input_dim)             # [B*N, input_dim]

        # FC -> "imagen" 8x8
        x = self.fc1(x)                               # [B*N, 64]
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = x.view(-1, 1, self.img_hw, self.img_hw)   # [B*N, 1, 8, 8]

        # CNN
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B*N, 32, 4, 4]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B*N, 128, 2, 2]
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [B*N, 256, 1, 1]

        # Flatten + FC final
        x = x.view(x.size(0), -1)                     # [B*N, 256]
        x = self.dropout(x)
        x = self.fc2(x)                               # [B*N, num_classes]

        # Volver a [B, N, num_classes]
        x = x.view(B, N, self.num_classes)
        return x, None


class get_loss(nn.Module):
    """
    Cross-entropy con ignore_index para soportar padding (-1) del collate_fn.
    """
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, trans_feat=None, weight=None):
        # pred: [B, N, num_classes] -> aplastamos a [B*N, num_classes]
        # target: [B, N] -> [B*N]
        loss = F.cross_entropy(
            pred.view(-1, pred.size(-1)),
            target.view(-1),
            weight=weight,
            ignore_index=self.ignore_index
        )
        return loss
