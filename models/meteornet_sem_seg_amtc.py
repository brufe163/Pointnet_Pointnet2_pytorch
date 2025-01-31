import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionST, PointNetFeaturePropagation

class MeteorNet(nn.Module):
    def __init__(self, num_classes, num_feats, num_frames=4, dropout=0.5):
        super(MeteorNet, self).__init__()
        self.num_frames = num_frames

        # Definir los radios temporales y espaciales para cada capa
        radius_t = 1  # Ajusta según sea necesario para tus datos

        # Módulos Set Abstraction con agrupamiento espacio-temporal
        self.sa1 = PointNetSetAbstractionST(
            npoint=1024, radius_s=0.1, radius_t=radius_t, nsample=32,
            in_channel=num_feats + 4, mlp=[32, 32, 64], group_all=False
        )
        self.sa2 = PointNetSetAbstractionST(
            npoint=256, radius_s=0.2, radius_t=radius_t, nsample=32,
            in_channel=64 + 4, mlp=[64, 64, 128], group_all=False
        )
        self.sa3 = PointNetSetAbstractionST(
            npoint=64, radius_s=0.4, radius_t=radius_t, nsample=32,
            in_channel=128 + 4, mlp=[128, 128, 256], group_all=False
        )
        self.sa4 = PointNetSetAbstractionST(
            npoint=16, radius_s=0.8, radius_t=radius_t, nsample=32,
            in_channel=256 + 4, mlp=[256, 256, 512], group_all=False
        )

        # Módulos Feature Propagation
        self.fp4 = PointNetFeaturePropagation(in_channel=768, mlp=[256, 256])
        self.fp3 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=320, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + num_feats, mlp=[128, 128, 128])

        # Capas finales
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz_seq):
        # xyz_seq: [B, T, C, N]
        B, T, C, N = xyz_seq.shape

        # Separar coordenadas y características adicionales
        coords = xyz_seq[:, :, :3, :]  # [B, T, 3, N]
        features = xyz_seq[:, :, 3:, :]  # [B, T, num_feats, N]

        # Agregar coordenada temporal
        device = xyz_seq.device
        time_indices = torch.arange(T, device=device).float()  # [T]
        time_indices = time_indices.view(1, T, 1, 1).repeat(B, 1, 1, N)  # [B, T, 1, N]

        # Concatenar tiempo a las coordenadas
        coords = torch.cat([coords, time_indices], dim=2)  # [B, T, 4, N]

        # Combinar batch y frames
        coords = coords.view(B * T, 4, N)  # [B*T, 4, N]
        if features is not None and features.shape[2] > 0:
            features = features.view(B * T, -1, N)  # [B*T, num_feats, N]
        else:
            features = None

        # Aplicar los módulos Set Abstraction con agrupamiento espacio-temporal
        l1_xyz, l1_points = self.sa1(coords, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        # Aplicar los módulos Feature Propagation
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(coords, l1_xyz, None, l1_points)

        # Restaurar dimensiones originales
        l0_points = l0_points.view(B, T, l0_points.shape[1], N)

        # Agregación temporal (e.g., max-pooling a través de los frames)
        x = torch.max(l0_points, dim=1)[0]  # [B, C, N]

        # Capas finales
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.conv2(x)

        x = x.permute(0, 2, 1)  # [B, N, num_classes]
        return x

class get_loss(nn.Module):
    def __init__(self, ignore_index=None):
        super(get_loss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, weight):
        total_loss = F.cross_entropy(pred, target, weight=weight, ignore_index=self.ignore_index)
        return total_loss

if __name__ == '__main__':
    import torch
    model = MeteorNet(num_classes=2, num_feats=1, num_frames=4)
    xyz_seq = torch.rand(6, 4, 4, 2048)  # [B, T, C, N] (C = 3 coords + 1 feature)
    out = model(xyz_seq)
    print(out.shape)  # Debería ser [B, N, num_classes]

