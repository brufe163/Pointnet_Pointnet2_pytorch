import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, num_classes, num_feats, dropout=0.5):
        super(get_model, self).__init__()
        # Reduce output features in sa1 and sa2
        self.sa1 = PointNetSetAbstraction(256, 0.1, 16, num_feats + 3, [16, 32], False)
        self.sa2 = PointNetSetAbstraction(64, 0.2, 16, 32 + 3, [32, 64], False)

        # Simplify feature propagation
        self.fp2 = PointNetFeaturePropagation(64 + 32, [32])
        self.fp1 = PointNetFeaturePropagation(32 + num_feats, [32])

        # Simplify final layers
        self.conv1 = nn.Conv1d(32, 16, 1)  # Reduce number of channels
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(16, num_classes, 1)  # Output layer

    def forward(self, xyz):
        l0_points = xyz  # xyz includes additional features
        l0_xyz = xyz[:, :3, :]  # Extract XYZ coordinates

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.dropout(x)
        x = self.conv2(x)

        x = x.permute(0, 2, 1)
        return x, None


if __name__ == '__main__':
    # Ejemplo de uso
    num_classes = 2
    num_feats = 4  # Por ejemplo, 4 características adicionales además de XYZ
    model = get_model(num_classes, num_feats)

    # Supongamos un batch con B=6, N=2048 puntos
    # Cada punto tiene 3 coords + 4 feats = 7 canales
    xyz = torch.rand(6, 7, 2048)
    output, _ = model(xyz)
    print(output.shape)  # Debería ser [6, 2048, 2]

        
class get_loss(nn.Module):
    def __init__(self, ignore_index=None):
        super(get_loss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.cross_entropy(pred, target, weight=weight, ignore_index=self.ignore_index)
        return total_loss

if __name__ == '__main__':
    import torch
    model = get_model(2, 4)
    xyz = torch.rand(6, 4, 2048)
    output, _ = model(xyz)
    print(output.shape)
