import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, num_classes, num_feats, dropout=0.5):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(512, 0.1, 32, num_feats + 3, [16, 16, 32], False)
        self.sa2 = PointNetSetAbstraction(128, 0.2, 32, 32 + 3, [32, 32, 64], False)

        # Ajustamos in_channel para que coincida con la suma de canales de points1 y points2
        self.fp2 = PointNetFeaturePropagation(96, [64, 32])
        self.fp1 = PointNetFeaturePropagation(32 + num_feats, [32, 32])

        self.conv1 = nn.Conv1d(32, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(32, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz  # xyz incluye características adicionales
        l0_xyz = xyz[:, :3, :]  # Coordenadas XYZ

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # Pasamos l0_points como points1, como en el modelo original
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.dropout(x)
        x = self.conv2(x)

        x = x.permute(0, 2, 1)
        return x, None
        
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
