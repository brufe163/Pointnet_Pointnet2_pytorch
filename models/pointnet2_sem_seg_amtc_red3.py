import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

class get_model(nn.Module):
    def __init__(self, num_classes, num_feats, dropout=0.5):
        super(get_model, self).__init__()
        
        # in_channel = num_feats (no sumamos +3, las coords se agregan internamente en SA)
        self.sa1 = PointNetSetAbstraction(
            npoint=256,
            radius=0.1,
            nsample=32,
            in_channel=num_feats,
            mlp=[8, 8, 16],
            group_all=False
        )

        # En fp1: l1_points = 16 canales, l0_points = num_feats canales (sin xyz)
        # total in_channel = 16 + num_feats
        self.fp1 = PointNetFeaturePropagation(16 + num_feats, [16,16])

        self.conv1 = nn.Conv1d(16, 16, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(16, num_classes, 1)

    def forward(self, xyz):
        # xyz: [B, 3+num_feats, N]
        # Separamos xyz y feats
        l0_xyz = xyz[:, :3, :]               # xyz: 3 canales
        l0_points = xyz[:, 3:, :]            # solo las características: num_feats canales

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points) 
        # l1_points: [B,16,256]

        # Al llegar a fp1, sumamos l0_points (4 canales si num_feats=4) y l1_points (16 canales)
        # total = 20 canales
        # Nos aseguramos que fp1 se inicializó con in_channel=20 (16+4)
        # Como la definimos con (16+num_feats), si num_feats=4, es 20.
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points) 
        # salida: [B,16,N]

        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # [B, N, num_classes]

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
