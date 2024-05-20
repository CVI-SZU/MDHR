import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer import *
from .resnet import *
import math
from .GAT import GATConv
from .utils import calculate_adjmask

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x

class NodeClassifier(nn.Module):
    def __init__(self, class_num=12, in_channels=512):
        super(NodeClassifier, self).__init__()
        self.num_classes = class_num
        self.in_channels = in_channels
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()

        nn.init.xavier_uniform_(self.sc)

    def forward(self,x):
        assert x.shape[1] == self.num_classes
        assert x.shape[2] == self.in_channels
        b, n, c = x.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(x, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        return cl

class MultiscaleFacialDynamic(nn.Module):
    def __init__(self, backbone, k=5):
        super(MultiscaleFacialDynamic, self).__init__()
        self.backbone = backbone
        self.k = k
        self.inchannel = self.get_inchannel(backbone)
        self.out_channel = 512

        self.conv1 = nn.Conv2d(self.inchannel, 8 * self.inchannel, kernel_size=8, stride=8, padding=0)
        self.conv2 = nn.Conv2d(2 * self.inchannel, 8 * self.inchannel, kernel_size=4, stride=4, padding=0)
        self.conv3 = nn.Conv2d(4 * self.inchannel, 8 * self.inchannel, kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(8 * self.inchannel, 8 * self.inchannel, kernel_size=1, stride=1, padding=0)

        self.convcat = nn.Conv2d(8 * self.inchannel, self.out_channel, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(8 * self.inchannel)
        self.bn2 = nn.BatchNorm2d(8 * self.inchannel)
        self.bn3 = nn.BatchNorm2d(8 * self.inchannel)
        self.bn4 = nn.BatchNorm2d(8 * self.inchannel)

        self.bn = nn.BatchNorm1d(self.out_channel)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

        self.pooling = nn.AvgPool3d(kernel_size=(self.k*2, 1, 1), stride=(1, 1, 1), padding=(self.k, 0, 0))

        self.relu = nn.ReLU(inplace=True)

        self.weight_1 = nn.Conv2d(8*self.inchannel,16,kernel_size=1,stride=1,padding=0)
        self.weight_2 = nn.Conv2d(8*self.inchannel,16,kernel_size=1,stride=1,padding=0)
        self.weight_3 = nn.Conv2d(8*self.inchannel,16,kernel_size=1,stride=1,padding=0)
        self.weight_4 = nn.Conv2d(8*self.inchannel,16,kernel_size=1,stride=1,padding=0)

        self.weight = nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0)

        self.wbn1 = nn.BatchNorm2d(16)
        self.wbn2 = nn.BatchNorm2d(16)
        self.wbn3 = nn.BatchNorm2d(16)
        self.wbn4 = nn.BatchNorm2d(16)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_inchannel(self,backbone):
        if backbone == 'resnet':
            return 256
        elif backbone == 'swin_transformer':
            return 128
        else:
            raise ValueError(f"Error: wrong backbone: {backbone}")

    def forward(self, m_out, B, T):
        x1, x2, x3, x4 = m_out
        if self.backbone == 'resnet':
            x1 = x1.view(B, T, self.inchannel, 56, 56)
            x2 = x2.view(B, T, 2 * self.inchannel, 28, 28)
            x3 = x3.view(B, T, 4 * self.inchannel, 14, 14)
            x4 = x4.view(B, T, 8 * self.inchannel, 7, 7)
        elif self.backbone == 'swin_transformer':
            x1 = x1.permute(0, 2, 1).view(B, T, self.inchannel, 56, 56)
            x2 = x2.permute(0, 2, 1).view(B, T, 2 * self.inchannel, 28, 28)
            x3 = x3.permute(0, 2, 1).view(B, T, 4 * self.inchannel, 14, 14)
            x4 = x4.permute(0, 2, 1).view(B, T, 8 * self.inchannel, 7, 7)
        else:
            raise Exception("Error: wrong backbone: ", backbone)

        x1_diff = torch.diff(x1, dim=1)
        x1_diff = x1_diff.view(B * (T-1), self.inchannel, 56, 56)

        x2_diff = torch.diff(x2, dim=1)
        x2_diff = x2_diff.view(B * (T-1), 2 * self.inchannel, 28, 28)

        x3_diff = torch.diff(x3, dim=1)
        x3_diff = x3_diff.view(B * (T-1), 4 * self.inchannel, 14, 14)

        x4_diff = torch.diff(x4, dim=1)
        x4_diff = x4_diff.view(B * (T-1), 8 * self.inchannel, 7, 7)

        x1_diff = self.conv1(x1_diff)
        x1_diff = self.relu(self.bn1(x1_diff))
        x1_diff = self.pooling(x1_diff.view(B, T-1, 8 * self.inchannel, 7, 7).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous().view(B * T, 8 * self.inchannel, 7, 7)

        x2_diff = self.conv2(x2_diff)
        x2_diff = self.relu(self.bn2(x2_diff))
        x2_diff = self.pooling(x2_diff.view(B, T-1, 8 * self.inchannel, 7, 7).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous().view(B * T, 8 * self.inchannel, 7, 7)

        x3_diff = self.conv3(x3_diff)
        x3_diff = self.relu(self.bn3(x3_diff))
        x3_diff = self.pooling(x3_diff.view(B, T-1, 8 * self.inchannel, 7, 7).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous().view(B * T, 8 * self.inchannel, 7, 7)

        x4_diff = self.conv4(x4_diff)
        x4_diff = self.relu(self.bn4(x4_diff))
        x4_diff = self.pooling(x4_diff.view(B, T-1, 8 * self.inchannel, 7, 7).permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4).contiguous().view(B * T, 8 * self.inchannel, 7, 7)

        x1_weight = self.weight_1(x1_diff)
        x1_weight = self.relu(self.wbn1(x1_weight))
        x2_weight = self.weight_2(x2_diff)
        x2_weight = self.relu(self.wbn2(x2_weight))
        x3_weight = self.weight_3(x3_diff)
        x3_weight = self.relu(self.wbn3(x3_weight))
        x4_weight = self.weight_4(x4_diff)
        x4_weight = self.relu(self.wbn4(x4_weight))

        x_weight = torch.cat([x1_weight, x2_weight, x3_weight, x4_weight], dim=1)
        x_weight = self.weight(x_weight)
        x_weight = F.softmax(x_weight, dim=1)
        x = x1_diff * x_weight[:, 0:1, :, :] + x2_diff * x_weight[:, 1:2, :, :] + x3_diff * x_weight[:, 2:3, :, :] + x4_diff * x_weight[:, 3:4, :, :]
        x = self.convcat(x)
        x = x.view(B * T, self.out_channel, 7 * 7)
        x = self.relu(self.bn(x)).permute(0, 2, 1)
        return x

class LocalRelationshipModeling(nn.Module):
    def __init__(self, num_classes=12, in_channels=512, out_channels=512):
        super(LocalRelationshipModeling, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.sub_class = [16, 2, 8, 16]

        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(self.in_channels, self.out_channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)

        self.up_linear = LinearBlock(self.in_channels, self.out_channels)
        self.mid_linear = LinearBlock(self.in_channels, self.out_channels)
        self.down1_linear = LinearBlock(self.in_channels, self.out_channels)
        self.down2_linear = LinearBlock(self.in_channels, self.out_channels)

        self.up_fc = nn.Linear(self.in_channels, self.sub_class[0])
        self.mid_fc = nn.Linear(self.in_channels, self.sub_class[1])
        self.down1_fc = nn.Linear(self.in_channels, self.sub_class[2])
        self.down2_fc = nn.Linear(self.in_channels, self.sub_class[3])

    def forward(self,x):
        BT, HW, C = x.shape
        assert HW == 49
        assert C == self.in_channels

        x = x.view(BT, 7, 7, self.in_channels)
        x_up = x[:, :3, :, :].view(BT, -1, self.in_channels)
        x_mid = x[:, 2:5, :, :].view(BT, -1, self.in_channels)
        x_down = x[:, 4:, :, :].view(BT, -1, self.in_channels)

        x_up = self.up_linear(x_up)
        x_mid = self.mid_linear(x_mid)
        x_down1 = self.down1_linear(x_down)
        x_down2 = self.down2_linear(x_down)

        up_pred = self.up_fc(x_up.mean(dim=1))
        mid_pred = self.mid_fc(x_mid.mean(dim=1))
        down1_pred = self.down1_fc(x_down1.mean(dim=1))
        down2_pred = self.down2_fc(x_down2.mean(dim=1))

        AU1 = self.class_linears[0](x_up).unsqueeze(1)
        AU2 = self.class_linears[1](x_up).unsqueeze(1)
        AU4 = self.class_linears[2](x_up).unsqueeze(1)
        AU6 = self.class_linears[3](x_mid).unsqueeze(1)
        AU7 = self.class_linears[4](x_up).unsqueeze(1)
        AU10 = self.class_linears[5](x_down2).unsqueeze(1)
        AU12 = self.class_linears[6](x_down1).unsqueeze(1)
        AU14 = self.class_linears[7](x_down1).unsqueeze(1)
        AU15 = self.class_linears[8](x_down1).unsqueeze(1)
        AU17 = self.class_linears[9](x_down2).unsqueeze(1)
        AU23 = self.class_linears[10](x_down2).unsqueeze(1)
        AU24 = self.class_linears[11](x_down2).unsqueeze(1)

        AU_node = [AU1, AU2, AU4, AU6, AU7, AU10, AU12, AU14, AU15, AU17, AU23, AU24]

        f_v = torch.cat(AU_node, dim=1)
        f_v = f_v.mean(dim=-2)

        return up_pred, mid_pred, down1_pred, down2_pred, f_v

class CrossregionRelationshipModeling(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CrossregionRelationshipModeling, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.gat = GATConv(in_feats=in_channels, out_feats=in_channels, num_heads=1, dropout=0., thred=0.1, residual=True)

    def forward(self, x, adj_mask):
        adj = torch.tensor([[0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], #AU1
                             [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], #AU2
                             [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], #AU4
                             [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], #AU6
                             [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1], #AU7
                             [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], #AU10
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], #AU12
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], #AU14
                             [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], #AU15
                             [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], #AU17
                             [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0], #AU23
                             [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]]) #AU24
        eye = torch.eye(12).float().cuda()
        adj = adj.unsqueeze(0).float().cuda()
        adj = adj*adj_mask + eye
        x = self.gat(adj, x)
        return x

class IndividualTemporalModeling(nn.Module):
    def  __init__(self, num_classes=12, in_channels=512):
        super(IndividualTemporalModeling, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.bn = nn.BatchNorm1d(num_classes)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

        temporal_cnn_layers = []
        for i in range(self.num_classes):
            layer = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=5, padding=2, groups=self.in_channels)
            temporal_cnn_layers += [layer]
        self.temporal_cnn = nn.ModuleList(temporal_cnn_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, B, T):
        BT, N, d = x.shape
        x = x.view(B,T,N,d)
        v = []
        for i, layer in enumerate(self.temporal_cnn):
            v.append(layer(x[:, :, i, :].permute(0, 2, 1)).permute(0, 2, 1).unsqueeze(1))
        v = torch.cat(v, dim=1)
        v = v.view(B, N, T, d).transpose(1, 2).contiguous().view(B * T, N, d)
        x = self.relu(self.bn(v))
        return x

class MDHR(nn.Module):
    def __init__(self, dataset='BP4D', num_classes=12, backbone='resnet', k=5):
        super(MDHR, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.backbone = backbone
        self.k = k
        
        if 'swin' in backbone:
            self.Backbone = swin_transformer_base(pretrained=True)
            self.in_channels = self.Backbone.num_features
            self.out_channels = self.in_channels // 2
            self.Backbone.head = None

        elif 'resnet' in backbone:
            self.Backbone = resnet50(pretrained=True)
            self.in_channels = self.Backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.Backbone.fc = None
        else:
            raise Exception("Error: wrong backbone: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.MFD = MultiscaleFacialDynamic(backbone=self.backbone, k=self.k)
        self.LRM = LocalRelationshipModeling(num_classes=self.num_classes, in_channels=self.out_channels, out_channels=self.out_channels)
        self.CRM = CrossregionRelationshipModeling(in_channels=self.out_channels, num_classes=self.num_classes)
        self.TCN = IndividualTemporalModeling(num_classes=self.num_classes, in_channels=self.out_channels)
        self.node_classifier = NodeClassifier(class_num=self.num_classes, in_channels=self.out_channels)

    def forward(self, x):
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        x, multiscale_out = self.Backbone(x)
        x_static = self.global_linear(x)
        x_dynamic = self.MFD(multiscale_out, B, T)
        x = x_static + x_dynamic
        up_pred, mid_pred, down1_pred, down2_pred, x = self.LRM(x)
        adj_mask = calculate_adjmask(up_pred.detach(), mid_pred.detach(), down1_pred.detach(), down2_pred.detach())
        x = self.CRM(x, adj_mask)
        x = self.TCN(x, B, T)
        x = self.node_classifier(x)
        return up_pred, mid_pred, down1_pred, down2_pred, x



