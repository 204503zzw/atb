import torch
import torch.nn as nn
import torch.nn.functional as F
class S_Block(nn.Module):

    def __init__(self, in_c):
        super().__init__()

        self.linear1 = nn.Linear(in_features=in_c,
                                 out_features=in_c,
                                 bias=False)
        self.bn1 = nn.BatchNorm1d(in_c)

    def forward(self, x):

        output = self.linear1(x)
        output = self.bn1(output)
        output = torch.tanh(output)
        output = 1 + output

        return output

class Sj_Block(nn.Module):

    def __init__(self, channel):
        super().__init__()

        reduction=2
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        print(max_out.shape)
        avg_out = self.mlp(self.avg_pool(x))
        x = self.sigmoid(max_out + avg_out)
        return x

class SCIW(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args
        if self.args.resnet:
            self.in_c = 640
        else:
            self.in_c = 64
        self.p_self = S_Block(25)
        self.q_self = S_Block(25)

        self.S_p = Sj_Block(self.in_c)
        self.S_q = Sj_Block(self.in_c)

        self.prt_self = S_Block(25)
        self.qry_self = S_Block(25)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def add_noise(self, input):
        if self.training and self.args.noise:
            noise = ((torch.rand(input.shape).to(input.device) - .5) * 2) * self.args.noise_value
            input = input + noise
            input = input.clamp(min=0., max=2.)

        return input

    def dist(self, input, spt=False, normalize=True):

        if spt:
            way, c, m = input.shape
            input_C_gap = input.mean(dim=-2)

            input = input.reshape(way * c, m)
            input = input.unsqueeze(dim=1)
            input_C_gap = input_C_gap.unsqueeze(dim=0)

            dist = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)
            if normalize:
                dist = dist / m
            dist = dist.reshape(way, c, -1)
            dist = dist.transpose(-1, -2)

            indices_way = torch.arange(way)

            dist_self = dist[indices_way, indices_way]

            return dist_self

        else:
            batch, c, m = input.shape
            input_C_gap = input.mean(dim=-2).unsqueeze(dim=-2)

            dist = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)#求和
            if normalize:
                dist = dist / m

            return dist

    def weight(self, spt, qry):

        way, shot, c, m = spt.shape
        batch, _, _, _ = qry.shape

        prt = spt.view(way * shot, c, m)
        qry = qry.squeeze(dim=1)

        prt1 = prt.mean(dim=-2)
        qry1 = qry.mean(dim=-2)
        s = self.p_self(prt1)
        q = self.q_self(qry1)
        prt = prt * s.view(way* shot, 1, m)
        qry = qry * q.view(batch, 1, m)

        prt2 = self.S_p(prt.view(way* shot, c,5,5))
        qry2 = self.S_q(qry.view(batch, c, 5, 5))

        dist1 = torch.sum(torch.pow(prt - prt2.view(way*shot, c, 1), 2), dim=-2)
        dist2 = torch.sum(torch.pow(qry - qry2.view(batch, c, 1), 2), dim=-2)

        weight_prt_self = self.prt_self((-1)*dist1)
        weight_prt_self = weight_prt_self.view(way*shot, 1, m)
        weight_qry_self = self.qry_self((-1)*dist2)
        weight_qry_self = weight_qry_self.view(1, batch, m)
        weight = 0.5 * weight_prt_self + 0.5 * weight_qry_self

        return weight


    def forward(self, spt, qry):
        weight = self.weight(spt, qry)
        weight = self.add_noise(weight)

        return weight
