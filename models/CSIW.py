import torch
import torch.nn as nn
import torch.nn.functional as F
class S_Block(nn.Module):

    def __init__(self, in_c):
        super().__init__()

        self.linear1 = nn.Linear(in_features=in_c ,
                                 out_features=in_c,
                                 bias=False)
        self.bn1 = nn.BatchNorm1d(in_c)

    def forward(self, x):
        output = self.linear1(x)
        output = self.bn1(output)
        output = torch.tanh(output)
        output = 1 + output

        return output

class Si_Block(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            #nn.Conv2d(2, 1, 1, groups=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        max_out1, _ = torch.max(x, dim=1, keepdim=True)
        avg_out1 = torch.mean(x, dim=1, keepdim=True)
        pool_cat1 = torch.cat([max_out1, avg_out1], dim=1)
        x = self.spatial_attention(pool_cat1)
        return x

class CSIW(nn.Module):

    def __init__(self, args):

        super().__init__()

        self.args = args
        if self.args.resnet:
            self.in_c = 640
        else:
            self.in_c = 64

        self.p_self = S_Block(self.in_c)
        self.q_self = S_Block(self.in_c)

        self.S_p = Si_Block(self.in_c)
        self.S_q = Si_Block(self.in_c)

        self.prt_self = S_Block(self.in_c)
        self.qry_self = S_Block(self.in_c)

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

            dist = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)
            if normalize:
                dist = dist / m

            return dist

    def weight(self, spt, qry):

        way, shot, c, m = spt.shape
        batch, _, _, _ = qry.shape

        prt = spt.view(way*shot,c,m)
        qry = qry.squeeze(dim=1)

        prt1 = prt.mean(dim=-1)
        qry1 = qry.mean(dim=-1)
        s = self.p_self(prt1)
        q = self.q_self(qry1)
        prt = prt * s.view(way*shot, c, 1)
        qry = qry * q.view(batch, c, 1)

        prt2 = self.S_p(prt.view(way*shot, c, 5, 5))
        qry2 = self.S_q(qry.view(batch, c, 5, 5))

        dist1 = torch.sum(torch.pow(prt - prt2.view(way*shot, 1, m), 2), dim=-1)
        dist2 = torch.sum(torch.pow(qry - qry2.view(batch, 1, m), 2), dim=-1)
        weight_prt_self = self.prt_self((-1)*dist1)
        weight_prt_self = weight_prt_self.view(way*shot, 1, c)
        weight_qry_self = self.qry_self((-1)*dist2)
        weight_qry_self = weight_qry_self.view(1, batch, c)
        weight = 0.5 * weight_prt_self + 0.5 * weight_qry_self

        return weight


    def forward(self, spt, qry):
        weight = self.weight(spt, qry)
        weight = self.add_noise(weight)

        return weight
