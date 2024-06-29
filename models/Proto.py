import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import Conv_4, ResNet
import math
from .CSIW import CSIW
from .SCIW import SCIW
from .CompactBilinearPooling import CompactBilinearPooling

class Bilinear_feature_Block(nn.Module):

    def __init__(self, in_c, p):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_c,
                                 out_features=in_c ,
                                 bias=False)
        self.bn1 = nn.BatchNorm1d(in_c )

        self.dp = nn.Dropout(p)


    def forward(self, x):

        output = torch.tanh(x)
        output = self.bn1(output)
        output = self.dp(output)
        output = self.linear1(output)

        return output

class Proto(nn.Module):

    def __init__(self, args=None):

        super().__init__()

        self.args = args
        self.shots = [self.args.train_shot, self.args.train_query_shot]
        self.way = self.args.train_way
        self.resnet = self.args.resnet

        self.resolution = 25
        if self.resnet:
            self.num_channel = 640
            self.dim = 640
            self.feature_extractor = ResNet.resnet12(drop=True)

        else:
            self.num_channel = 64
            self.dim = 64 * 25
            self.feature_extractor = Conv_4.BackBone(self.num_channel)

        self.G = self.args.G
        self.Q = self.args.Q
        if self.resnet:
            self.in_c = 32 * 32
            p = 0.2
        else:
            self.in_c = 8 * 8
            p = 0
        self.qry_self = Bilinear_feature_Block(self.Q * self.Q, p)
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        self.W1 = CSIW(self.args)
        self.W2 = SCIW(self.args)
        self.CBP= CompactBilinearPooling(640, 640, 640)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def get_feature_vector(self, inp):

        feature_map = self.feature_extractor(inp)

        return feature_map

    def get_neg_l2_dist(self, inp, way, shot, query_shot):

        feature_map = self.get_feature_vector(inp)
        _, d, h, w = feature_map.shape
        m = h * w
        support = feature_map[:way * shot].view(way, shot, d, m)
        query = feature_map[way * shot:].view(1, -1, d, m)
        query_num = query.shape[1]

        if self.args.Model:

            weight1 = self.W1(support, query.transpose(0, 1))
            weight2 = self.W2(support, query.transpose(0, 1))
            centroid = support
            centroid1 = centroid.view(way * shot, 1, d, m) * weight1.view(way * shot, query_num, d, 1)
            query1 = query * weight1.view(way * shot, query_num, d, 1)
            centroid2 = centroid.view(way * shot, 1, d, m) * weight2.view(way * shot, query_num, 1, m)
            query2 = query * weight2.view(way * shot, query_num, 1, m)
            G = self.G
            Q = self.Q

            if self.args.Group == 'E':

                img = torch.cat((support.mean(dim=1).view(way, d, m), query.view(query_num, d, m)), 0)
                img = img.mean(dim=0)
                img2 = img.view(1, d, m)
                img1 = img.view(d, 1, m)
                my = torch.sum(torch.pow(img1 - img2, 2), dim=-1)
                a_value, a_indices = torch.sort(my, dim=-1)
                a_value_mean = a_value.mean(dim=0)
                a_value_sum = a_value[:, :Q].sum(dim=1)
                t_values, t_indices = torch.min(a_value_sum, dim=0)
                a_indices_i = a_indices[t_indices, :Q].view(1, Q)
                my_ii = my
                indices = a_indices_i
                values = a_value[t_indices, :Q].view(1, Q)
                C = 0
                while C < G - 1:
                    my_ii[:, a_indices_i] = 1000
                    a_value_i, a_indices_i = torch.sort(my_ii, dim=-1)
                    a_value_i_sum = a_value_i[:, :Q].sum(dim=1)
                    t_i, t_i_indices = torch.min(a_value_i_sum, dim=0)
                    a_indices_i = a_indices_i[t_i_indices, :Q].view(1, Q)
                    a_value_i = a_value_i[t_i_indices, :Q].view(1, Q)
                    indices = torch.cat((indices, a_indices_i), 0)
                    values = torch.cat((values, a_value_i), 0)
                    C = C + 1

            elif self.args.Group == 'C':

                img = torch.cat((support.mean(dim=1).view(way, d, m), query.view(query_num, d, m)), 0)
            #     img = img.mean(dim=0)
            #     my_rho_i = torch.corrcoef(img)
            #     a_value, a_indices = torch.sort(my_rho_i, dim=-1, descending=True)
            #     a_value_mean = a_value.mean(dim=0)
            #     a_value_sum = a_value[:, :Q].sum(dim=1)
            #     t_values, t_indices = torch.max(a_value_sum, dim=0)
            #     a_indices_i = a_indices[t_indices, :Q].view(1, Q)
            #     my_rho_ii = my_rho_i
            #     indices = a_indices_i
            #     values = a_value[t_indices, :Q].view(1, Q)
            #     C = 0
            #     while C < G - 1:
            #         my_rho_ii[:, a_indices_i] = -1
            #         a_value_i, a_indices_i = torch.sort(my_rho_ii, dim=-1, descending=True)
            #         a_value_i_sum = a_value_i[:, :Q].sum(dim=1)
            #         t_i, t_i_indices = torch.max(a_value_i_sum, dim=0)
            #         a_indices_i = a_indices_i[t_i_indices, :Q].view(1, Q)
            #         a_value_i = a_value_i[t_i_indices, :Q].view(1, Q)
            #         indices = torch.cat((indices, a_indices_i), 0)
            #         values = torch.cat((values, a_value_i), 0)
            #         C = C + 1
            #
            # j = 0
            # p_c = torch.zeros([way * shot * query_num, Q * Q, 1]).cuda()
            # p_q = torch.zeros([way * shot*query_num, Q * Q, 1]).cuda()
            # while j < G:
            #     centroid_1 = torch.index_select(centroid1, -2, indices[j])
            #     query_1 = torch.index_select(query1, -2, indices[j])
            #     centroid_2 = torch.index_select(centroid2, -2, indices[j])
            #     query_2 = torch.index_select(query2, -2, indices[j])
            #     centroid_3 = torch.bmm(centroid_1.view(way * shot * query_num, Q, m),
            #                            centroid_2.view(way * shot * query_num, Q, m).transpose(-1, -2)) / m
            #     query_3 = torch.bmm(query_1.view(way * shot * query_num, Q, m),
            #                         query_2.view(way * shot * query_num, Q, m).transpose(-1, -2)) / m
            #     centroid_3 = centroid_3.view(way * shot * query_num, Q * Q)
            #     query_3 = query_3.view(way * shot * query_num, Q * Q)
            #     centroid_4 = self.qry_self(centroid_3).view(way * shot * query_num, Q * Q, 1)
            #     query_4 = self.qry_self(query_3).view(way * shot * query_num, Q * Q, 1)
            #     p_c = torch.cat((p_c, centroid_4), dim=-1)
            #     p_q = torch.cat((p_q, query_4), dim=-1)
            #     j = j + 1
            #
            # x = torch.arange(1, G + 1).cuda()
            # p_c = torch.index_select(p_c, -1, x)
            # p_q = torch.index_select(p_q, -1, x)
            #
            # centroid = p_c.view(way, shot, query_num,self.G*self.Q*self.Q).sum(dim=1)
            # query = p_q.view(way,  shot,query_num, self.G*self.Q*self.Q).sum(dim=1)

            centroid = support
            p_c = centroid.view(way * shot, 1, d, m) * weight2.view(way * shot, query_num, 1, m)
            p_q = query * weight2.view(way * shot, query_num, 1, m)
            centroid = p_c.mean(dim=-1)
            query = p_q.mean(dim=-1)

        if self.args.Proto:

            centroid = support.mean(dim=1).unsqueeze(dim=1)
            centroid = centroid.mean(dim=-1)
            query = query.mean(dim=-1)

        l2_dist = torch.sum(torch.pow(centroid - query, 2), dim=-1).transpose(0, 1)
        neg_l2_dist = l2_dist.neg()

        return neg_l2_dist

    def meta_test(self, inp, way, shot, query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                          way=way,
                                          shot=shot,
                                          query_shot=query_shot
                                          )

        _, max_index = torch.max(neg_l2_dist, 1)

        return max_index

    def forward(self, inp):

        neg_l2_dist= self.get_neg_l2_dist(inp=inp,
                                          way=self.way,
                                          shot=self.shots[0],
                                          query_shot=self.shots[1]
                                          )
        if self.args.Model:
            logits = neg_l2_dist.cuda() / (self.G * self.Q * self.Q * self.shots[0]) * self.scale.cuda()
        if self.args.Proto:
            logits = neg_l2_dist / self.dim * self.scale
        log_prediction = F.log_softmax(logits, dim=1)
        return log_prediction
