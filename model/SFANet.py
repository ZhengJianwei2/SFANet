import torch.nn.functional as F
from model.Res import resnet18
from model.Res2Net_v1b import res2net50_v1b_26w_4s
from torch_geometric.nn import BatchNorm, global_max_pool
from model.graphbatch import Generate_edges_globally, graph_batch_re_trans
from model.myGatedEdgeConv import Gatedgcn, GatedEdgeConv, AdGatedEdgeConv
import math
from model.Attention import *

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, s=1, p=0, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class URM(nn.Module):
    def __init__(self, cur_channel):
        super(URM, self).__init__()
        self.relu = nn.ReLU(True)

        self.ca_fg = ChannelAttention(cur_channel)
        self.ca_edge = ChannelAttention(cur_channel)
        self.sa_fg = SpatialAttention()
        self.sa_edge = SpatialAttention()
        self.FE_conv = BasicConv2d(cur_channel, cur_channel, 3, padding=1)
        self.conv_fg = BasicConv2d(cur_channel, cur_channel, 1)
        self.conv_edge = BasicConv2d(cur_channel, cur_channel, 1)
        self.conv1 = BasicConv2d(cur_channel, cur_channel, 1)
        self.conv3 = BasicConv2d(cur_channel, cur_channel, 3, padding=1)
        self.conv5 = BasicConv2d(cur_channel, cur_channel, 5, padding=2)
        self.conv7 = BasicConv2d(cur_channel, cur_channel, 7, padding=3)
        self.cbr = convbnrelu(cur_channel * 4, cur_channel, 1)

    def forward(self, x):
        #  The code will be open source soon

        return x_define, x_edge


class SEM(nn.Module):

    def __init__(self, c_in, c_feat, c_atten):
        super(SEM, self).__init__()
        self.c_feat = c_feat
        self.c_atten = c_atten
        self.conv_feat = nn.Conv2d(c_in, c_feat, kernel_size=1)
        self.conv_atten = nn.Conv2d(c_in, c_atten, kernel_size=1)

    def forward(self, input: torch.Tensor):
        #  The code will be open source soonn)

        return descriptors


class SDM(nn.Module):

    def __init__(self, c_atten, c_de):
        super(SDM, self).__init__()
        self.c_atten = c_atten
        self.c_de = c_de
        self.conv_de = nn.Conv2d(c_atten, c_atten // 4, kernel_size=1)
        self.out_conv = nn.Conv2d(c_atten, c_de, kernel_size=1)

    def forward(self, descriptors: torch.Tensor, input_de: torch.Tensor):
        b, c, h, w = input_de.size()
        atten_vectors = F.softmax(self.conv_de(input_de), dim=1)
        output = descriptors.matmul(atten_vectors.view(b, self.c_atten // 4, -1)).view(b, -1, h, w)

        return self.out_conv(output)


class LRM(nn.Module):

    def __init__(self, c_en, c_de):
        super(LRM, self).__init__()
        self.c_en = c_en
        self.c_de = c_de
        self.conv_1 =  nn.Conv2d(c_de, c_en, kernel_size=1, bias=False)
    def forward(self, input_en: torch.Tensor, input_de: torch.Tensor, gate_map):
        b, c, h, w = input_de.size()
        input_en = input_en.view(b, self.c_en, -1)
        # Channel Resampling
        energy = self.conv_1(input_de).view(b, self.c_en, -1).matmul(input_en.transpose(-1, -2))
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(
            energy) - energy  # Prevent loss divergence during training
        channel_attention_map = torch.softmax(energy_new, dim=-1)
        input_en = channel_attention_map.matmul(input_en).view(b, -1, h, w)  # channel_attention_feat

        # Spatial Gating
        gate_map = torch.sigmoid(gate_map)
        input_en = input_en.mul(gate_map)

        return input_en



class IFM(nn.Module):

    def __init__(self, fpn_dim, c_atten):
        super(IFM, self).__init__()
        self.fqn_dim = fpn_dim
        self.c_atten = c_atten
        self.sdm = SDM(c_atten, fpn_dim)
        self.lrm = LRM(fpn_dim, c_atten)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            # norm_layer(fpn_dim),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Conv2d(c_atten, fpn_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, input_en, input_de, global_descripitors):
        #  The code will be open source soon
        return self.conv_fusion(self.conv(input_de) + self.alpha * feat_global + self.beta * feat_local)


class GraphInference(nn.Module):
    def __init__(self, dim, loop, bknum, thr1):
        super(GraphInference, self).__init__()
        self.thr1 = thr1  # 阈值
        self.bknum = bknum  #块数量
        self.loop = loop  #循环次数
        self.gcn1 = AdGatedEdgeConv(dim, dim)

        self.bn1 = BatchNorm(dim)

        self.relu = nn.ReLU()
        self.mlpCA1 = torch.nn.Sequential(torch.nn.Linear(dim, dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(dim, dim))
        self.lineFu = torch.nn.Linear(dim, dim)

    def forward(self, x):  # [B, C, node_num]
        #  The code will be open source soon
        return output  # [B, C_new, node_num]


class GraphProjection(nn.Module):
    def __init__(self, bnum, bnod, dim, normalize_input=False):
        super(GraphProjection, self).__init__()
        self.bnum = bnum
        self.bnod = bnod
        self.node_num = bnum * bnum * bnod
        self.dim = dim
        self.normalize_input = normalize_input
        self.anchor = nn.Parameter(torch.rand(self.node_num, dim))
        self.sigma = nn.Parameter(torch.rand(self.node_num, dim))

    def gen_soft_assign(self, x, sigma):
        B, C, H, W = x.size()
        soft_assign = torch.zeros([B, self.node_num, self.n], device=x.device, dtype=x.dtype, layout=x.layout)
        soft_ass = torch.zeros([B, self.node_num, self.n], device=x.device, dtype=x.dtype, layout=x.layout)

        for node_id in range(self.node_num):
            block_id = math.floor(node_id / self.bnod)
            h_sta = math.floor(block_id / self.bnum) * self.h
            w_sta = block_id % (self.bnum) * self.w
            h_end = h_sta + self.h
            w_end = w_sta + self.w
            tmp = x.view(B, C, H, W)[:, :, h_sta:h_end, w_sta: w_end]
            tmp = tmp.reshape(B, C, -1).permute(0, 2, 1).contiguous()
            residual = (tmp - self.anchor[node_id, :]).div(sigma[node_id, :])
            soft_assign[:, node_id, :] = -torch.pow(torch.norm(residual, dim=2), 2) / 2

        for block_id in range(self.bnum * self.bnum):
            node_sta = self.bnod * block_id
            node_end = node_sta + self.bnod
            soft_ass[:, node_sta:node_end, :] = F.softmax(soft_assign[:, node_sta:node_end, :], dim=1)
        return soft_ass  # B node_num n

    def forward(self, x):
       #  The code will be open source soon
        return nodes.view(B, self.node_num, C).permute(0, 2, 1).contiguous(), soft_assign


class GraphProcess(nn.Module):
    def __init__(self, bnum, bnod, dim, loop, thr1):
        super(GraphProcess, self).__init__()
        self.loop = loop
        self.bnum = bnum  # 块的数量
        self.bnod = bnod  # 每个块中的节点数量
        self.dim = dim
        self.node_num = bnum * bnum * bnod   # 节点数量
        self.proj = GraphProjection(self.bnum, self.bnod, self.dim)
        self.gconv = GraphInference(self.dim, self.loop, self.bnum, thr1)

    def GraphReprojection(self, Q, Z):   # Q:(B, N, C), Z:(B, C, N)
        #  The code will be open source soon
        return res  # 投影特征

    def forward(self, x):  #
        #  The code will be open source soon
        return res


class GCNModule(nn.Module):
    def __init__(self, channel, rdim):
        super(GCNModule, self).__init__()
        self.dim = channel
        self.rdim = rdim
        self.conv = BasicConv2d(self.dim, self.rdim, kernel_size=1)
        self.gcnlayer = GraphProcess(bnum=8, bnod=2, dim=self.rdim, loop=3, thr1=0.6)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x3):
        #  The code will be open source soon
        return x


class decoder(nn.Module):
    def __init__(self, channel=512):
        super(decoder, self).__init__()

        self.decoder3 = nn.Sequential(
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1),
            # nn.Dropout(0.5),
            TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S3 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder2 = nn.Sequential(
            BasicConv2d(96, 32, 3, padding=1),
            BasicConv2d(32, 32, 3, padding=1),
            # nn.Dropout(0.5),
            TransBasicConv2d(32, 32, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        )
        self.S2 = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        self.decoder1 = nn.Sequential(
            BasicConv2d(48, 16, 3, padding=1),
            BasicConv2d(16, 16, 3, padding=1),
        )
        self.S1 = nn.Conv2d(16, 1, 3, stride=1, padding=1)

    def forward(self,x3, x2, x1):


        x3_up = self.decoder3(x3)
        s3 = self.S3(x3_up)

        x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
        s2 = self.S2(x2_up)

        x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        s1 = self.S1(x1_up)


        return s1, s2, s3



class SFANet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(SFANet, self).__init__()
        #Backbone model
        self.bkbone = res2net50_v1b_26w_4s(pretrained=True)
        # self.bkbone = resnet50_XMZ(pretrained=1)

        self.conv1 = BasicConv2d(64, 16, kernel_size=1, stride=1)
        self.conv2 = BasicConv2d(256, 32, kernel_size=1, stride=1)
        self.conv3 = BasicConv2d(512, 64, kernel_size=1, stride=1)
        self.conv4 = BasicConv2d(1024, 128, kernel_size=1, stride=1)
        self.conv5 = BasicConv2d(2048, 256, kernel_size=1, stride=1)

        # self.sem_d5 = SEM(512, 512, 512 // 4)
        # self.ifm_d4 = IFM(256, 512)
        # self.sem_d4 = SEM(256, 256, 256 // 4)
        # self.ifm_d3 = IFM(128, 256)

        self.sem_d5 = SEM(256, 256, 256 // 4)
        self.ifm_d4 = IFM(128, 256)
        self.sem_d4 = SEM(128, 128, 128 // 4)
        self.ifm_d3 = IFM(64, 128)

        self.gcnlayer = GCNModule(64, 64)

        self.urm_d2 = URM(32)
        self.urm_d1 = URM(16)

        self.decoder_rgb = decoder()

        # self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        # self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()

        self.S1 = nn.Conv2d(16, 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

    def forward(self, x):
        # x1, x2, x3, x4, x5 = self.bkbone(x)

        x = self.bkbone.conv1(x)
        x = self.bkbone.bn1(x)
        x1 = self.bkbone.relu(x)
        x = self.bkbone.maxpool(x1)
        # ---- low-level features ----
        x2 = self.bkbone.layer1(x)
        x3 = self.bkbone.layer2(x2)
        x4 = self.bkbone.layer3(x3)
        x5 = self.bkbone.layer4(x4)

        # print(x1.shape, x2.shape,x3.shape, x4.shape, x5.shape)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)

        # up means update
        att5 = self.sem_d5(x5)
        x5 = self.upsample2(x5)
        s4 = self.ifm_d4(x4, x5, att5)
        att4 = self.sem_d4(s4)
        x4 = self.upsample2(x4)
        s3 = self.ifm_d3(x3, x4, att4)
        out_s3 = self.S3(s3)

        s3 = self.gcnlayer(s3)
        s2, edge2 = self.urm_d2(x2)
        s1, edge1 = self.urm_d1(x1)
      
        s1, s2, s3 = self.decoder_rgb(s3, s2, s1)

        s1 = self.upsample2(s1)
        s2 = self.upsample2(s2)
        s3 = self.upsample4(s3)
        edge1 = self.upsample2(self.S1(edge1))
        edge2 = self.upsample4(self.S2(edge2))

        return out_s3, s1, s2, s3, self.sigmoid(s1), self.sigmoid(s2), self.sigmoid(s3), edge1, edge2
