import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, ChebConv, GCNConv, SAGEConv

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Parameter):
        nn.init.xavier_normal_(m)


def softmax_one(x, dim=1, _stacklevel=3, dtype=None):
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

class GCNNet(torch.nn.Module):
    def __init__(self, args):
        super(GCNNet, self).__init__()
        self.args = args

        if self.args.ss:

            self.beta_q10 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q11 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q12 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.beta_q20 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q21 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q22 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)

            self.bn = nn.BatchNorm1d(self.args.model_dim)

        elif self.args.ca:
            self.beta_q10 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q11 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q12 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)

            self.bn = nn.BatchNorm1d(self.args.model_dim)

        self.bn1 = nn.BatchNorm1d(32)


        if self.args.ss:
            self.nets_1 = torch.nn.ModuleList()
            self.nets_2 = torch.nn.ModuleList()
            self.bns_1 = torch.nn.ModuleList()
            self.bns_2 = torch.nn.ModuleList()

            self.nets_1.append(GCNConv(self.args.model_dim, 64))
            self.nets_1.append(GCNConv(64, 128))
            self.nets_1.append(GCNConv(128, 64))

            self.nets_2.append(GCNConv(self.args.model_dim, 64))
            self.nets_2.append(GCNConv(64, 128))
            self.nets_2.append(GCNConv(128, 64))

            self.bns_1.append(nn.BatchNorm1d(64))
            self.bns_1.append(nn.BatchNorm1d(128))
            self.bns_1.append(nn.BatchNorm1d(64))

            self.bns_2.append(nn.BatchNorm1d(64))
            self.bns_2.append(nn.BatchNorm1d(128))
            self.bns_2.append(nn.BatchNorm1d(64))

            for layer in range(self.args.num_layer):
                self.nets_1.append(GCNConv(64, 64))
                self.nets_2.append(GCNConv(64, 64))
                self.bns_1.append(nn.BatchNorm1d(64))
                self.bns_2.append(nn.BatchNorm1d(64))

            self.nets_1.append(GCNConv(64, 32))

            self.nets_2.append(GCNConv(64, 32))

            self.loss_w = torch.nn.Parameter(torch.ones([64, 32]))
        else:
            self.nets_1 = torch.nn.ModuleList()
            self.bns_1 = torch.nn.ModuleList()
            self.nets_1.append(GCNConv(self.args.model_dim, 64))
            self.nets_1.append(GCNConv(64, 128))
            self.nets_1.append(GCNConv(128, 64))
            self.bns_1.append(nn.BatchNorm1d(64))
            self.bns_1.append(nn.BatchNorm1d(128))
            self.bns_1.append(nn.BatchNorm1d(64))
            for layer in range(self.args.num_layer):
                self.nets_1.append(GCNConv(64, 64))
                self.bns_1.append(nn.BatchNorm1d(64))
            self.nets_1.append(GCNConv(64, 32))

            self.ESG_weight = torch.nn.Parameter(torch.ones([18, 4]))
            self.ESG_bias = torch.nn.Parameter(torch.ones(4))
        self.out = torch.nn.Parameter(torch.ones([32, 1]))
        self.apply(init_weights)

    def forward(self, data, edge_label_index):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_ESG = torch.where(x[:, 4:13] <= 0, 1e-10, x[:, 4:13])
        x_esg = torch.where(x[:, 13:22] <= 0, 1e-10, x[:, 13:22])
        x_features = x[:, 0:4]
        reverse_edge_index = torch.cat([edge_index[1].unsqueeze(0), edge_index[0].unsqueeze(0)], dim=0)

        if self.args.ss:
            x_q11 = self.beta_q11(x_esg)
            x_q12 = self.beta_q12(torch.mul(x_ESG, x_esg))
            x_q1 = self.beta_q10 + x_q11 + x_q12

            x_q21 = self.beta_q21(x_esg)
            x_q22 = self.beta_q22(torch.mul(x_ESG, x_esg))
            x_q2 = self.beta_q20 + x_q21 + x_q22

            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)
        elif self.args.ca:
            x_q11 = self.beta_q11(x_esg)
            x_q12 = self.beta_q12(torch.mul(x_ESG, x_esg))
            x_q1 = self.beta_q10 + x_q11 + x_q12

            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)




        if self.args.ss:

            attn_weight1 = torch.mm(x_q1, x_k.T) / math.sqrt(x_q1.size(-1))
            attn_weight1 = softmax_one(attn_weight1, dim=1)

            attn_weight2 = torch.mm(x_q2, x_k.T) / math.sqrt(x_q2.size(-1))
            attn_weight2 = softmax_one(attn_weight2, dim=1)

            x1 = torch.mm(attn_weight1, x_v)
            x2 = torch.mm(attn_weight2, x_v)

            x1 = x1 + x_features
            x2 = x2 + x_features

            res = []

            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index, edge_attr)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))


            x1 = self.nets_1[-1](x1, edge_index, edge_attr)
            x1 = softmax_one(x1)

            res = []
            for layer in range(len(self.bns_1)):
                x2 = self.nets_2[layer](x2, reverse_edge_index, edge_attr)
                if layer >= 2:
                    res.append(x2)
                if layer >= 3:
                    for r in res[:-1]:
                        x2 = x2 + r
                x2 = softmax_one(self.bns_2[layer](x2))

            x2 = self.nets_2[-1](x2, edge_index, edge_attr)
            x2 = softmax_one(x2)

            _x1 = x1 / torch.norm(x1, dim=-1, keepdim=True)
            _x2 = x2 / torch.norm(x2, dim=-1, keepdim=True)
            similarity = torch.mm(_x1, _x2.T)
            out = similarity[edge_label_index[1], edge_label_index[0]]

            loss = torch.mm(x1, self.loss_w.T)
            loss = torch.mm(loss, self.loss_w)
            loss = torch.mm(loss, x2.T)
            return out, loss
        elif self.args.ca:
            attn_weight1 = torch.mm(x_q1, x_k.T) / math.sqrt(x_q1.size(-1))
            attn_weight1 = softmax_one(attn_weight1, dim=1)
            x1 = torch.mm(attn_weight1, x_v)
            x1 = x1 + x_features
            res = []

            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index, edge_attr)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))

            x1 = self.nets_1[-1](x1, edge_index, edge_attr)
            x1 = softmax_one(x1)
            x = self.bn1(x1)
            out = torch.mm(x, self.out)
            out = out.squeeze(-1)
            return out
        else:
            x1 = torch.mm(x[:, 4:22], self.ESG_weight) + self.ESG_bias
            x1 = x1 + x_features
            res = []
            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index, edge_attr)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))
            x1 = self.nets_1[-1](x1, edge_index, edge_attr)
            x1 = softmax_one(x1)
            x = self.bn1(x1)
            out = torch.mm(x, self.out)
            out = out.squeeze(-1)
            return out


class GATNet(torch.nn.Module):
    def __init__(self, args):
        super(GATNet, self).__init__()
        self.args = args

        if self.args.ss:

            self.beta_q10 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q11 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q12 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.beta_q20 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q21 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q22 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)

            self.bn = nn.BatchNorm1d(self.args.model_dim)
        elif self.args.ca:
            self.beta_q10 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q11 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q12 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)

            self.bn = nn.BatchNorm1d(self.args.model_dim)

        self.bn1 = nn.BatchNorm1d(32)


        if self.args.ss:
            self.nets_1 = torch.nn.ModuleList()
            self.nets_2 = torch.nn.ModuleList()
            self.bns_1 = torch.nn.ModuleList()
            self.bns_2 = torch.nn.ModuleList()

            self.nets_1.append(GATConv(self.args.model_dim, 64 // args.num_heads, heads=args.num_heads, concat=True))
            self.nets_1.append(GATConv(64, 128 // args.num_heads, heads=args.num_heads, concat=True))
            self.nets_1.append(GATConv(128, 64 // args.num_heads, heads=args.num_heads, concat=True))

            self.nets_2.append(GATConv(self.args.model_dim, 64 // args.num_heads, heads=args.num_heads, concat=True))
            self.nets_2.append(GATConv(64, 128 // args.num_heads, heads=args.num_heads, concat=True))
            self.nets_2.append(GATConv(128, 64 // args.num_heads, heads=args.num_heads, concat=True))

            self.bns_1.append(nn.BatchNorm1d(64))
            self.bns_1.append(nn.BatchNorm1d(128))
            self.bns_1.append(nn.BatchNorm1d(64))

            self.bns_2.append(nn.BatchNorm1d(64))
            self.bns_2.append(nn.BatchNorm1d(128))
            self.bns_2.append(nn.BatchNorm1d(64))

            for layer in range(self.args.num_layer):
                self.nets_1.append(GATConv(64, 64 // args.num_heads, heads=args.num_heads, concat=True))
                self.nets_2.append(GATConv(64, 64 // args.num_heads, heads=args.num_heads, concat=True))
                self.bns_1.append(nn.BatchNorm1d(64))
                self.bns_2.append(nn.BatchNorm1d(64))

            self.nets_1.append(GATConv(64, 32 // args.num_heads, heads=args.num_heads, concat=True))

            self.nets_2.append(GATConv(64, 32 // args.num_heads, heads=args.num_heads, concat=True))

            self.loss_w = torch.nn.Parameter(torch.ones([64, 32]))
        else:
            self.nets_1 = torch.nn.ModuleList()
            self.bns_1 = torch.nn.ModuleList()
            self.nets_1.append(GATConv(self.args.model_dim, 64 // args.num_heads, heads=args.num_heads, concat=True))
            self.nets_1.append(GATConv(64, 128 // args.num_heads, heads=args.num_heads, concat=True))
            self.nets_1.append(GATConv(128, 64 // args.num_heads, heads=args.num_heads, concat=True))
            self.bns_1.append(nn.BatchNorm1d(64))
            self.bns_1.append(nn.BatchNorm1d(128))
            self.bns_1.append(nn.BatchNorm1d(64))
            for layer in range(self.args.num_layer):
                self.nets_1.append(GATConv(64, 64 // args.num_heads, heads=args.num_heads, concat=True))
                self.bns_1.append(nn.BatchNorm1d(64))
            self.nets_1.append(GATConv(64, 32 // args.num_heads, heads=args.num_heads, concat=True))

            self.ESG_weight = torch.nn.Parameter(torch.ones([18, 4]))
            self.ESG_bias = torch.nn.Parameter(torch.ones(4))
        self.out = torch.nn.Parameter(torch.ones([32, 1]))


        self.apply(init_weights)

    def forward(self, data, edge_label_index):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_ESG = torch.where(x[:, 4:13] <=0, 1e-10, x[:, 4:13])
        x_esg = torch.where(x[:, 13:22] <=0, 1e-10, x[:, 13:22])
        x_features = x[:, 0:4]
        reverse_edge_index = torch.cat([edge_index[1].unsqueeze(0), edge_index[0].unsqueeze(0)], dim=0)
        if self.args.ss:
            x_q11 = self.beta_q11(x_esg)
            x_q12 = self.beta_q12(torch.mul(x_ESG, x_esg))
            x_q1 = self.beta_q10 + x_q11 + x_q12

            x_q21 = self.beta_q21(x_esg)
            x_q22 = self.beta_q22(torch.mul(x_ESG, x_esg))
            x_q2 = self.beta_q20 + x_q21 + x_q22

            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)
        elif self.args.ca:
            x_q11 = self.beta_q11(x_esg)
            x_q12 = self.beta_q12(torch.mul(x_ESG, x_esg))
            x_q1 = self.beta_q10 + x_q11 + x_q12

            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)



        if self.args.ss:

            attn_weight1 = torch.mm(x_q1, x_k.T) / math.sqrt(x_q1.size(-1))
            attn_weight1 = softmax_one(attn_weight1, dim=1)

            attn_weight2 = torch.mm(x_q2, x_k.T) / math.sqrt(x_q2.size(-1))
            attn_weight2 = softmax_one(attn_weight2, dim=1)

            x1 = torch.mm(attn_weight1, x_v)
            x2 = torch.mm(attn_weight2, x_v)

            x1 = x1 + x_features
            x2 = x2 + x_features

            res = []

            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))

            x1 = self.nets_1[-1](x1, edge_index)
            x1 = softmax_one(x1)

            res = []
            for layer in range(len(self.bns_1)):
                x2 = self.nets_2[layer](x2, reverse_edge_index)
                if layer >= 2:
                    res.append(x2)
                if layer >= 3:
                    for r in res[:-1]:
                        x2 = x2 + r
                x2 = softmax_one(self.bns_2[layer](x2))

            x2 = self.nets_2[-1](x2, edge_index)
            x2 = softmax_one(x2)

            _x1 = x1 / torch.norm(x1, dim=-1, keepdim=True)
            _x2 = x2 / torch.norm(x2, dim=-1, keepdim=True)
            similarity = torch.mm(_x1, _x2.T)
            out = similarity[edge_label_index[1], edge_label_index[0]]

            loss = torch.mm(x1, self.loss_w.T)
            loss = torch.mm(loss, self.loss_w)
            loss = torch.mm(loss, x2.T)
            return out, loss
        elif self.args.ca:
            attn_weight1 = torch.mm(x_q1, x_k.T) / math.sqrt(x_q1.size(-1))
            attn_weight1 = softmax_one(attn_weight1, dim=1)
            x1 = torch.mm(attn_weight1, x_v)
            x1 = x1 + x_features
            res = []

            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))

            x1 = self.nets_1[-1](x1, edge_index)
            x1 = softmax_one(x1)
            x = self.bn1(x1)
            out = torch.mm(x, self.out)
            out = out.squeeze(-1)
            return out
        else:
            x1 = torch.mm(x[:, 4:22], self.ESG_weight) + self.ESG_bias
            x1 = x1 + x_features
            res = []

            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))

            x1 = self.nets_1[-1](x1, edge_index)
            x1 = softmax_one(x1)
            x = self.bn1(x1)
            out = torch.mm(x, self.out)
            out = out.squeeze(-1)
            return out


class ChebNet(torch.nn.Module):
    def __init__(self, args):
        super(ChebNet, self).__init__()
        self.args = args
        if self.args.ss:

            self.beta_q10 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q11 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q12 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.beta_q20 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q21 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q22 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)

            self.bn = nn.BatchNorm1d(self.args.model_dim)
        elif self.args.ca:
            self.beta_q10 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q11 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q12 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)

            self.bn = nn.BatchNorm1d(self.args.model_dim)

        self.bn1 = nn.BatchNorm1d(32)


        if self.args.ss:
            self.nets_1 = torch.nn.ModuleList()
            self.nets_2 = torch.nn.ModuleList()
            self.bns_1 = torch.nn.ModuleList()
            self.bns_2 = torch.nn.ModuleList()

            self.nets_1.append(ChebConv(self.args.model_dim, 64, K=3))
            self.nets_1.append(ChebConv(64, 128, K=3))
            self.nets_1.append(ChebConv(128, 64, K=3))

            self.nets_2.append(ChebConv(self.args.model_dim, 64, K=3))
            self.nets_2.append(ChebConv(64, 128, K=3))
            self.nets_2.append(ChebConv(128, 64, K=3))

            self.bns_1.append(nn.BatchNorm1d(64))
            self.bns_1.append(nn.BatchNorm1d(128))
            self.bns_1.append(nn.BatchNorm1d(64))

            self.bns_2.append(nn.BatchNorm1d(64))
            self.bns_2.append(nn.BatchNorm1d(128))
            self.bns_2.append(nn.BatchNorm1d(64))

            for layer in range(self.args.num_layer):
                self.nets_1.append(ChebConv(64, 64, K=3))
                self.nets_2.append(ChebConv(64, 64, K=3))
                self.bns_1.append(nn.BatchNorm1d(64))
                self.bns_2.append(nn.BatchNorm1d(64))

            self.nets_1.append(ChebConv(64, 32, K=3))

            self.nets_2.append(ChebConv(64, 32, K=3))

            self.loss_w = torch.nn.Parameter(torch.ones([64, 32]))

        else:
            self.nets_1 = torch.nn.ModuleList()
            self.bns_1 = torch.nn.ModuleList()
            self.nets_1.append(ChebConv(self.args.model_dim, 64, K=3))
            self.nets_1.append(ChebConv(64, 128, K=3))
            self.nets_1.append(ChebConv(128, 64, K=3))
            self.bns_1.append(nn.BatchNorm1d(64))
            self.bns_1.append(nn.BatchNorm1d(128))
            self.bns_1.append(nn.BatchNorm1d(64))
            for layer in range(self.args.num_layer):
                self.nets_1.append(ChebConv(64, 64, K=3))
                self.bns_1.append(nn.BatchNorm1d(64))
            self.nets_1.append(ChebConv(64, 32, K=3))
            self.ESG_weight = torch.nn.Parameter(torch.ones([18, 4]))
            self.ESG_bias = torch.nn.Parameter(torch.ones(4))
        self.out = torch.nn.Parameter(torch.ones([32, 1]))

        self.apply(init_weights)

    def forward(self, data, edge_label_index):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_ESG = torch.where(x[:, 4:13] <=0, 1e-10, x[:, 4:13])
        x_esg = torch.where(x[:, 13:22] <=0, 1e-10, x[:, 13:22])
        x_features = x[:, 0:4]
        reverse_edge_index = torch.cat([edge_index[1].unsqueeze(0), edge_index[0].unsqueeze(0)], dim=0)
        if self.args.ss:
            x_q11 = self.beta_q11(x_esg)
            x_q12 = self.beta_q12(torch.mul(x_ESG, x_esg))
            x_q1 = self.beta_q10 + x_q11 + x_q12

            x_q21 = self.beta_q21(x_esg)
            x_q22 = self.beta_q22(torch.mul(x_ESG, x_esg))
            x_q2 = self.beta_q20 + x_q21 + x_q22

            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)
        elif self.args.ca:
            x_q11 = self.beta_q11(x_esg)
            x_q12 = self.beta_q12(torch.mul(x_ESG, x_esg))
            x_q1 = self.beta_q10 + x_q11 + x_q12

            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)



        if self.args.ss:

            attn_weight1 = torch.mm(x_q1, x_k.T) / math.sqrt(x_q1.size(-1))
            attn_weight1 = softmax_one(attn_weight1, dim=1)

            attn_weight2 = torch.mm(x_q2, x_k.T) / math.sqrt(x_q2.size(-1))
            attn_weight2 = softmax_one(attn_weight2, dim=1)

            x1 = torch.mm(attn_weight1, x_v)
            x2 = torch.mm(attn_weight2, x_v)

            x1 = x1 + x_features
            x2 = x2 + x_features

            res = []

            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index, edge_attr)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))

            x1 = self.nets_1[-1](x1, edge_index, edge_attr)
            x1 = softmax_one(x1)

            res = []
            for layer in range(len(self.bns_1)):
                x2 = self.nets_2[layer](x2, reverse_edge_index, edge_attr)
                if layer >= 2:
                    res.append(x2)
                if layer >= 3:
                    for r in res[:-1]:
                        x2 = x2 + r
                x2 = softmax_one(self.bns_2[layer](x2))

            x2 = self.nets_2[-1](x2, edge_index, edge_attr)
            x2 = softmax_one(x2)

            _x1 = x1 / torch.norm(x1, dim=-1, keepdim=True)
            _x2 = x2 / torch.norm(x2, dim=-1, keepdim=True)
            similarity = torch.mm(_x1, _x2.T)
            out = similarity[edge_label_index[1], edge_label_index[0]]

            loss = torch.mm(x1, self.loss_w.T)
            loss = torch.mm(loss, self.loss_w)
            loss = torch.mm(loss, x2.T)
            return out, loss
        elif self.args.ca:
            attn_weight1 = torch.mm(x_q1, x_k.T) / math.sqrt(x_q1.size(-1))
            attn_weight1 = softmax_one(attn_weight1, dim=1)
            x1 = torch.mm(attn_weight1, x_v)
            x1 = x1 + x_features
            res = []

            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index, edge_attr)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))

            x1 = self.nets_1[-1](x1, edge_index, edge_attr)
            x1 = softmax_one(x1)
            x = self.bn1(x1)
            out = torch.mm(x, self.out)
            out = out.squeeze(-1)
            return out

        else:
            x1 = torch.mm(x[:, 4:22], self.ESG_weight) + self.ESG_bias
            x1 = x1 + x_features
            res = []

            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index, edge_attr)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))

            x1 = self.nets_1[-1](x1, edge_index, edge_attr)
            x1 = softmax_one(x1)
            x = self.bn1(x1)
            out = torch.mm(x, self.out)
            out = out.squeeze(-1)
            return out



class SageNet(torch.nn.Module):
    def __init__(self, args):
        super(SageNet, self).__init__()
        self.args = args
        if self.args.ss:

            self.beta_q10 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q11 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q12 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.beta_q20 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q21 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q22 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)

            self.bn = nn.BatchNorm1d(self.args.model_dim)

        elif self.args.ca:
            self.beta_q10 = torch.nn.Parameter(torch.zeros([1]))
            self.beta_q11 = torch.nn.Linear(9, self.args.model_dim, bias=False)
            self.beta_q12 = torch.nn.Linear(9, self.args.model_dim, bias=False)

            self.w_k = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)
            self.w_v = torch.nn.Linear(self.args.model_dim, self.args.model_dim, bias=False)

            self.bn = nn.BatchNorm1d(self.args.model_dim)
        self.bn1 = nn.BatchNorm1d(32)

        if self.args.ss:
            self.nets_1 = torch.nn.ModuleList()
            self.nets_2 = torch.nn.ModuleList()
            self.bns_1 = torch.nn.ModuleList()
            self.bns_2 = torch.nn.ModuleList()

            self.nets_1.append(SAGEConv(self.args.model_dim, 64))
            self.nets_1.append(SAGEConv(64, 128))
            self.nets_1.append(SAGEConv(128, 64))

            self.nets_2.append(SAGEConv(self.args.model_dim, 64))
            self.nets_2.append(SAGEConv(64, 128))
            self.nets_2.append(SAGEConv(128, 64))

            self.bns_1.append(nn.BatchNorm1d(64))
            self.bns_1.append(nn.BatchNorm1d(128))
            self.bns_1.append(nn.BatchNorm1d(64))

            self.bns_2.append(nn.BatchNorm1d(64))
            self.bns_2.append(nn.BatchNorm1d(128))
            self.bns_2.append(nn.BatchNorm1d(64))

            for layer in range(self.args.num_layer):
                self.nets_1.append(SAGEConv(64, 64))
                self.nets_2.append(SAGEConv(64, 64))
                self.bns_1.append(nn.BatchNorm1d(64))
                self.bns_2.append(nn.BatchNorm1d(64))

            self.nets_1.append(SAGEConv(64, 32))

            self.nets_2.append(SAGEConv(64, 32))

            self.loss_w = torch.nn.Parameter(torch.ones([64, 32]))

        else:
            self.nets_1 = torch.nn.ModuleList()
            self.bns_1 = torch.nn.ModuleList()
            self.nets_1.append(SAGEConv(self.args.model_dim, 64))
            self.nets_1.append(SAGEConv(64, 128))
            self.nets_1.append(SAGEConv(128, 64))
            self.bns_1.append(nn.BatchNorm1d(64))
            self.bns_1.append(nn.BatchNorm1d(128))
            self.bns_1.append(nn.BatchNorm1d(64))
            for layer in range(self.args.num_layer):
                self.nets_1.append(SAGEConv(64, 64))
                self.bns_1.append(nn.BatchNorm1d(64))
            self.nets_1.append(SAGEConv(64, 32))
            self.ESG_weight = torch.nn.Parameter(torch.ones([18, 4]))
            self.ESG_bias = torch.nn.Parameter(torch.ones(4))
        self.out = torch.nn.Parameter(torch.ones([32, 1]))


        self.apply(init_weights)

    def forward(self, data, edge_label_index):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x_ESG = torch.where(x[:, 4:13] <= 0, 1e-10, x[:, 4:13])
        x_esg = torch.where(x[:, 13:22] <= 0, 1e-10, x[:, 13:22])
        x_features = x[:, 0:4]
        reverse_edge_index = torch.cat([edge_index[1].unsqueeze(0), edge_index[0].unsqueeze(0)], dim=0)
        if self.args.ss:
            x_q11 = self.beta_q11(x_esg)
            x_q12 = self.beta_q12(torch.mul(x_ESG, x_esg))
            x_q1 = self.beta_q10 + x_q11 + x_q12

            x_q21 = self.beta_q21(x_esg)
            x_q22 = self.beta_q22(torch.mul(x_ESG, x_esg))
            x_q2 = self.beta_q20 + x_q21 + x_q22

            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)
        elif self.args.ca:
            x_q11 = self.beta_q11(x_esg)
            x_q12 = self.beta_q12(torch.mul(x_ESG, x_esg))
            x_q1 = self.beta_q10 + x_q11 + x_q12

            x_features = self.bn(x_features)

            x_k = self.w_k(x_features)
            x_v = self.w_v(x_features)


        if self.args.ss:

            attn_weight1 = torch.mm(x_q1, x_k.T) / math.sqrt(x_q1.size(-1))
            attn_weight1 = softmax_one(attn_weight1, dim=1)

            attn_weight2 = torch.mm(x_q2, x_k.T) / math.sqrt(x_q2.size(-1))
            attn_weight2 = softmax_one(attn_weight2, dim=1)

            x1 = torch.mm(attn_weight1, x_v)
            x2 = torch.mm(attn_weight2, x_v)

            x1 = x1 + x_features
            x2 = x2 + x_features

            res = []

            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))

            x1 = self.nets_1[-1](x1, edge_index)
            x1 = softmax_one(x1)

            res = []
            for layer in range(len(self.bns_1)):
                x2 = self.nets_2[layer](x2, reverse_edge_index)
                if layer >= 2:
                    res.append(x2)
                if layer >= 3:
                    for r in res[:-1]:
                        x2 = x2 + r
                x2 = softmax_one(self.bns_2[layer](x2))

            x2 = self.nets_2[-1](x2, edge_index)
            x2 = softmax_one(x2)

            _x1 = x1 / torch.norm(x1, dim=-1, keepdim=True)
            _x2 = x2 / torch.norm(x2, dim=-1, keepdim=True)
            similarity = torch.mm(_x1, _x2.T)
            out = similarity[edge_label_index[1], edge_label_index[0]]

            loss = torch.mm(x1, self.loss_w.T)
            loss = torch.mm(loss, self.loss_w)
            loss = torch.mm(loss, x2.T)
            return out, loss
        elif self.args.ca:
            attn_weight1 = torch.mm(x_q1, x_k.T) / math.sqrt(x_q1.size(-1))
            attn_weight1 = softmax_one(attn_weight1, dim=1)
            x1 = torch.mm(attn_weight1, x_v)
            x1 = x1 + x_features
            res = []

            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))

            x1 = self.nets_1[-1](x1, edge_index)
            x1 = softmax_one(x1)
            x = self.bn1(x1)
            out = torch.mm(x, self.out)
            out = out.squeeze(-1)
            return out
        else:
            x1 = torch.mm(x[:, 4:22], self.ESG_weight) + self.ESG_bias
            x1 = x1 + x_features
            res = []

            for layer in range(len(self.bns_1)):
                x1 = self.nets_1[layer](x1, edge_index)
                if layer >= 2:
                    res.append(x1)
                if layer >= 3:
                    for r in res[:-1]:
                        x1 = x1 + r
                x1 = softmax_one(self.bns_1[layer](x1))

            x1 = self.nets_1[-1](x1, edge_index)
            x1 = softmax_one(x1)
            x = self.bn1(x1)
            out = torch.mm(x, self.out)
            out = out.squeeze(-1)
            return out