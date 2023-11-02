import os

import numpy as np
import random
import torch
import pickle

from scipy.sparse import coo_array
from sklearn.metrics import f1_score, roc_auc_score, log_loss, accuracy_score, recall_score, precision_score, roc_curve
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from torch_geometric.utils import negative_sampling

from model_link import GCNNet, GATNet, ChebNet, SageNet
import argparse
import warnings
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit


writer = SummaryWriter('./boardlog')

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='ESG_pre_link')
parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
parser.add_argument('--optimizer', type=str, default='adam', help="optimizer")
parser.add_argument('--model_dim', type=int, default=4, help="hidden dim should match with x_feature dim")
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat', 'sage', 'cheb'])
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--lr', type=float, default=2e-2, help="learning rate")
parser.add_argument('--iteration', type=int, default=4000, help="iteration")
parser.add_argument('--num_heads', type=int, default=4, help="num_heads for GAT")
parser.add_argument('--num_layer', type=int, default=1, help="middle layer")
parser.add_argument('--lam', type=float, default=1, help="lam")
parser.add_argument('--ss', action='store_true', help="Self-Supervise")
parser.add_argument('--ca', action='store_true', help="Cross-Attention")
args = parser.parse_args()

SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if args.ss:
    model_name = args.model + "_ss"
    suffix = "_ss"
elif args.ca:
    model_name = args.model + "_ca"
    suffix = "_ca"
else:
    model_name = args.model
    suffix = ""



dirs = "./data/link/{}/{}".format(model_name, args.seed)
if not os.path.exists(dirs):
    os.makedirs(dirs)

log_file = open(
    './data/link/{}/{}/log_{}_{}.txt'.format(model_name, args.seed, model_name,
                                           args.seed), 'a')

pkl_file = open('./data/2018_data_NPR.pkl', 'rb')
data_task = pickle.load(pkl_file)
data_task = Data(x=data_task[0], y=data_task[1], edge_index=data_task[2])



if args.model == 'gcn':
    print('gcn')
    model = GCNNet(args)
elif args.model == 'gat':
    print('gat')
    model = GATNet(args)
elif args.model == 'cheb':
    print('cheb')
    model = ChebNet(args)
elif args.model == 'sage':
    print('sage')
    model = SageNet(args)
else:
    raise SystemExit


device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
loss_func = torch.nn.BCEWithLogitsLoss().to(device)

transform = RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=False,
                            add_negative_train_samples=False, neg_sampling_ratio=0.0)


train_data, val_data, test_data = transform(data_task)

train_data.to(device)
val_data.to(device)
test_data.to(device)
model.to(device)






edge_index = data_task.edge_index.cpu()
adj = sp.coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0, :], edge_index[1, :])),
                          shape=(train_data.y.shape[0], train_data.y.shape[0]),
                          dtype=np.float32).todense()
adj = torch.tensor(adj).to(device)


pkl_file = open('./data/2019_data_NPR.pkl', 'rb')
data_task_2019 = pickle.load(pkl_file)
data_task_2019 = Data(x=data_task_2019[0], y=data_task_2019[1], edge_index=data_task_2019[2])


edge_index = data_task_2019.edge_index.cpu()
adj_2019 = sp.coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0, :], edge_index[1, :])),
                          shape=(train_data.y.shape[0], train_data.y.shape[0]),
                          dtype=np.float32).todense()
adj_2019 = torch.tensor(adj_2019).to(device)

gap_adj = adj - adj_2019
gap_adj = torch.where(gap_adj <= -1, 1, 0)


POS_edge_index = edge_index = gap_adj.nonzero().t().contiguous().to(device)


gap_adj = adj - adj_2019
NS_edge_index = torch.where(gap_adj >= 1, 1, 0)

NS_edge_index = NS_edge_index.nonzero().t().contiguous().to(device)
NS_edge_index_label = torch.zeros(NS_edge_index.shape[1]).to(device)
edge_index = torch.cat([NS_edge_index, edge_index], dim=1)
edge_index_label = torch.cat([NS_edge_index_label, torch.ones(POS_edge_index.shape[1]).to(device)], dim=0)
train_data.edge_label = edge_index_label
train_data.edge_label_index = edge_index


def Evaluation(target, pre):
    roc_auc = roc_auc_score(target, pre)
    _, __, thresholds = roc_curve(target, pre)
    threshold = np.mean(thresholds) * 1.025
    f1 = f1_score(target, (torch.tensor(pre).ge(threshold)).int().numpy())
    logloss = log_loss(target, (torch.tensor(pre).ge(threshold)).int().numpy())
    accuracy = accuracy_score(target, (torch.tensor(pre).ge(threshold)).int().numpy())
    recall = recall_score(target, (torch.tensor(pre).ge(threshold)).int().numpy())
    precision = precision_score(target, (torch.tensor(pre).ge(threshold)).int().numpy())
    return {"roc_auc": roc_auc, "f1": f1, "accuracy": accuracy, "recall": recall, "precision": precision, "log_loss": logloss}


if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-3)
elif args.optimizer == 'Nadam':
    optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, weight_decay=1e-4)



def test_model(adj):
    global model
    with torch.no_grad():
        model.eval()
        if args.ss:
            prediction, self_loss = model(train_data, train_data.edge_label_index)
            likelihood = - torch.mul(self_loss, adj) + torch.log(
                1 + torch.exp(self_loss))
            self_loss_2 = torch.mean(likelihood)
            loss = torch.sqrt(
                loss_func(prediction, train_data.edge_label).float()) + self_loss_2
        else:
            prediction = model(train_data, train_data.edge_label_index)
            loss = torch.sqrt(
                loss_func(prediction, train_data.edge_label).float())
        pre = prediction.reshape(-1).detach().cpu().numpy()
        test_result = Evaluation(train_data.edge_label.cpu().numpy(), pre)

        print(pre)

        return test_result, loss






def read_path_model(path):
    global model
    if args.model == 'gcn':
        model = GCNNet(args)
    elif args.model == 'gat':
        model = GATNet(args)
    elif args.model == 'cheb':
        model = ChebNet(args)
    elif args.model == 'sage':
        model = SageNet(args)
    else:
        raise SystemExit
    pretrained_dict = torch.load(path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    return model



def test(path):
    global model
    model = read_path_model(path)

    test_result, loss = test_model(adj)
    print(
        "Epoch:{}  test_loss:{:.4f}  {}".format(0, loss.item(), test_result))





if __name__ == '__main__':

    test("../node_task/data/NPR/gcn_ss/42/model_gcn_ss_42_8000.pth")