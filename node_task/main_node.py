import os

import numpy as np
import random
import torch
import pickle

from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from sklearn.metrics import r2_score
from model_node_ss_ca import GCNNet, GATNet, ChebNet, SageNet
import argparse
import warnings
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit


warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='ESG_pre')
parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
parser.add_argument('--optimizer', type=str, default='adam', help="optimizer")
parser.add_argument('--model_dim', type=int, default=4, help="hidden dim should match with x_feature dim")
parser.add_argument('--model', type=str, default='sage', choices=['gcn', 'gat', 'sage', 'cheb'])
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--task', type=str, default='NPR', choices=['NPR', 'OGR'])
parser.add_argument('--lr', type=float, default=5e-2, help="learning rate")
parser.add_argument('--iteration', type=int, default=8000, help="iteration")
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
# torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

if args.task == 'NPR':
    pkl_file = open('data/2018_data_NPR.pkl', 'rb')
    data_task = pickle.load(pkl_file)
    data_task = Data(x=data_task[0], y=data_task[1], edge_index=data_task[2])

elif args.task == 'OGR':
    pkl_file = open('data/2018_data_OGR.pkl', 'rb')
    data_task = pickle.load(pkl_file)
    data_task = Data(x=data_task[0], y=data_task[1], edge_index=data_task[2])
else:
    raise SystemExit




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
if args.ss:
    model_name = args.model + "_ss"
    suffix = "_ss"
elif args.ca:
    model_name = args.model + "_ca"
    suffix = "_ca"
else:
    model_name = args.model
    suffix = ""

dirs = "./data/{}/{}/{}".format(args.task, model_name, args.seed)
if not os.path.exists(dirs):
    os.makedirs(dirs)

log_file = open(
    './data/{}/{}/{}/log_{}_{}.txt'.format(args.task, model_name, args.seed, model_name,
                                           args.seed), 'a')

writer = SummaryWriter('./boardlog')


device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
loss_func = torch.nn.MSELoss().to(device)
transform = RandomNodeSplit(num_val=0.1, num_test=0.2, split="train_rest")
data_task = transform(data_task)



model.to(device)
data_task.to(device)
edge_index = data_task.edge_index.cpu()
adj = sp.coo_matrix((torch.ones(edge_index.shape[1]), (edge_index[0, :], edge_index[1, :])),
                        shape=(data_task.y.shape[0], data_task.y.shape[0]),
                        dtype=np.float32).todense()
adj = torch.tensor(adj).to(device)


def write_numpy_to_txt(input_numpy, input_name, txt_path="./text.txt"):
    txt_file = open(txt_path, "a")
    txt_file.write(input_name + "\n")
    if len(input_numpy.shape) == 1:
        for i in range(len(input_numpy)):
            txt_file.write("{:4f}    ".format(input_numpy[i]))
        txt_file.write("\n")
    else:
        for i in range(len(input_numpy)):
            for j in input_numpy[i]:
                txt_file.write("{:4f}    ".format(j))
            txt_file.write("\n")
        txt_file.write("\n")
    txt_file.flush()



def Evaluation(target, pre):
    mae = np.mean(np.abs(target - pre))
    mape = np.mean(np.abs((target - pre) / target))
    rmse = np.sqrt(np.mean(np.power(target - pre, 2)))
    r2 = r2_score(target, pre)

    return mae, mape, rmse, r2


if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=1e-3)
elif args.optimizer == 'Nadam':
    optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, weight_decay=1e-4)

def train_epoch(adj):
    model.train()
    optimizer.zero_grad()
    if args.ss:
        prediction, self_loss = model(data_task)
        likelihood = - torch.mul(self_loss[data_task.train_mask, :][:, data_task.train_mask], adj[data_task.train_mask, :][:, data_task.train_mask]) + torch.log(1 + torch.exp(self_loss[data_task.train_mask, :][:, data_task.train_mask]))
        self_loss_2 = torch.mean(likelihood)
        loss = torch.sqrt(loss_func(prediction[data_task.train_mask], data_task.y[data_task.train_mask]).float()) + self_loss_2 * args.lam


    else:
        prediction = model(data_task)
        loss = torch.sqrt(
            loss_func(prediction[data_task.train_mask], data_task.y[data_task.train_mask]).float())
    loss.backward()

    optimizer.step()
    if args.ss:
        return loss.item(), self_loss_2
    return loss.item()

def validate(adj):
    global model
    target = data_task.y[data_task.val_mask].reshape(-1).detach().cpu().numpy()
    with torch.no_grad():
        model.eval()
        if args.ss:
            prediction, self_loss = model(data_task)
            likelihood = - torch.mul(self_loss[data_task.val_mask, :][:, data_task.val_mask],
                                     adj[data_task.val_mask, :][:, data_task.val_mask]) + torch.log(
                1 + torch.exp(self_loss[data_task.val_mask, :][:, data_task.val_mask]))
            self_loss_2 = torch.mean(likelihood)
            loss = torch.sqrt(
                loss_func(prediction[data_task.val_mask], data_task.y[data_task.val_mask]).float()) + self_loss_2 * args.lam

        else:
            prediction = model(data_task)
            loss = torch.sqrt(
                loss_func(prediction[data_task.val_mask], data_task.y[data_task.val_mask]).float())
        pre = prediction[data_task.val_mask].reshape(-1).detach().cpu().numpy()
        mae, mape, rmse, r2 = Evaluation(target, pre)

        return mae, mape, rmse, r2, loss


def test_model(adj):
    global model
    target = data_task.y[data_task.test_mask].reshape(-1).detach().cpu().numpy()
    with torch.no_grad():
        model.eval()
        if args.ss:
            prediction, self_loss = model(data_task)
            likelihood = - torch.mul(self_loss[data_task.test_mask, :][:, data_task.test_mask],
                                     adj[data_task.test_mask, :][:, data_task.test_mask]) + torch.log(
                1 + torch.exp(self_loss[data_task.test_mask, :][:, data_task.test_mask]))
            self_loss_2 = torch.mean(likelihood)
            loss = torch.sqrt(
                loss_func(prediction[data_task.test_mask], data_task.y[data_task.test_mask]).float()) + self_loss_2 * args.lam

        else:
            prediction = model(data_task)
            loss = torch.sqrt(
                loss_func(prediction[data_task.test_mask], data_task.y[data_task.test_mask]).float())
        pre = prediction[data_task.test_mask].reshape(-1).detach().cpu().numpy()
        mae, mape, rmse, r2 = Evaluation(target, pre)



        return mae, mape, rmse, r2, loss

def print_write_test(epoch, mae, mape, rmse, r2, loss):

    print(
        "Epoch:{}  test_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}  r2:{:.4f}".format(epoch,
                                                                                            loss.item(),
                                                                                            rmse,
                                                                                            mae, mape, r2))
    log_file.write(
        "Epoch:{}  test_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}  r2:{:.4f}".format(epoch,
                                                                                            loss.item(),
                                                                                            rmse,
                                                                                            mae, mape,
                                                                                            r2) + "\n")
    log_file.flush()



def print_write(epoch, mae, mape, rmse, r2, loss):

    print(
        "Epoch:{}  val_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}  r2:{:.4f}".format(epoch,
                                                                                            loss.item(),
                                                                                            rmse,
                                                                                            mae, mape, r2))
    log_file.write(
        "Epoch:{}  val_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}  r2:{:.4f}".format(epoch,
                                                                                            loss.item(),
                                                                                            rmse,
                                                                                            mae, mape,
                                                                                            r2) + "\n")
    log_file.flush()

def print_write_train(epoch, mae, mape, rmse, r2, loss):

    print(
        "Epoch:{}  train_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}  r2:{:.4f}".format(epoch,
                                                                                            loss.item(),
                                                                                            rmse,
                                                                                            mae, mape, r2))
    log_file.write(
        "Epoch:{}  train_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}  r2:{:.4f}".format(epoch,
                                                                                            loss.item(),
                                                                                            rmse,
                                                                                            mae, mape,
                                                                                            r2) + "\n")
    log_file.flush()

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
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model

def train(path=None):
    global optimizer
    global model
    if path:
        model = read_path_model(path)
    scheduler_lr = lr_scheduler.StepLR(optimizer, 1000, 0.5)


    for epoch in range(args.iteration + 1):
        if args.ss:
            train_loss, self_loss = train_epoch(adj)
        else:
            train_loss = train_epoch(adj)

        scheduler_lr.step()

        mae, mape, rmse, r2, loss = validate(adj)

        writer.add_scalars("Node/{}/{}/{}/val/mae/".format(args.task, args.model, str(args.seed)),
                           {"val_mae"+suffix: mae}, epoch)

        writer.add_scalars("Node/{}/{}/{}/val/mape/".format(args.task, args.model, str(args.seed)),
                           {"val_mape"+suffix: mape}, epoch)

        writer.add_scalars("Node/{}/{}/{}/val/rmse/".format(args.task, args.model, str(args.seed)),
                           {"val_rmse"+suffix: rmse}, epoch)

        writer.add_scalars("Node/{}/{}/{}/val/r2/".format(args.task, args.model, str(args.seed)),
                           {"val_r2"+suffix: r2}, epoch)

        if args.ss:
            writer.add_scalars("Node/{}/{}/{}/train&val/Self_Loss/".format(args.task, args.model, str(args.seed)),
                               {"train_self_loss"+suffix: self_loss * args.lam, "train_loss"+suffix: train_loss}, epoch)



        print_write(epoch, mae, mape, rmse, r2, loss)





        if epoch % 100 == 0:
            torch.save(model.state_dict(), './data/{}/{}/{}/model_{}_{}_{}.pth'.format(args.task, model_name, args.seed, path.split("_")[-2] + "more_train" if path else model_name,
                                                                                    args.seed, epoch))


    torch.save(model.state_dict(), './data/{}/{}/{}/model_{}_{}_{}.pth'.format(args.task, model_name, args.seed,
                                                                               path.split("_")[-2] + "more_train" if path else model_name,
                                                                               args.seed, epoch))







def test(path):
    global model
    model = read_path_model(path)

    mae, mape, rmse, r2, loss = test_model(adj)
    log_file.write(
        "path:{}  test_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}  r2:{:.4f}".format(path, loss.item(),
                                                                                            rmse,
                                                                                            mae, mape, r2) + "\n")
    log_file.flush()

    print("test_loss:{:.4f}  RMSE:{:.4f}  MAE:{:.4f}  MAPE:{:.4f}  r2:{:.4f}".format(loss.item(), rmse, mae, mape, r2))

    return mape, r2


if __name__ == '__main__':
    train()
