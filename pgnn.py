"""
Add node to node distance in graphs
"""

import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import os


from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from model import *
from utils import *

from tensorboardX import SummaryWriter

log_dir = os.path.join('./log', "RUN_" + str(3))
writer = SummaryWriter(log_dir=log_dir)

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type, args):
    model.train()

    train_loss = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
        #     pass
        # else:
        # Compute distance
        graph_idx, batch = batch
        # Find dists using graph_idx
        # if batch.num_nodes == 3:
        #     print(1)
        graph_idx = loader.dataset.indices()[graph_idx.item()]
        dists = precompute_distance[graph_idx]
        batch.dists = torch.from_numpy(dists).float()
        preselect_anchor(batch, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')

        batch = batch.to(device)
        pred = model(batch)
        optimizer.zero_grad()
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y
        if "classification" in task_type: 
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        else:
            loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        train_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return train_loss / (step + 1)
  

def eval(model, device, loader, evaluator, args):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # dists = precompute_dist_data(batch.edge_index.numpy(), batch.num_nodes, approximate=-1)
        # batch.dists = torch.from_numpy(dists).float()
        # preselect_anchor(batch, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
        graph_idx, batch = batch
        # Find dists using graph_idx
        graph_idx = loader.dataset.indices()[graph_idx.item()]
        dists = precompute_distance[graph_idx]
        batch.dists = torch.from_numpy(dists).float()
        preselect_anchor(batch, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)
    
parser = argparse.ArgumentParser()
# GNN settings
parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
parser.add_argument('--use_val_as_input', action='store_true')
parser.add_argument('--graph_pooling', dest='graph_pooling', default='mean', type=str)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', dest='epochs', default=400, type=int) # implemented via accumulating gradient
parser.add_argument('--batch_size', dest='batch_size', default=1, type=int) # implemented via accumulating gradient
parser.add_argument('--layer_num', dest='layer_num', default=2, type=int)
parser.add_argument('--feature_dim', dest='feature_dim', default=64, type=int)
parser.add_argument('--hidden_dim', dest='hidden_dim', default=64, type=int)
parser.add_argument('--emb_dim', dest='emb_dim', default=64, type=int)
parser.add_argument('--output_dim', dest='output_dim', default=64, type=int)
parser.add_argument('--anchor_num', dest='anchor_num', default=8, type=int)
parser.add_argument('--dropout', dest='dropout', action='store_true',
                    help='whether dropout, default 0.5')
parser.add_argument('--dropout_no', dest='dropout', action='store_false',
                    help='whether dropout, default 0.5')
parser.add_argument('--feature_pre', dest='feature_pre', action='store_true',
                        help='whether pre transform feature')
parser.add_argument('--filename', type=str, default="modelsave",
                        help='filename to output result (default: )')
parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')

parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args(args=[])
# args = parser.parse_args()

### automatic dataloading and splitting
dataset = PygGraphPropPredDataset(name ='ogbg-molhiv')

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

if not os.path.exists("dists.npy"):
    precompute_distance = {}
    for i, (_, data) in enumerate(tqdm(dataset, desc="Create distance")):
        dists = precompute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=-1)
        # dists = torch.from_numpy(dists).float() 
        precompute_distance[i] = dists 
else:
    # np.save('dists.npy', precompute_distance)
    precompute_distance = np.load('dists.npy', allow_pickle=True)

precompute_distance = precompute_distance.item()

# Save dataset object
if args.feature == 'full':
    pass 
elif args.feature == 'simple':
    print('using simple feature')
    # only retain the top two node/edge features
    dataset.data.x = dataset.data.x[:,:2]
    dataset.data.edge_attr = dataset.data.edge_attr[:,:2]
    

split_idx = dataset.get_idx_split()
evaluator = Evaluator(name ='ogbg-molhiv')

# TODO: Precompute distance and map each graph index
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

num_features = 9 # dataset[0][1].x.shape[1] # use dataset.idx
# model = PGNN(input_dim=num_features, feature_dim=args.feature_dim,
#             hidden_dim=args.hidden_dim, output_dim=args.output_dim,
#             feature_pre=args.feature_pre, layer_num=args.layer_num, 
#             dropout=args.dropout, graph_pooling=args.graph_pooling).to(device)

model = PGNN_node(input_dim=num_features, feature_dim=args.emb_dim,
            hidden_dim=args.emb_dim, output_dim=args.emb_dim,
            feature_pre=args.feature_pre, layer_num=args.layer_num, 
            dropout=args.dropout, graph_pooling=args.graph_pooling, emb_dim=args.emb_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)


valid_curve = []
test_curve = []
train_curve = []

print("Start")
for epoch in range(1, args.epochs + 1):
    print("=====Epoch {}".format(epoch))
    print('Training...')
    loss = train(model, device, train_loader, optimizer, dataset.task_type, args)

    print('Evaluating...')
    train_perf = eval(model, device, train_loader, evaluator, args)
    valid_perf = eval(model, device, valid_loader, evaluator, args)
    test_perf = eval(model, device, test_loader, evaluator, args)

    print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

    train_curve.append(train_perf[dataset.eval_metric])
    valid_curve.append(valid_perf[dataset.eval_metric])
    test_curve.append(test_perf[dataset.eval_metric])


    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    writer.add_scalar('train/_loss', loss, epoch)
    
    writer.add_scalar('train/_perf', train_perf['rocauc'], epoch)

    writer.add_scalar('val/_perf', valid_perf['rocauc'], epoch)
    
    writer.add_scalar('test/_perf', test_perf['rocauc'], epoch)
    
    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch) 


    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

writer.close()
if not args.filename == '':
    torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)



