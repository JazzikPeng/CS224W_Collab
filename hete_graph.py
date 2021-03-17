"""
Create a Heterogeneous graph for edges in different years.
(node, year, node), where year can be used a feature of embedding
"""

import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_scatter import scatter

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from torch_sparse import SparseTensor, matmul
from sklearn.metrics import f1_score

from torch_geometric.utils import to_networkx

from logger import Logger

from hete_models import *

# Please do not change the following parameters
args = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'hidden_size': 128,
    'weight_decay': 1e-5,
    'lr': 0.01,
    'attn_size': 32,
    'run': 2,
    'epochs': 200,
    'batch_size': 128*2048,
    'eval_steps': 1,
    'log_steps': 1
}



def train(model, predictor, hetero_graph, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(hetero_graph.node_label['author'].device)

    total_loss = total_examples = 0
    counter = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        print("Batch Number:", counter)
        counter += 1
        h = model(hetero_graph.node_feature, hetero_graph.edge_index) # encode every edges with GNN model

        edge = pos_train_edge[perm].t() # (2, B)

        pos_out = predictor(h['author'][edge[0]], h['author'][edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, hetero_graph.num_nodes()['author'], edge.size(), dtype=torch.long,
                             device=h['author'].device)
        neg_out = predictor(h['author'][edge[0]], h['author'][edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples



@torch.no_grad()
def test(model, predictor, hetero_graph, data, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(hetero_graph.node_feature, hetero_graph.edge_index) # encode every edges with GNN model
    h = h['author']

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    h = model(hetero_graph.node_feature, hetero_graph.edge_index) # encode every edges with GNN model
    h = h['author']

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

# Transfer data to heterogenous graph
dataset = PygLinkPropPredDataset(name='ogbl-collab')
data = dataset[0]

# Specify all the message types
years = torch.unique(data.edge_year).numpy().astype(int)
msg_types = []
for y in years:
    msg_type = ("author", str(y), "author")
    msg_types.append(msg_type)

# Dictionary of edge indices
edge_index = {}
for mt in msg_types:
    y = int(mt[1])
    yr_idx = (data.edge_year==y).T[0]
    edge_idx = data.edge_index.T[yr_idx]
    edge_idx = torch.cat([edge_idx[edge_idx[:,0] < edge_idx[:,1]], \
                edge_idx[edge_idx[:,0] > edge_idx[:,1]]]).T
    edge_index[mt] = edge_idx

# Dictionary of node features
node_feature = {}
node_feature["author"] = data.x

# Dictionary of edge indices, we have one type of node "author"
node_label = {}
node_label['author'] = torch.zeros(data.num_nodes, dtype=int)

# Construct a deepsnap tensor backend HeteroGraph
hetero_graph = HeteroGraph(
    node_feature=node_feature,
    node_label=node_label,
    edge_index=edge_index,
    directed=False
)
print(f"Collab heterogeneous graph: {hetero_graph.num_nodes()} nodes, {hetero_graph.num_edges()} edges")

# Node feature and node label to device
for key in hetero_graph.node_feature:
    hetero_graph.node_feature[key] = hetero_graph.node_feature[key].to(args['device'])
for key in hetero_graph.node_label:
    hetero_graph.node_label[key] = hetero_graph.node_label[key].to(args['device'])

# Edge_index to sparse tensor and to device
for key in hetero_graph.edge_index:
    edge_index = hetero_graph.edge_index[key]
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(hetero_graph.num_nodes('author'), hetero_graph.num_nodes('author')))
    hetero_graph.edge_index[key] = adj.t().to(args['device'])


# 
edge_index = data.edge_index
data.edge_weight = data.edge_weight.view(-1).to(torch.float)
data = T.ToSparseTensor()(data)

split_edge = dataset.get_edge_split()

model = HeteroGNN(hetero_graph, args, aggr="mean").to(args['device'])

predictor = LinkPredictor(args['hidden_size'], args['hidden_size'], 1,
                            3, dropout=0).to(args['device'])

evaluator = Evaluator(name='ogbl-collab')
loggers = {
    'Hits@10': Logger(args['run'], args),
    'Hits@50': Logger(args['run'], args),
    'Hits@100': Logger(args['run'], args),
}


for run in range(args['run']):
    model.reset_parameters()
    predictor.reset_parameters()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=args['lr'])

    for epoch in range(1, 1 + args['epochs']):
        loss = train(model, predictor, hetero_graph, split_edge, optimizer, args['batch_size'])

        if epoch % args['eval_steps'] == 0:
            results = test(model, predictor, hetero_graph, data, split_edge, evaluator,
                            args['batch_size'])
            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args['log_steps'] == 0:
                for key, result in results.items():
                    train_hits, valid_hits, test_hits = result
                    print(key)
                    print(f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                print('---')

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(run)

for key in loggers.keys():
    print(key)
    loggers[key].print_statistics()