import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree

from torch.nn import init
import pdb

####################### Basic Ops #############################

# # PGNN layer, only pick closest node for message passing
class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim,dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze(-1) # Nonlinear layer return incorrect shape

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1) # (2708, 121, 32) * (2708, 121, 1)

        # Sizes of tensors must match except in dimension 2. Got 30 and 23 in dimension 0 (The offending index is 1)
        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1) # messages: (2708, 121, 32*2)
        
        if messages.shape[1] == 1:
            messages = self.linear_hidden(messages)
            messages = self.act(messages) # n*m*d
        else:
            messages = self.linear_hidden(messages).squeeze()
            messages = self.act(messages) # n*m*d

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure


### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x): # x: torch.Size([2708, 121, 1])
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x) # x: torch.Size([2708, 121, 1])
        return x

####################### NNs #############################

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.linear_first = nn.Linear(feature_dim, hidden_dim)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)


    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.linear_first(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.SAGEConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class GAT(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GAT, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GATConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GATConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GATConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class GIN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GIN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first_nn = nn.Linear(feature_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        else:
            self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_hidden = nn.ModuleList([tg.nn.GINConv(self.conv_hidden_nn[i]) for i in range(layer_num - 2)])

        self.conv_out_nn = nn.Linear(hidden_dim, output_dim)
        self.conv_out = tg.nn.GINConv(self.conv_out_nn)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x



class PGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, graph_pooling='mean', **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.graph_pooling = graph_pooling
        self.output_dim = output_dim
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
        if layer_num>1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(output_dim, 2*output_dim), torch.nn.BatchNorm1d(2*output_dim), torch.nn.ReLU(), torch.nn.Linear(2*output_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(output_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")


        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.output_dim, 1)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.output_dim, 1)

    def forward(self, data):
        x = data.x 
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax) # (2708, 121), (2708, 32)
        if self.layer_num == 1:
            return x_position
        x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)
        # x_position = F.normalize(x_position, p=2, dim=-1)
        h_graph = self.pool(x, data.batch)
        return self.graph_pred_linear(h_graph)



### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr) # (num_edges, emb_dim)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# # PGNN layer, only pick closest node for message passing
class PGNNConv(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim=300, dist_trainable=True):
        super(PGNNConv, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable
        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_feat = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim,1)
        self.linear_out_structure = nn.Linear(emb_dim * 2, emb_dim)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax, edge_attr):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze(-1) # Nonlinear layer return incorrect shape

        # edge_attr = self.edge_convs(feature, edge_index, edge_attr)
        feature = self.linear_feat(feature)

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1) # (2708, 121, 32) * (2708, 121, 1)

        # Sizes of tensors must match except in dimension 2. Got 30 and 23 in dimension 0 (The offending index is 1)
        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1) # messages: (2708, 121, 32*2)
        
        if messages.shape[1] == 1:
            messages = self.linear_hidden(messages)
            messages = self.act(messages) # n*m*d
        else:
            messages = self.linear_hidden(messages).squeeze()
            messages = self.act(messages) # n*m*d

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d
        out_structure = torch.cat([out_structure, edge_attr], dim=-1)
        out_structure = self.linear_out_structure(out_structure)
        return out_position, out_structure

class PGNN_node(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, graph_pooling='mean', 
                 emb_dim=300, **kwargs):
        super(PGNN_node, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.graph_pooling = graph_pooling
        self.output_dim = output_dim
        self.batch_norms = torch.nn.ModuleList()
        self.edge_convs = torch.nn.ModuleList()
        if layer_num == 1:
            hidden_dim = output_dim
        hidden_dim = emb_dim
        output_dim = emb_dim
        if feature_pre:
            self.linear_pre = nn.Linear(emb_dim, feature_dim)
            self.conv_first = PGNNConv(emb_dim, hidden_dim, emb_dim=emb_dim)
            self.edge_conv1 = GCNConv(emb_dim)
            self.bn_first = torch.nn.BatchNorm1d(hidden_dim)
        else:
            self.conv_first = PGNNConv(emb_dim, hidden_dim, emb_dim=emb_dim)
            self.edge_conv1 = GCNConv(emb_dim)
            self.bn_first = torch.nn.BatchNorm1d(hidden_dim)
        if layer_num>1:
            self.conv_hidden = nn.ModuleList([PGNNConv(hidden_dim, hidden_dim, emb_dim=emb_dim) for i in range(layer_num - 2)])
            self.batch_norms = nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for i in range(layer_num - 2)])
            self.edge_convs = nn.ModuleList([GCNConv(emb_dim) for i in range(layer_num - 1)])
            self.conv_out = PGNNConv(hidden_dim, output_dim, emb_dim=emb_dim)

        self.atom_encoder = AtomEncoder(emb_dim)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(output_dim, 2*output_dim), torch.nn.BatchNorm1d(2*output_dim), torch.nn.ReLU(), torch.nn.Linear(2*output_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(output_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")


        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.output_dim, 1)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.output_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.atom_encoder(x)

        if self.feature_pre:
            x = self.linear_pre(x)
        h = self.edge_conv1(x, edge_index, edge_attr)
        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax, h) # (2708, 121), (2708, 32)

        if self.layer_num == 1:
            return x_position
        x = self.bn_first(x)
        x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            h = self.edge_convs[i](h, edge_index, edge_attr)
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax, edge_index, h)
            x = self.batch_norms[i](x) 
            x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        h = self.edge_convs[-1](h, edge_index, edge_attr)
        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax, h)
        # x_position = F.normalize(x_position, p=2, dim=-1)
        h_graph = self.pool(x, data.batch)
        return self.graph_pred_linear(h_graph)

