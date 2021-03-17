"""
Heterougeous model import to hete_graph.py
Notes: Solve CUDA OOM issue. Groups Years in brackets 
"""
import copy
import torch
import deepsnap
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter

from sklearn.metrics import f1_score
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from torch_sparse import SparseTensor, matmul

class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        # To simplify implementation, please initialize both self.lin_dst
        # and self.lin_src out_features to out_channels
        self.lin_dst = None
        self.lin_src = None

        self.lin_update = None

        ############# Your code here #############
        ## (~3 lines of code)
        self.lin_dst = nn.Linear(self.in_channels_dst, out_channels)
        self.lin_src = nn.Linear(self.in_channels_src, out_channels)  
        self.lin_update = nn.Linear(out_channels*2, out_channels) # 


        ##########################################

    def reset_parameters(self):
        self.lin_dst.reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_update.reset_parameters()

    def forward(
        self,
        node_feature_src,
        node_feature_dst,
        edge_index,
        size=None,
        res_n_id=None,
    ):
        ############# Your code here #############
        ## (~1 line of code)
        out = self.propagate(edge_index, node_feature_src=node_feature_src, node_feature_dst=node_feature_dst, size=size, res_n_id=res_n_id)
        return out
        ##########################################

    def message_and_aggregate(self, edge_index, node_feature_src):

        ############# Your code here #############
        ## (~1 line of code)
        ## Note:
        ## 1. Different from what we implemented in Colab 3, we use message_and_aggregate
        ## to replace the message and aggregate. The benefit is that we can avoid
        ## materializing x_i and x_j, and make the implementation more efficient.
        ## 2. To implement efficiently, following PyG documentation is helpful:
        ## https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
        ## 3. Here edge_index is torch_sparse SparseTensor.
        out = matmul(edge_index, node_feature_src, reduce='mean')
        ##########################################

        return out

    def update(self, aggr_out, node_feature_dst, res_n_id):

        ############# Your code here #############
        ## (~4 lines of code)
        # Takes in the output of aggregation as first argument
        src = self.lin_src(aggr_out) 
        dst = self.lin_dst(node_feature_dst)
        concat = torch.cat((dst, src), dim=1) # (B, hidden_dim)
        aggr_out = self.lin_update(concat)

        ##########################################

        return aggr_out


class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    def __init__(self, convs, args, aggr="mean"):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr
        self.device = args['device']

        # Map the index and message type
        self.mapping = {}

        # A numpy array that stores the final attention probability
        self.alpha = None

        self.attn_proj = None

        if self.aggr == "attn":
            ############# Your code here #############
            ## (~1 line of code)
            ## Note:
            ## 1. Initialize self.attn_proj here.
            ## 2. You should use nn.Sequential for self.attn_proj
            ## 3. nn.Linear and nn.Tanh are useful.
            ## 4. You can create a vector parameter by using:
            ## nn.Linear(some_size, 1, bias=False)
            ## 5. The first linear layer should have out_features as args['attn_size']
            ## 6. You can assume we only have one "head" for the attention.
            ## 7. We recommend you to implement the mean aggregation first. After 
            ## the mean aggregation works well in the training, then you can 
            ## implement this part.
            self.attn_proj = nn.Sequential(nn.Linear(in_features=args['hidden_size'], out_features=args['attn_size']),
                                           nn.Tanh(),
                                           nn.Linear(args['attn_size'], 1, bias=False))

            ##########################################
            
    def reset_parameters(self):
        super(HeteroGNNWrapperConv, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                try:
                    layer.reset_parameters()
                except:
                    pass
    
    def forward(self, node_features, edge_indices):
        # Save CUDA space for mean aggregation
        # if self.aggr == 'mean':
        #     num_period = float(len(edge_indices))
        #     message_type_emb = {}
        #     for message_key, message_type in edge_indices.items():
        #         src_type, edge_type, dst_type = message_key
        #         node_feature_src = node_features[src_type]
        #         node_feature_dst = node_features[dst_type]
        #         edge_index = edge_indices[message_key]
        #         message_emb = self.convs[message_key](
        #                     node_feature_src,
        #                     node_feature_dst,
        #                     edge_index,
        #                 ) / num_period
        #         if 'author' in message_type_emb:
        #             message_type_emb['author'] += message_emb
        #         else:
        #             message_type_emb['author'] = message_emb
        #     return message_type_emb
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
                )
            )
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}        
        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb
    
    def aggregate(self, xs):
        # TODO: Implement this function that aggregates all message type results.
        # Here, xs is a list of tensors (embeddings) with respect to message 
        # type aggregation results.

        if self.aggr == "mean":

            ############# Your code here #############
            ## (~2 lines of code)
            out = torch.mean(torch.stack(xs), dim=0)
            return out
            ##########################################

        elif self.aggr == "attn":

            ############# Your code here #############
            ## (~10 lines of code)
            ## Note:
            ## 1. Store the value of attention alpha (as a numpy array) to self.alpha,
            ## which has the shape (len(xs), ) self.alpha will be not be used 
            ## to backpropagate etc. in the model. We will use it to see how much 
            ## attention the layer pays on different message types.
            ## 2. torch.softmax and torch.cat are useful.
            ## 3. You might need to reshape the tensors by using the 
            ## `view()` function https://pytorch.org/docs/stable/tensor_view.html
            
            # Compute e_m for every message
            ems = torch.zeros((len(xs),), device=self.device)
            for i in range(len(xs)):
                m = xs[i]
                em = self.attn_proj(m) # (3025, 1) 3025 nodes
                em = torch.mean(em) 
                ems[i] = em
            # Compute alpha
            alpha = torch.softmax(ems, dim=0)
            # alpha = alpha.cuda()
            reshape_xs = torch.cat(xs, dim=1).view(xs[0].shape[0], len(xs), xs[0].shape[1]) # (Vd, M, num of message, hidden_size=64)
            out = torch.matmul(alpha, reshape_xs) 
            # print(alpha[0] * xs[0][1] + alpha[1] * xs[1][1])
            # print(out[1]) 
            # print(alpha[0] * xs[0][1] + alpha[1] * xs[1][1] - out[1])
            # assert (alpha[0] * xs[0][1] + alpha[1] * xs[1][1] == out[1]).all()
            self.alpha = alpha.cpu().detach().numpy()
            return out

            ##########################################


def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    # TODO: Implement this function that returns a dictionary of `HeteroGNNConv` 
    # layers where the keys are message types. `hetero_graph` is deepsnap `HeteroGraph`
    # object and the `conv` is the `HeteroGNNConv`.

    convs = {}

    ############# Your code here #############
    ## (~9 lines of code)
    mtype = hetero_graph.message_types
    for mt in mtype:
        src, edge, dst = mt 
        if first_layer:
            src_dim = hetero_graph.num_node_features(src)
            dst_dim = hetero_graph.num_node_features(dst)
            convs[mt] = conv(in_channels_src=src_dim, in_channels_dst=dst_dim, out_channels=hidden_size)
        else:
            convs[mt] = conv(in_channels_src=hidden_size, in_channels_dst=hidden_size, out_channels=hidden_size)
    ##########################################
    
    return convs



class HeteroGNN(torch.nn.Module):
    def __init__(self, hetero_graph, args, aggr="mean"):
        super(HeteroGNN, self).__init__()
        
        self.graph = hetero_graph
        self.aggr = aggr
        self.hidden_size = args['hidden_size']

        self.convs1 = None
        self.convs2 = None

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.post_mps = nn.ModuleDict()

        ############# Your code here #############
        ## (~10 lines of code)
        ## Note:
        ## 1. For self.convs1 and self.convs2, call generate_convs at first and then
        ## pass the returned dictionary of `HeteroGNNConv` to `HeteroGNNWrapperConv`.
        ## 2. For self.bns, self.relus and self.post_mps, the keys are node_types.
        ## `deepsnap.hetero_graph.HeteroGraph.node_types` will be helpful.
        ## 3. Initialize all batchnorms to torch.nn.BatchNorm1d(hidden_size, eps=1.0).
        ## 4. Initialize all relus to nn.LeakyReLU().
        ## 5. For self.post_mps, each value in the ModuleDict is a linear layer 
        ## where the `out_features` is the number of classes for that node type.
        ## `deepsnap.hetero_graph.HeteroGraph.num_node_labels(node_type)` will be
        ## useful.
        self.convs1 = HeteroGNNWrapperConv(generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, first_layer=True), args, aggr=self.aggr)
        self.convs2 = HeteroGNNWrapperConv(generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size), args, aggr=self.aggr)
        for nt in hetero_graph.node_types:
            self.bns1[nt] = torch.nn.BatchNorm1d(self.hidden_size, eps=1.0)
            self.bns2[nt] = torch.nn.BatchNorm1d(self.hidden_size, eps=1.0)
            self.relus1[nt] = nn.LeakyReLU()
            self.relus2[nt] = nn.LeakyReLU()
            self.post_mps[nt] = nn.Linear(self.hidden_size, hetero_graph.num_node_labels(nt))

        ##########################################

    def reset_parameters(self):
        self.convs1.reset_parameters()
        self.convs2.reset_parameters()
        for nt in self.graph.node_types:
            self.bns1[nt].reset_parameters()
            self.bns2[nt].reset_parameters()
            self.post_mps[nt].reset_parameters()

    def forward(self, node_feature, edge_index):
        # TODO: Implement the forward function. Notice that `node_feature` is 
        # a dictionary of tensors where keys are node types and values are 
        # corresponding feature tensors. The `edge_index` is a dictionary of 
        # tensors where keys are message types and values are corresponding
        # edge index tensors (with respect to each message type).

        x = node_feature

        ############# Your code here #############
        ## (~7 lines of code)
        ## Note:
        ## 1. `deepsnap.hetero_gnn.forward_op` can be helpful.
        x = self.convs1(x, edge_index)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(x, edge_index)
        ##########################################
        return x

    def loss(self, preds, y, indices):
        
        loss = 0
        loss_func = F.cross_entropy

        ############# Your code here #############
        ## (~3 lines of code)
        ## Note:
        ## 1. For each node type in preds, accumulate computed loss to `loss`
        ## 2. Loss need to be computed with respect to the given index
        for k in preds:
            loss += loss_func(preds[k][indices[k]], y[k][indices[k]])
        ##########################################

        return loss


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)