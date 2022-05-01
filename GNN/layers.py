import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.nn import init
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import math


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act=False, dropout=False, p=0.5, **kwargs):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.act = act
        self.p = p
        concat_dim = [input_dim] + list(hidden_dim) + [output_dim]

        self.module_list = nn.ModuleList()
        for i in range(len(concat_dim)-1):
            self.module_list.append(nn.Linear(concat_dim[i], concat_dim[i+1]))
    
    def forward(self, x):
        for i, module in enumerate(self.module_list):
            x = module(x)
            if self.act and i != len(self.module_list)-1:
            # if self.act:
                x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=self.p, training=self.training)
        return x

        
    
class GraphNetwork_layer(MessagePassing):
    def __init__(self, input_dim, output_dim, edge_attr_dim=3, message_dim=None, aggr='mean'):
        super(GraphNetwork_layer, self).__init__()
        self.aggr = aggr
        message_dim = input_dim if message_dim is None else message_dim

        self.messageMLP = MLP(input_dim * 2 + edge_attr_dim, [], message_dim, act=False, dropout=False)
        self.outputMLP = MLP(input_dim + message_dim, [], output_dim, act=False, dropout=False)
        

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, messageMLP=self.messageMLP, outputMLP=self.outputMLP)

    def message(self, x_i, x_j, edge_attr, messageMLP):
        return messageMLP(torch.cat((x_i, x_j, edge_attr), dim=-1))

    def update(self, aggr_out, x, outputMLP):
        return outputMLP(torch.cat((x, aggr_out), dim=-1))


     
