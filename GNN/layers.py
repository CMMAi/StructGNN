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

        
        
class GraphNet_BeamAsNode_layer(MessagePassing):
    def __init__(self, input_dim, output_dim, message_dim=None, aggr='mean'):
        super(GraphNet_BeamAsNode_layer, self).__init__()
        self.aggr = aggr
        message_dim = input_dim if message_dim is None else message_dim
        self.messageMLP = MLP(input_dim * 2, [], message_dim, act=False)
        self.outputMLP = MLP(input_dim + message_dim, [], output_dim, act=False)


    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x, messageMLP=self.messageMLP, outputMLP=self.outputMLP)

    def message(self, x_i, x_j, messageMLP):
        return messageMLP(torch.cat((x_i, x_j), dim=-1))

    def update(self, aggr_out, x, outputMLP):
        return outputMLP(torch.cat((x, aggr_out), dim=-1))
 


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



class GraphNetwork_layer_restricted(MessagePassing):
    def __init__(self, input_dim, output_dim, message_dim=None, aggr='max'):
        super(GraphNetwork_layer_restricted, self).__init__()
        self.aggr = aggr
        message_dim = input_dim if message_dim is None else message_dim
        self.messageMLP = MLP(input_dim + input_dim, [], message_dim, act=False, dropout=False)
        self.outputMLP = MLP(input_dim + message_dim, [], output_dim, act=False, dropout=False)
        

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x, messageMLP=self.messageMLP, outputMLP=self.outputMLP)

    def message(self, x_i, x_j, messageMLP):
        return messageMLP(torch.cat((x_i, x_j), dim=-1))

    def update(self, aggr_out, x, outputMLP):
        return outputMLP(torch.cat((x, aggr_out), dim=-1))



class GraphNetwork_reg_layer(MessagePassing):
    def __init__(self, input_dim, output_dim, edge_attr_dim=3, message_dim=None, aggr='max'):
        super(GraphNetwork_reg_layer, self).__init__()
        self.aggr = aggr
        message_dim = input_dim if message_dim is None else message_dim
        
        self.messageMLP = MLP(input_dim * 2 + edge_attr_dim, [], message_dim, act=True)
        self.outputMLP = MLP(input_dim + message_dim, [], output_dim, act=True)
        
        self.edge_message = None
        

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, messageMLP=self.messageMLP, outputMLP=self.outputMLP)

    def message(self, x_i, x_j, edge_attr, messageMLP):
        self.edge_message = messageMLP(torch.cat((x_i, x_j, edge_attr), dim=-1))
        # self.edge_message = messageMLP(torch.cat((x_j, edge_attr), dim=-1))
        return self.edge_message

    def update(self, aggr_out, x, outputMLP):
        return outputMLP(torch.cat((x, aggr_out), dim=-1)), self.edge_message



class GraphNet_pseudo_layer(MessagePassing):
    def __init__(self, aggr='max'):
        super(GraphNet_pseudo_layer, self).__init__()
        self.aggr = aggr
        
        self.edge_message = None
        

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # force1 = torch.cat((x_i[:, 9].view(-1, 1), x_j[:, 9].view(-1, 1)), dim=1)
        # force2 = torch.cat((x_i[:, 10].view(-1, 1), x_j[:, 10].view(-1, 1)), dim=1)
        # message_1 = torch.max(force1, dim=1)[0].view(-1, 1)
        # message_2 = torch.max(force2, dim=1)[0].view(-1, 1)
        
        
        # Height
        # height_i = x_i[:, 4]
        # height_j = x_j[:, 4]
        # height = ((height_i + height_j) / 2).view(-1, 1)
        # message_1 = height
        # message_2 = height
        
        
        # try only the node sends force are message
        message_1 = x_j[:, 9].view(-1, 1)


        # message_1[x_i[:, 9] == x_j[:, 9]] = 0
                
        edge_message = message_1
                
        self.edge_message = edge_message
        return self.edge_message

    def update(self, aggr_out, x):
        x[:, 9] = torch.max(torch.cat((x[:, 9].view(-1, 1), aggr_out[:, 0].view(-1, 1)), dim=1), dim=1)[0]
        return x, self.edge_message



     
