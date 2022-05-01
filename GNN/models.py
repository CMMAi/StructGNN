import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from .layers import *


class Structure_GraphNetwork(torch.nn.Module):
    def __init__(self, layer_num, input_dim, hidden_dim, edge_attr_dim, aggr, 
                 gnn_act=True, gnn_dropout=True, dropout_p=0.0,
                 node_out_dispX_dim=1, node_out_dispZ_dim=1,
                 node_out_momentY_dim=6, node_out_momentZ_dim=6,
                 node_out_shearY_dim=6, node_out_shearZ_dim=6, device="cuda",
                 **kwargs):
        super(Structure_GraphNetwork, self).__init__()
        self.layer_num = layer_num
        self.gnn_act = gnn_act
        self.gnn_dropout = gnn_dropout
        self.dropout_p = dropout_p
        self.device = device
        
        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)
        
        self.conv_layer = GraphNetwork_layer(hidden_dim, hidden_dim, aggr=aggr, edge_attr_dim=edge_attr_dim)      
            
        self.node_dispX_decoder = MLP(hidden_dim, [64], node_out_dispX_dim, act=True, dropout=False)
        self.node_dispZ_decoder = MLP(hidden_dim, [64], node_out_dispZ_dim, act=True, dropout=False)
        self.node_momentY_decoder = MLP(hidden_dim, [64], node_out_momentY_dim, act=True, dropout=False)
        self.node_momentZ_decoder = MLP(hidden_dim, [64], node_out_momentZ_dim, act=True, dropout=False)
        self.node_shearY_decoder = MLP(hidden_dim, [64], node_out_shearY_dim, act=True, dropout=False)
        self.node_shearZ_decoder = MLP(hidden_dim, [64], node_out_shearZ_dim, act=True, dropout=False)

        
    def forward(self, x, edge_index, edge_attr):
        x = self.encoder(x)
                
        for i in range(self.layer_num):
            x = self.conv_layer(x, edge_index, edge_attr)
            if self.gnn_act:
                x = F.relu(x)
            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                
        # Node output
        node_out_dispX = self.node_dispX_decoder(x)
        node_out_dispZ = self.node_dispZ_decoder(x)
        node_out_momentY = self.node_momentY_decoder(x)
        node_out_momentZ = self.node_momentZ_decoder(x)
        node_out_shearY = self.node_shearY_decoder(x)
        node_out_shearZ = self.node_shearZ_decoder(x)
        
        node_out = torch.zeros(size=(x.shape[0], 26)).to(self.device)
        node_out[:, 0] = node_out_dispX[:, 0]
        node_out[:, 1] = node_out_dispZ[:, 0]
        node_out[:, 2:8] = node_out_momentY[:, :]
        node_out[:, 8:14] = node_out_momentZ[:, :]
        node_out[:, 14:20] = node_out_shearY[:, :]
        node_out[:, 20:26] = node_out_shearZ[:, :]
                
        return node_out
    



class Structure_GraphNetwork_pseudo(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, aggr, edge_attr_dim,
                 gnn_act=True, gnn_dropout=True, dropout_p=0.0,
                 node_out_dispX_dim=1, node_out_dispZ_dim=1,
                 node_out_momentY_dim=6, node_out_momentZ_dim=6,
                 node_out_shearY_dim=6, node_out_shearZ_dim=6, device="cuda",
                 **kwargs):
        super(Structure_GraphNetwork_pseudo, self).__init__()
        self.gnn_act = gnn_act
        self.gnn_dropout = gnn_dropout
        self.dropout_p = dropout_p
        self.device = device
        
        self.encoder = MLP(input_dim, [], hidden_dim, act=False, dropout=False)
        
        self.conv_layer = GraphNetwork_layer(hidden_dim, hidden_dim, aggr=aggr)      
            
        self.node_dispX_decoder = MLP(hidden_dim, [64], node_out_dispX_dim, act=True, dropout=False)
        self.node_dispZ_decoder = MLP(hidden_dim, [64], node_out_dispZ_dim, act=True, dropout=False)
        self.node_momentY_decoder = MLP(hidden_dim, [64], node_out_momentY_dim, act=True, dropout=False)
        self.node_momentZ_decoder = MLP(hidden_dim, [64], node_out_momentZ_dim, act=True, dropout=False)
        self.node_shearY_decoder = MLP(hidden_dim, [64], node_out_shearY_dim, act=True, dropout=False)
        self.node_shearZ_decoder = MLP(hidden_dim, [64], node_out_shearZ_dim, act=True, dropout=False)
        
    def forward(self, x, edge_index, edge_attr, layer_num):
        x = self.encoder(x)
                
        for i in range(layer_num):
            x = self.conv_layer(x, edge_index, edge_attr)
            if self.gnn_act:
                x = F.relu(x)
            if self.gnn_dropout:
                x = F.dropout(x, p=self.dropout_p, training=self.training)
                
        # Node output
        node_out_dispX = self.node_dispX_decoder(x)
        node_out_dispZ = self.node_dispZ_decoder(x)
        node_out_momentY = self.node_momentY_decoder(x)
        node_out_momentZ = self.node_momentZ_decoder(x)
        node_out_shearY = self.node_shearY_decoder(x)
        node_out_shearZ = self.node_shearZ_decoder(x)

        
        node_out = torch.zeros(size=(x.shape[0], 26)).to(self.device)
        node_out[:, 0] = node_out_dispX[:, 0]
        node_out[:, 1] = node_out_dispZ[:, 0]
        node_out[:, 2:8] = node_out_momentY[:, :]
        node_out[:, 8:14] = node_out_momentZ[:, :]
        node_out[:, 14:20] = node_out_shearY[:, :]
        node_out[:, 20:26] = node_out_shearZ[:, :]
                
        return node_out

