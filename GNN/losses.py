# Self-defined loss class
# https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
import torch
import torch.nn as nn


class L1_Loss(nn.Module):
    def __init__(self):
        super(L1_Loss, self).__init__()
        
    def forward(self, node_out, node_y, accuracy_threshold):
        # Calculating loss uses normalized y.
        condition = torch.abs(node_y) > accuracy_threshold
        disp_loss = torch.abs(node_y[condition] - node_out[condition]).sum()
        # disp_loss = torch.abs(node_y - node_out).sum()
        return disp_loss


class L2_Loss(nn.Module):
    def __init__(self):
        super(L2_Loss, self).__init__()
        
    def forward(self, node_out, node_y, accuracy_threshold):
        condition = torch.abs(node_y) > accuracy_threshold
        disp_loss = (torch.abs(node_y[condition] - node_out[condition])**2).sum()
        return disp_loss
    



