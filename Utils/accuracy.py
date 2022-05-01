import torch

    
def node_accuracy(node_out, node_y, accuracy_threshold):
    # The value smaller than max_value * accuracy_threshold value will be set to 0, then be ignored.
    # Calculating accuracy uses denormalized y.
    device = node_out.device
    condition = torch.abs(node_y) > accuracy_threshold 

    ones = torch.ones(node_y.shape).to(device)[condition]
    zeros = torch.zeros(node_y.shape).to(device)[condition]
    relative_accuracy = torch.max(ones - torch.div(torch.abs(node_y[condition] - node_out[condition]), torch.abs(node_y[condition])), zeros)
    return relative_accuracy.sum(), torch.numel(relative_accuracy)