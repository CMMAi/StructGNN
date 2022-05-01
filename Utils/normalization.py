import torch
from torch_geometric.loader import DataLoader


def getMinMax_x(dataset, norm_dict):
    # norm_dict = {}
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))
    x, y, edge_attr = data.x, data.y, data.edge_attr 

    # data.x
    min_grid_num = 0
    max_grid_num = torch.max(torch.abs(x[:, :3]))
    norm_dict['grid_num'] = [min_grid_num, max_grid_num]
    
    min_coord = torch.min(torch.abs(x[:, 3:6]))
    max_coord = torch.max(torch.abs(x[:, 3:6]))
    norm_dict['coord'] = [min_coord, max_coord]
    
    min_mass = 0
    max_mass = torch.max(torch.abs(x[:, 8]))
    norm_dict['mass'] = [min_mass, max_mass]
    
    min_force = 0
    max_force = torch.max(torch.abs(x[:, 9:]))
    norm_dict['force'] = [min_force, max_force]

    # data.edge_attr
    min_length = 0
    max_length = torch.max(torch.abs(edge_attr[:, 2]))
    norm_dict['length'] = [min_length, max_length]
    
    del x, y, edge_attr
    return norm_dict




def getMinMax_y_linear(dataset, norm_dict):
    loader = DataLoader(dataset, batch_size=len(dataset))
    data = next(iter(loader))
    x, y = data.x, data.y
    
    # data.y 
    min_disp = 0
    max_disp = torch.max(torch.abs(y[:, :2]))
    norm_dict['disp'] = [min_disp, max_disp]
    
    min_momentY = 0
    max_momentY = torch.max(torch.abs(y[:, 2:8]))
    norm_dict['momentY'] = [min_momentY, max_momentY]
    
    min_momentZ = 0
    max_momentZ = torch.max(torch.abs(y[:, 8:14]))
    norm_dict['momentZ'] = [min_momentZ, max_momentZ]
    
    min_shearY = 0
    max_shearY = torch.max(torch.abs(y[:, 14:20]))
    norm_dict['shearY'] = [min_shearY, max_shearY]
    
    min_shearZ = 0
    max_shearZ = torch.max(torch.abs(y[:, 20:26]))
    norm_dict['shearZ'] = [min_shearZ, max_shearZ]
    
    min_axialForce = 0
    max_axialForce = torch.max(torch.abs(y[:, 26:32]))
    norm_dict['axialForce'] = [min_axialForce, max_axialForce]
    
    min_torsion = 0
    max_torsion = torch.max(torch.abs(y[:, 32:38]))
    norm_dict['torsion'] = [min_torsion, max_torsion]
    
    del x, y
    return norm_dict
    
  
    

def normalize_linear(data, norm_dict):
    data.x[:, :3] = (data.x[:, :3] - norm_dict['grid_num'][0]) / (norm_dict['grid_num'][1] - norm_dict['grid_num'][0])
    data.x[:, 3:6] = (data.x[:, 3:6] - norm_dict['coord'][0]) / (norm_dict['coord'][1] - norm_dict['coord'][0])
    data.x[:, 8] = (data.x[:, 8] - norm_dict['mass'][0]) / (norm_dict['mass'][1] - norm_dict['mass'][0])
    data.x[:, 9:] = (data.x[:, 9:] - norm_dict['force'][0]) / (norm_dict['force'][1] - norm_dict['force'][0])
        
    data.edge_attr[:, 2] = (data.edge_attr[:, 2] - norm_dict['length'][0]) / (norm_dict['length'][1] - norm_dict['length'][0])

    data.y[:, 0:2] = (data.y[:, 0:2] - norm_dict['disp'][0]) / (norm_dict['disp'][1] - norm_dict['disp'][0])
    data.y[:, 2:8] = (data.y[:, 2:8] - norm_dict['momentY'][0]) / (norm_dict['momentY'][1] - norm_dict['momentY'][0])
    data.y[:, 8:14] = (data.y[:, 8:14] - norm_dict['momentZ'][0]) / (norm_dict['momentZ'][1] - norm_dict['momentZ'][0])
    data.y[:, 14:20] = (data.y[:, 14:20] - norm_dict['shearY'][0]) / (norm_dict['shearY'][1] - norm_dict['shearY'][0])
    data.y[:, 20:26] = (data.y[:, 20:26] - norm_dict['shearZ'][0]) / (norm_dict['shearZ'][1] - norm_dict['shearZ'][0])
    data.y[:, 26:32] = (data.y[:, 26:32] - norm_dict['axialForce'][0]) / (norm_dict['axialForce'][1] - norm_dict['axialForce'][0])
    data.y[:, 32:38] = (data.y[:, 32:38] - norm_dict['torsion'][0]) / (norm_dict['torsion'][1] - norm_dict['torsion'][0])
  
 
 
 
 
def denormalize_grid_num(data, norm_dict):
    data = data * (norm_dict['grid_num'][1] - norm_dict['grid_num'][0]) + norm_dict['grid_num'][0]
    return data

def denormalize_coord(data, norm_dict):
    data = data * (norm_dict['coord'][1] - norm_dict['coord'][0]) + norm_dict['coord'][0]
    return data

def denormalize_momentZ(data, norm_dict):
    data = data * (norm_dict['momentZ'][1] - norm_dict['momentZ'][0]) + norm_dict['momentZ'][0]
    return data

def denormalize_shearY(data, norm_dict):
    data = data * (norm_dict['shearY'][1] - norm_dict['shearY'][0]) + norm_dict['shearY'][0]
    return data

def denormalize_disp(disp, norm_dict):
    disp = disp * (norm_dict['disp'][1] - norm_dict['disp'][0]) + norm_dict['disp'][0]
    return disp


def denormalize_y_linear(y, norm_dict):
    y[:, 0:2] = y[:, 0:2] * (norm_dict['disp'][1] - norm_dict['disp'][0]) + norm_dict['disp'][0]
    y[:, 2:8] = y[:, 2:8] * (norm_dict['momentY'][1] - norm_dict['momentY'][0]) + norm_dict['momentY'][0]
    y[:, 8:14] = y[:, 8:14] * (norm_dict['momentZ'][1] - norm_dict['momentZ'][0]) + norm_dict['momentZ'][0]
    y[:, 14:20] = y[:, 14:20] * (norm_dict['shearY'][1] - norm_dict['shearY'][0]) + norm_dict['shearY'][0]
    y[:, 20:26] = y[:, 20:26] * (norm_dict['shearZ'][1] - norm_dict['shearZ'][0]) + norm_dict['shearZ'][0]
    y[:, 26:32] = y[:, 26:32] * (norm_dict['axialForce'][1] - norm_dict['axialForce'][0]) + norm_dict['axialForce'][0]
    y[:, 32:38] = y[:, 32:38] * (norm_dict['torsion'][1] - norm_dict['torsion'][0]) + norm_dict['torsion'][0]

    return y
  
    
   
# Normalize the whole dataset.
def normalize_dataset(dataset, analysis='linear'):
    norm_dict = dict()
    norm_dict = getMinMax_x(dataset, norm_dict)
    
    norm_dict = getMinMax_y_linear(dataset, norm_dict)
    for data in dataset:
        normalize_linear(data, norm_dict)

    return dataset, norm_dict


# Normalize the new dataset with old datset's norm_dict
def normalize_dataset_byNormDict(dataset, norm_dict, analysis='linear'):
    for data in dataset:
        normalize_linear(data, norm_dict) 

    return dataset
