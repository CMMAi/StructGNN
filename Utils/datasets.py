import torch
from torch.utils.data import random_split



def get_dataset(dataset_name='Static_Linear_Analysis', whatAsNode='NodeAsNode', structure_num=300, special_path=None):
    if special_path != None:
        root = special_path
    else:
        root = 'Data/' + dataset_name + '/'
        
    data_list = []
    for index in range(1, structure_num+1):
        folder_name = root + 'structure_' + str(index) + '/'
        structure_graph_path = folder_name + 'structure_graph_' + whatAsNode + '.pt'
        try:
            graph = torch.load(structure_graph_path)
            data_list.append(graph)
            # print(graph)
        except:
            print("No file: ", structure_graph_path)
     
    return data_list




def split_dataset(dataset, 
                  train_ratio=0.9, valid_ratio=0.1, test_ratio=None):
    # Split into 2 cases, with test data and without test data
    length = dataset.__len__()
    if test_ratio == None:
        train_len = int(length * train_ratio)
        valid_len = length - train_len
        train_dataset, valid_dataset = random_split(dataset,
                                                    [train_len, valid_len],
                                                    generator=torch.Generator().manual_seed(731))
        return train_dataset, valid_dataset, None
    
    else:
        train_len = int(length * train_ratio)
        valid_len = int(length * valid_ratio)
        test_len = length - train_len - valid_len
        train_dataset, valid_dataset, test_dataset = random_split(dataset,
                                                                  [train_len, valid_len, test_len],
                                                                  generator=torch.Generator().manual_seed(731))
        return train_dataset, valid_dataset, test_dataset





def get_target_index(target):
    if target == 'disp_x':
        y_start = 0
        y_finish = 1
    elif target == 'disp_z':
        y_start = 1
        y_finish = 2
    elif target == 'disp':
        y_start = 0
        y_finish = 2
        
    elif target == 'momentY':
        y_start = 2
        y_finish = 8
    elif target == 'momentZ':
        y_start = 8
        y_finish = 14
    elif target == 'moment':
        y_start = 2
        y_finish = 14
        
    elif target == 'shearY':
        y_start = 14
        y_finish = 20
    elif target == 'shearZ':
        y_start = 20
        y_finish = 26
    elif target == 'shear':
        y_start = 14
        y_finish = 26    
        
    elif target == 'axialForce':
        y_start = 26
        y_finish = 32
    elif target == 'torsion':
        y_start = 32
        y_finish = 38
        
    elif target == 'all':
        y_start = 0
        y_finish = 26
    else:
        raise ValueError(f"There are no such output: {target} !")   
    
    return y_start, y_finish     