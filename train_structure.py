import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from argparse import ArgumentParser


from GNN.models import *
from GNN import losses
from Utils import plot
from Utils import datasets
from Utils import accuracy
from Utils import normalization


import time
from datetime import datetime
import os
import json



space = plot.print_space()


# Args
parser = ArgumentParser()
# general args
# Dataset
parser.add_argument('--dataset_name', dest='dataset_name', default='Static_Linear_Analysis', type=str)
parser.add_argument('--whatAsNode', dest='whatAsNode', default='NodeAsNode', type=str)
parser.add_argument('--data_num', dest='data_num', default=100, type=int)
parser.add_argument('--train_ratio', dest='train_ratio', default=0.9, type=float)
parser.add_argument('--normalization', dest='normalization', default=True, type=bool)

# GNN
parser.add_argument('--model', dest='model', default='Structure_GraphNetwork', type=str, help='model class name. E.g., GCN, Tony_GNN, ...')
parser.add_argument('--aggr_type', dest='aggr', default='mean', type=str)
parser.add_argument('--hidden_dim', dest='hidden_dim', default=256, type=int)
parser.add_argument('--layer_num', dest='layer_num', default=9, type=int)
parser.add_argument('--dropout_p', dest='dropout_p', default=0.0, type=float)
parser.add_argument('--gnn_dropout', dest='gnn_dropout', default='True', type=bool, help='whether dropout, default 0.5')
parser.add_argument('--gnn_act', dest='gnn_act', default='True', type=bool, help='whether activation, default RELU')

# Training
parser.add_argument('--target', dest='target', default='all', type=str, help='which output target you are going to train, like displacement, moement, shear......')
parser.add_argument('--epoch_num', dest='epoch_num', default=10, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=1, type=int)
parser.add_argument('--lr', dest='lr', default=5e-5, type=float)
parser.add_argument('--loss_function', dest='loss_function', default='L1_Loss', type=str)
parser.add_argument('--accuracy_threshold', dest='accuracy_threshold', default=1e-4, type=float, help='The normalized output value smaller than the threshold will be ignored.')

# Training
parser.add_argument('--training_time', dest='training_time', default=0, type=float)

args = parser.parse_args()
print(args, end=space)





torch.manual_seed(0)
date_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
print("Start time: " + datetime.now().strftime('%b %d, %H:%M:%S'), end=space)


# Load in data
dataset = datasets.get_dataset(dataset_name=args.dataset_name,
                               whatAsNode=args.whatAsNode,
                               structure_num=args.data_num)
print(f"dataset: {args.dataset_name}")


# Normalization
if args.normalization:
    dataset, norm_dict = normalization.normalize_dataset(dataset, analysis='linear')
    print("Normalization: True")
    print("Normlization dictionary: ")
    print(norm_dict, end=space)


# Get the first graph object.
data = dataset[0] 
print(data.x.shape)
print(data.x[:5])
print(data.edge_index)
print(data.edge_attr)



# Device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}", end=space)


# Split data into train, test set
torch.manual_seed(0)

train_dataset, valid_dataset, _ = datasets.split_dataset(dataset,
                                                       train_ratio=args.train_ratio,
                                                       valid_ratio=1-args.train_ratio)

print(f"Number of training graphs: {len(train_dataset)}")
print(f"Number of test graphs: {len(valid_dataset)}", end=space)


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)



# Training targets
y_start, y_finish = datasets.get_target_index(args.target)




# Model setup
input_dim = data.x.shape[1]
edge_attr_dim = 3
node_output_dim = data.y.shape[1]
model_constructor_args = {
    'input_dim': input_dim, 'hidden_dim': args.hidden_dim, 'dropout_p': args.dropout_p,
    'layer_num': args.layer_num, 'edge_attr_dim': edge_attr_dim, 'aggr': args.aggr,
    'gnn_act': args.gnn_act, 'gnn_dropout': args.gnn_dropout, 'device': device
}
model = locals()[args.model](**model_constructor_args).to(device)


# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.loss_function == 'L1_Loss':
    criterion = losses.L1_Loss()
elif args.loss_function == 'L2_Loss':
    criterion = losses.L2_Loss()


accuracy_record = np.zeros((3, args.epoch_num))
loss_record = np.zeros((3, args.epoch_num))

best_accuracy = 0



print(model, end=space)



# Save Model
task = args.dataset_name
save_model_dir = 'Results/' + task + '/'
if os.path.exists(save_model_dir) == False:
    os.mkdir(save_model_dir)
save_model_dir += date_str + '/'
os.mkdir(save_model_dir)





# Train and Test
start_time = time.time()
for epoch in range(args.epoch_num):
    # Train
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        node_out = model(data.x, data.edge_index, data.edge_attr) 
        loss = criterion(node_out[:, y_start:y_finish], data.y[:, y_start:y_finish], args.accuracy_threshold)
        loss.backward()
        optimizer.step()
        
    # Get train and test score
    model.eval()
    with torch.no_grad():
        for i, loader in enumerate([train_loader, valid_loader]):
            loss_val = 0
            total_correct = 0
            total_elems = 0
            
            for data in loader:
                data = data.to(device)
                node_out = model(data.x, data.edge_index, data.edge_attr)                
                loss = criterion(node_out[:, y_start:y_finish], data.y[:, y_start:y_finish], args.accuracy_threshold)
                loss_val += loss.item()
                
                correct, elems = accuracy.node_accuracy(node_out[:, y_start:y_finish], data.y[:, y_start:y_finish], args.accuracy_threshold)
                
                total_correct += correct
                total_elems += elems
                
            accuracy_record[i][epoch] = total_correct / total_elems
            loss_record[i][epoch] = loss_val / total_elems
            

    print(f' Epoch: {epoch:03d},  Train Acc: {accuracy_record[0][epoch]:.4f},  Valid Acc: {accuracy_record[1][epoch]:.4f},  ' + 
        f'Train Loss: {loss_record[0][epoch]:.4f},  Valid Loss: {loss_record[1][epoch]:.4f}')

    if accuracy_record[1][epoch] > best_accuracy:
        best_accuracy = accuracy_record[1][epoch]
        torch.save(model.state_dict(), save_model_dir + 'model.pt')
        print(f'Trained model saved, valid acc: {best_accuracy:.4f}')




finish_time = time.time()
args.training_time = (finish_time - start_time)/60
print(space)
print(f"Time spent: {(finish_time - start_time)/60:.2f} min")
print("Finish time: " + datetime.now().strftime('%b %d, %H:%M:%S'), end=space)

with open(save_model_dir + 'training_args.json', 'w') as f:
    json.dump(vars(args), f)






# Plot the results
# visualize_graph(dataset[int(torch.rand(1)*50)], save_model_dir)
plot.plot_learningCurve(accuracy_record, save_model_dir, title=', '.join([args.model, task+'\n', date_str, args.target]), target=args.target)
plot.plot_lossCurve(loss_record, save_model_dir, title=', '.join([args.model, task+'\n', date_str, args.target]), target=args.target)
