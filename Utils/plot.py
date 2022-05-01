import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils.convert import to_networkx
import networkx as nx


def print_space():
    return "\n"*3 + "="*100 + "\n"*3


def plot_learningCurve(accuracy_record, save_model_dir, title=None, target=None):
    train_acc, valid_acc, test_acc = accuracy_record    
    epochs = list(range(1, len(train_acc)+1))
    plt.figure(figsize=(8, 6))
    title += '\n' + f"Validation accuracy: {valid_acc.max()*100:.1f}%"
    plt.plot(epochs, train_acc, label='train')
    plt.plot(epochs, valid_acc, label='valid')
    if test_acc[-1] != 0:
        plt.plot(epochs, test_acc, label='test')
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([-0.1, 1.05])
    plt.title(title)
    plt.savefig(save_model_dir + "LearningCurve_" + target + ".png")
    # plt.show()
    plt.close()
    

def plot_lossCurve(loss_record, save_model_dir, title=None, target=None):
    train_acc, valid_acc, test_acc = loss_record    
    epochs = list(range(1, len(train_acc)+1))
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, label='train')
    plt.plot(epochs, valid_acc, label='valid')
    if test_acc[-1] != 0:
        plt.plot(epochs, test_acc, label='test')
    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    # plt.ylim([-0.1, 0.5])
    plt.title(title)
    plt.savefig(save_model_dir + "lossCurve_" + target + ".png")
    # plt.show()
    plt.close()
    
    
def plot_learningCurve_disp(accuracy_record, accuracy_record_disp, save_model_dir, title=None, train_or_valid='train'):
    acc = accuracy_record
    acc_x, acc_z = accuracy_record_disp  
    epochs = list(range(1, len(acc)+1))
    plt.figure(figsize=(8, 6))
    # plt.plot(epochs, moving_average(acc), label=f'{train_or_valid}')
    plt.plot(epochs, moving_average(acc_x), label=f'{train_or_valid}_x')
    plt.plot(epochs, moving_average(acc_z), label=f'{train_or_valid}_z')

    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([-0.1, 1.05])
    plt.title(title)
    plt.savefig(save_model_dir + f"LearningCurve_disp_{train_or_valid}.png")
    # plt.show()
    plt.close()



def plot_learningCurve_force(accuracy_record, accuracy_record_force, save_model_dir, title=None, train_or_valid='train'):
    acc = accuracy_record
    acc_momentY, acc_moementZ, acc_shearY, acc_shearZ, acc_axial = accuracy_record_force
    epochs = list(range(1, len(acc)+1))
    plt.figure(figsize=(8, 6))
    # plt.plot(epochs, moving_average(acc), label=f'{train_or_valid}')
    plt.plot(epochs, moving_average(acc_momentY), label=f'{train_or_valid}_momentY')
    plt.plot(epochs, moving_average(acc_moementZ), label=f'{train_or_valid}_momentZ')
    plt.plot(epochs, moving_average(acc_shearY), label=f'{train_or_valid}_shearY')
    plt.plot(epochs, moving_average(acc_shearZ), label=f'{train_or_valid}_shearZ')
    plt.plot(epochs, moving_average(acc_axial), label=f'{train_or_valid}_axial')

    plt.legend()
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim([-0.1, 1.05])
    plt.title(title)
    plt.savefig(save_model_dir + f"LearningCurve_force_{train_or_valid}.png")
    # plt.show()
    plt.close()



def plot_accuracy_distrubution(y_pred, y_real, save_model_dir, target=None, max_value=None, threshold=None):
    # absolute relative error 
    y_pred = y_pred[np.abs(y_real) > threshold * max_value]
    y_real = y_real[np.abs(y_real) > threshold * max_value]
        
    error = np.divide(np.absolute(y_pred - y_real), np.absolute(y_real))
    
    # BOXPLOTS
    error_accurate = error[error <= 0.1]    # errors for accurate predictions
    error_wrong = error[error > 0.1]        # errors for inaccurate predictions
    
    # 1. Boxplot of errors for all predictions
    plt.figure()
    e = error[np.isfinite(error)]
    plt.boxplot(e, vert=False, showfliers=True)
    plt.xlabel("relative absolute error")
    plt.title("All predictions")
    plt.savefig(save_model_dir + "error1.png")
    # plt.show()
    plt.close()
    
    # 2. boxplot of errors for accurate predictions
    plt.figure()
    e = error_accurate
    plt.boxplot(e, vert=False, showfliers=True)
    plt.xlabel("relative absolute error")
    plt.title("Accurate predictions")
    plt.savefig(save_model_dir + "error2.png")
    # plt.show()
    plt.close()
    
    # 3. boxplot of errors for inaccurate predictions
    plt.figure()
    e = error_wrong[np.isfinite(error_wrong)]
    plt.boxplot(e, vert=False, showfliers=True)
    plt.xlabel("relative absolute error")
    plt.title("Wrong predictions")
    plt.savefig(save_model_dir + "error3.png")
    # plt.show()
    plt.close()


    # HISTOGRAMS
    y_accurate = y_real[error <= 0.1]   # true labels for accurate predictions
    y_wrong = y_real[error > 0.1]       # true labels for inaccurate predictions
    
    
    # 4. histogram of error with accurate, inaccurate predictions
    # plt.figure()
    # plt.hist(error_accurate, bins=25)
    # plt.hist(error_wrong[np.isfinite(error_wrong)], bins=25, color='orange')
    # plt.xlabel("relative absolute error")
    # plt.title("Relative absolute error with accurate, inaccurate predictions")
    # plt.savefig(save_model_dir + "error4.png")
    # plt.show()
    
    # 5. histogram of true labels corresponding to accurate predictions
    plt.figure()
    plt.hist(y_accurate, bins=25)
    plt.xlabel(f'{target} (ground truth)')
    plt.ylabel('# of values')
    plt.xlim([np.min(y_real)-0.1*np.max(y_real), np.max(y_real)*1.1])
    y_min, y_max = plt.gca().get_ylim()
    plt.ylim([y_min, y_max])
    plt.title(f"True y labels of accurate predictions, {target}")
    plt.savefig(save_model_dir + "error5.png")
    # plt.show()
    plt.close()
    
    # 6. histogram of true labels corresponding to inaccurate predictions
    plt.figure()
    plt.hist(y_wrong[np.isfinite(y_wrong)], bins=5, color='orange')
    plt.xlabel(f'{target} (ground truth)')
    plt.ylabel('# of values')
    plt.xlim([np.min(y_real)-0.1*np.max(y_real), np.max(y_real)*1.1])
    plt.ylim([y_min, y_max])
    plt.title(f"True y labels of wrong predictions, {target}")
    plt.savefig(save_model_dir + "error6.png")
    # plt.show()
    plt.close()
    
    

def moving_average(record, half_length=10):
    record = np.array(record)
    average_record = np.zeros(len(record))
    for index in range(len(record)):
        if index < 10:
            average_record[index] = record[:20].mean()
        elif index + 10 > len(record):
            average_record[index] = record[-20:].mean()
        else:
            average_record[index] = record[index-10:index+10].mean()
    return average_record


def visualize_graph(data, save_model_dir, name):
    vis = to_networkx(data)
    plt.figure(1, figsize=(8, 8))
    nx.draw(vis, cmap=plt.get_cmap('Set3'),node_size=120, linewidths=13)
    plt.savefig(save_model_dir + name + ".png")
    plt.show()
    