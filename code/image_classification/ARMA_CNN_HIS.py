# system utilities
from __future__ import print_function
import os, datetime, argparse

# pytorch utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# computing utilities
import numpy as np
import math

# custom utilities
from tensorboardX import SummaryWriter
from visdom import Visdom
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tikzplotlib

import utils.config as cf
from utils.LeNet import *
from utils.AlexNet import *
from utils.VGG import *
from utils.ResNet import *

# debug
import readchar




def get_model(args):
    # path to the folder of all datasets
    data_path = args.data_path
    if not os.path.exists(data_path): 
        os.makedirs(data_path)

    # path to the folder of specified dataset
    dataset = args.dataset
    assert dataset in ["MNIST", "CIFAR10", "CIFAR100"], \
        "The specified dataset is not supported."
    print("Dataset: ", dataset)

    if args.dataset_path == "default":
        dataset_path = dataset
    else: # if args.dataset_path != "default":
        dataset_path = args.dataset_path

    data_path = os.path.join(data_path, dataset_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # path to the folder of all outputs
    outputs_path = args.outputs_path
    assert os.path.exists(outputs_path), \
        "The outputs folder does not exist."

    # path to the folder of specifed dataset
    outputs_path = os.path.join(outputs_path, dataset_path)
    assert os.path.exists(outputs_path), \
        "The outputs folder for the dataset does not exist."

    # create the name of the current network architecture
    if args.network_name == "default":
        network_name = args.model_type
    else:
        network_name = args.network_name
    outputs_path = os.path.join(outputs_path, network_name)
    assert os.path.exists(outputs_path), \
        "The outputs folder for the architecture does not exist."

    
    # path to the folder of current model
    outputs_path = os.path.join(outputs_path, 
        args.model_name + '_' + args.model_stamp)
    assert os.path.exists(outputs_path), \
        "The outputs folder for the specified model does not exist."

    # path to the folder of checkpoints
    model_path = os.path.join(outputs_path, args.model_path)
    assert os.path.exists(model_path), \
        "The models folder does not exist."


    ## Data formats and Dataloaders 
    Dataset = { "MNIST":   datasets.MNIST, 
                "CIFAR10": datasets.CIFAR10,
                "CIFAR100": datasets.CIFAR100}[dataset]

    if dataset == "MNIST" or dataset == "CIFAR10":
        num_classes = 10
    elif dataset == "CIFAR100":
        num_classes = 100


    ## Models (Multi-layer Perceptron or Convolutional Neural Networks)
    print("Model: ", args.model_type)

    # multi-layer perceptron
    if args.model_type == "LeNet5":
        model = LeNet5()

    elif args.model_type == "AlexNet":
        model = AlexNet(args.arma, num_classes, args.rf_init)

    elif args.model_type == "VGG":
        model = VGG(args.model_arch, args.arma, args.barch_norm, args.dropout, num_classes, args.rf_init
            ,args.conv_kernel_size, args.conv_padding)

    elif args.model_type == "ResNet":
        model = ResNet_(args.model_arch, args.arma, num_classes, args.rf_init)

    else:
        raise NotImplementedError


    ## Main script for testing
    if args.eval_auto:
        if args.eval_best:
            model_file = os.path.join(model_path, 'training_best.pt')
        else: # if args.eval_last:
            model_file = os.path.join(model_path, 'training_last.pt')
    else: # if args.eval_spec:
        model_file = os.path.join(model_path, 'training_%d.pt' % args.eval_epoch)

    assert os.path.exists(model_file), \
        "The specified model is not found in the folder."

    device = torch.device('cpu')
    checkpoint = torch.load(model_file, map_location=device)
    eval_epoch = checkpoint.get("epoch", args.eval_epoch)
    model.load_state_dict(checkpoint["model_state_dict"])


    # evaluation on the test set
    model.eval()

    
    count = 0
    for i, (name, param) in enumerate(model.named_parameters()):
        if "autoregressive" in name:
            count += 1

    return model, count


def generate_A(alpha):
    alpha = alpha.tanh() / math.sqrt(2)
    chunks = torch.chunk(alpha, alpha.size()[-1], -1)

    # size: [T, P, 1]
    A_x_left  = (chunks[0] * math.cos(-math.pi / 4) - 
                 chunks[1] * math.sin(-math.pi / 4))

    A_x_right = (chunks[0] * math.sin(-math.pi / 4) +
                 chunks[1] * math.cos(-math.pi / 4))

    A_y_left  = (chunks[2] * math.cos(-math.pi / 4) - 
                 chunks[3] * math.sin(-math.pi / 4))

    A_y_right = (chunks[2] * math.sin(-math.pi / 4) + 
                 chunks[3] * math.cos(-math.pi / 4))

    # zero padding + circulant shift: 
    # [A_x_left 1 A_x_right] -> [1 A_x_right 0 0 ... 0 A_x_left]
    # size: [T, P, 3]->[T, P, I1] or [T, P, I2]
    A_x = torch.cat((torch.ones(chunks[0].size(), device=alpha.device), 
        A_x_right, A_x_left), -1)
        # x.size()[-2] - 3, device = alpha.device), A_x_left), -1)

    A_y = torch.cat((torch.ones(chunks[2].size(), device = alpha.device), 
        A_y_right,  A_y_left), -1)
        # x.size()[-1] - 3, device = alpha.device), A_y_left), -1)

    # size: [T, P, 3] + [T, P, 3] -> [T, P, 3, 3]
    A = torch.einsum('tzi,tzj->tzij',(A_x, A_y))
    # A[:,:,0,0] = 0
    return A



def draw_tensorboard(model, count):
    his_writer = SummaryWriter('./histogram')
    num =  0
    for i, (name, param) in enumerate(model.named_parameters()):
        if "autoregressive" in name:
            A = generate_A(param)
            his_writer.add_histogram(name, A, bins=30)
            num += 1
    his_writer.close()
            
def draw_visdom(model):
    viz = Visdom()

    for i, (name, param) in enumerate(model.named_parameters()):
        if "autoregressive" in name:
            A = generate_A(param)
            viz.histogram(torch.flatten(A), opts=dict(title=name, numbins=10))








def draw_matplotlib(model, range_max, range_min, 
        xlabel, ylabel, title, save_name):
    """
    Draw the historgram of the autoregressive coefficients.
    """
    
    flag = False
    for i, (name, param) in enumerate(model.named_parameters()):
        if "autoregressive" in name:
            A = generate_A(param)
            if flag == False:
                data = A.detach().cpu().numpy().ravel()
                flag = True
            else:
                data = np.hstack((data, A.detach().cpu().numpy().ravel()))

    plt.hist(data, bins = 40, range = (range_min, range_max), 
        weights = np.ones(len(data)) / len(data) * 100)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    tikzplotlib.save(save_name + ".tex")
    # plt.savefig(save_name + ".png")   

def main(args):
    model, count = get_model(args)
    draw_matplotlib(model, args.range_max, args.range_min, 
                    args.xlabel, args.ylabel, args.title,  args.save_name)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Output ARMA Layers' Histogram.")

    ## Models (MLP, CNN-P, CNN-S)
    parser.add_argument("--model-type", default = "ResNet", type = str,
                        help = "The type of the model (options: LeNet5/AlexNet/VGG/ResNet).")
    
    parser.add_argument("--model-arch", default="ResNet18", type=str, 
                        help="The depth of a specified Net.")
    parser.add_argument("--no-barch-norm", dest="barch_norm", action="store_false",
                        help="Do not use batch_norm in the model.")
    parser.set_defaults(barch_norm = True)
    parser.add_argument("--no-arma", dest="arma", action="store_false",
                        help="Do not use arma layer in the model.")
    parser.set_defaults(arma=True)
    parser.add_argument("--no-dropout", dest="dropout", action="store_false",
                        help="Do not use dropout in the layer of vgg.")
    parser.set_defaults(dropout=True)
    parser.add_argument("--no-sgd", dest="sgd", action="store_false",
                        help="Do not use sgd in the optimizer.")
    parser.set_defaults(sgd=True)
    parser.add_argument("--rf-init", default=0, type=float, metavar='S',
                        help='rf-strength')


    parser.add_argument("--model-name",  default = "default", type = str,
        help = "The model name (to create the associated folder).")
    parser.add_argument("--model-stamp", default = "default", type = str,
        help = "The time stamp of the model (as a suffix to the its name).")


    ## Paths (Data, Checkpoints, Results and TensorboardX)

    # inputs:  data
    parser.add_argument("--dataset", default = "CIFAR10", type = str,
        help = "The dataset used for training (options: MNIST/CIFAR10/CIFAR100/ImageNet).")
    parser.add_argument("--data-path", default = "../data", type = str,
        help = "The path to the folder stroing the data.")
    parser.add_argument("--dataset-path", default = "default", type = str,
        help = "The folder for the dataset.")

    # outputs: checkpoints, statistics and tensorboard
    parser.add_argument("--outputs-path", default = "../data/outputs", type = str,
        help = "The path to the folder storing outputs from training.")
    parser.add_argument("--network-name", default = "default", type = str, 
        help = "The architecture model (to create the folder).")

    parser.add_argument("--model-path", default = "models", type = str,
        help = "The folder for all checkpoints in training.")


    ## Hyperparameters for evaluation
    parser.add_argument('--eval-auto', dest = 'eval_auto', action = 'store_true', 
        help = 'Evaluate the best or the last model.')
    parser.add_argument('--eval-spec', dest = 'eval_auto', action = 'store_false', 
        help = 'Evaluate the model of specified epoch')
    parser.set_defaults(eval_auto = True)

    # if eval_auto is True (--eval-auto)
    parser.add_argument('--eval-best', dest = 'eval_best', action = 'store_true',
        help = 'Evaluate the best model (in term of validation loss).')
    parser.add_argument('--eval-last', dest = 'eval_best', action = 'store_false',
        help = 'Evaluate the last model (in term of training epoch).')
    parser.set_defaults(eval_best = False)

    # if eval_auto is False (--eval-spec)
    parser.add_argument('--eval-epoch', default = 300, type = int, 
        help = 'Evaluate the model of specified epoch.')

    parser.add_argument('--range-max', default = 0.1, type = float, 
        help = 'max value of histogram range')
    parser.add_argument('--range-min', default = -0.1, type = float, 
        help = 'min value of histogram range')

    parser.add_argument("--xlabel", default = "coefficient", type = str,
        help = "The xlabel of histogram")
    parser.add_argument("--ylabel", default = "percentage(%)", type = str,
        help = "The ylabel of histogram")
    parser.add_argument("--title", default = "Histogram of the Autoregressive Coefficients", type = str,
        help = "The title of histogram")
    parser.add_argument("--save-name", default = "default", type = str,
        help = "The name of tex file")

    parser.add_argument('--conv-kernel-size', default = 3, type = int, 
        help = 'The kernel size of conv operation.')
    parser.add_argument('--conv-padding', default = 1, type = int, 
        help = 'The padding num of the conv operation.')


    main(parser.parse_args())