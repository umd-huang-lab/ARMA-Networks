# system utilities
from __future__ import print_function
import os, datetime, argparse

# pytorch utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# computing utilities
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 16

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import math

# custom utilities
from utils.convlstmnet import ConvLSTMNet
# import tikzplotlib

def get_model(args):
    """
    Load the training model.
    """

    ## Model preparation (Conv-LSTM)

    # whether to use GPU (or CPU) 
    use_cuda  = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # whether to use multi-GPU (or single-GPU)
    multi_gpu = (use_cuda and args.multi_gpu 
        and torch.cuda.device_count() > 1)

    # number of GPUs used for training
    num_gpus = (torch.cuda.device_count() 
        if multi_gpu else 1) if use_cuda else 0
    
    print("Device: %s (# of GPUs: %d)" % (device, num_gpus))

        # size of the Conv-LSTM network
    if   args.model_size == 'deep':   # 16-layers
        layers_per_block = (2, ) * 8
        hidden_channels  = (64,) * 8
        skip_stride = 2
    elif args.model_size == 'wide':   # 12-layers
        layers_per_block = (3, ) * 4
        hidden_channels  = (96,) * 4
        skip_stride = 2
    elif args.model_size == "origin":  # 12-layers
        layers_per_block = (3, ) * 4
        hidden_channels  = (32, 48, 48, 32)
        skip_stride = 2
    elif args.model_size == "small":
        layers_per_block = (3, ) * 4
        hidden_channels  = (32,) * 4
        skip_stride = 2
    elif args.model_size == "shallow": # 4-layers
        layers_per_block = (4, )
        hidden_channels  = (128, )
        skip_stride = None 
    else:
        raise NotImplementedError

    # construct the model with the specified hyper-parameters
    model = ConvLSTMNet(
        # architecture and interfaces of the model
        layers_per_block = layers_per_block, hidden_channels = hidden_channels,
        input_channels = args.img_channels, output_sigmoid = args.use_sigmoid,
        skip_stride = skip_stride,  
        # hyper-parameters of convolutional layers
        arma = args.use_arma, w_dilation = args.w_dilation,
        w_kernel_size = args.w_kernel_size, w_bias = args.use_bias,
        a_kernel_size = args.a_kernel_size, a_padding_mode = args.a_padding_mode)

    # count the total number of model parameters
    num_params = sum(param.numel() for param 
        in model.parameters() if param.requires_grad)
    print("# of params. = ", num_params)

    # move the model to the device (CPU, GPU, multi-GPU) 
    model.to(device)
    if multi_gpu: model = nn.DataParallel(model)

    # create the name and timestamp of the model
    model_name = args.model_name + '_' + args.model_stamp

    print("Model name:", model_name)


    ## Models and Results
    if args.output_path == "default":
        OUTPUT_DIR = {"MNIST": "./moving-mnist", 
                      "KTH":   "./kth", 
                      "KITTI": "./kitti"}[args.dataset]
    else: # if args.output_path != "default":
        OUTPUT_DIR = args.output_path

    OUTPUT_DIR = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)

    # path to the models
    MODEL_DIR  =  os.path.join(OUTPUT_DIR, args.model_path)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # load the best / last / specified model
    if args.eval_auto:
        if args.eval_best:
            MODEL_FILE = os.path.join(MODEL_DIR, 'training_best.pt')
        else: # if args.eval_last:
            MODEL_FILE = os.path.join(MODEL_DIR, 'training_last.pt')
    else: # if args.eval_spec:
        MODEL_FILE = os.path.join(MODEL_DIR, 'training_%d.pt' % args.eval_epoch)

    assert os.path.exists(MODEL_FILE), \
        "The specified model is not found in the folder."

    checkpoint = torch.load(MODEL_FILE)
    eval_epoch = checkpoint.get("epoch", args.eval_epoch)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model


def generate_A(alpha):
    """
    Generate the autoregressive coefficients from the model parameters. 
    """

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

    return A


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

    np.save(save_name + ".npy", data)

    # plt.hist(data, bins = 40, range = (range_min, range_max), 
    #     weights = np.ones(len(data)) / len(data) * 100)
        
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.title(title)

    # tikzplotlib.save(save_name + ".tex")
    # # plt.savefig(save_name + ".png")

def main(args):
    model = get_model(args)
    draw_matplotlib(model, args.range_max, args.range_min, 
        args.xlabel, args.ylabel, args.title, args.save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Output ARMA Layers' Histogram.")

    ## Devices (CPU, single-GPU or multi-GPU)

    # whether to use GPU for testing
    parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true',
        help = 'Use GPU for testing.')
    parser.add_argument('--no-cuda',  dest = 'use_cuda', action = 'store_false', 
        help = "Use CPU for testing.")
    parser.set_defaults(use_cuda = True)

    # whether to use multi-GPU for testing (given GPU is used)
    parser.add_argument('--multi-gpu',  dest = 'multi_gpu', action = 'store_true',
        help = 'Use multiple GPUs for testing.')
    parser.add_argument('--single-gpu', dest = 'multi_gpu', action = 'store_false',
        help = 'Use single GPU for testing.')
    parser.set_defaults(multi_gpu = True)


    ## Models (Conv-LSTM or Conv-TT-LSTM)
    parser.add_argument('--img-channels', default = 1, type = int, 
        help = 'The number of channels in each video frame.')

    # model name (with time stamp as suffix)
    parser.add_argument('--model-name', default = "test", type = str,
        help = 'The model name is used to create the folder names.')
    parser.add_argument('--model-stamp', default = "0000", type = str, 
        help = 'The stamp is used to create the suffix to the model name.')

    # model type and size (depth and width)
    parser.add_argument('--use-arma', dest = 'use_arma', 
        action = 'store_true',  help = 'Use ARMA convolutions in ConvLSTM.')
    parser.add_argument( '--no-arma', dest = 'use_arma', 
        action = 'store_false', help = 'Use standard convolutions in ConvLSTM.')
    parser.set_defaults(use_arma = True)

    parser.add_argument('--model-size', default = 'small', type = str,
        help = 'The model size (test, small, origin, deep, wide).')

    parser.add_argument('--use-sigmoid', dest = 'use_sigmoid', action = 'store_true',
        help = 'Use sigmoid function at the output of the model.')
    parser.add_argument('--no-sigmoid',  dest = 'use_sigmoid', action = 'store_false',
        help = 'Use output from the last layer as the final output.')
    parser.set_defaults(use_sigmoid = False)

    # parameters of the convolutional operations

    # (moving-average module)
    parser.add_argument('--w-kernel-size', default = 3, type = int, 
        help = 'The kernel size of the moving-average module.')
    parser.add_argument('--w-dilation', default = 1, type = int, 
        help = 'The dilation of convolution of the moving-average module')

    parser.add_argument('--use-bias', dest = "use_bias", 
        action = 'store_true',  help = "Add bias term to the moving-average module.")
    parser.add_argument( '--no-bias', dest = "use_bias", 
        action = 'store_false', help = "Do not add bias to the moving-average module.")
    parser.set_defaults(use_bias = True)

    # (auto-regressive module)
    parser.add_argument('--a-kernel-size', default = 3, type = int, 
        help = 'The kernel size of the autoregressive module.')
    parser.add_argument('--a-padding-mode', default = "circular", type = str,
        help = 'The padding mode of the autoregressive module.')


    ## Results and Models (Output)
    parser.add_argument('--dataset', default = "MNIST", type = str, 
        help = 'The dataset name. (Options: MNIST, KTH, KITTI)')
    
    parser.add_argument('--output-path', default = 'default', type = str,
        help = "The path to the folder storing the outputs (models and results).")

    parser.add_argument('--model-path',  default = 'models',  type = str, 
        help = 'Name of the folder for the models.')
    parser.add_argument('--result-path', default = 'results', type = str,
        help = 'Name of the folder for the results')

    ## Evaluation
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
    parser.set_defaults(eval_best = True)

    # if eval_auto is False (--eval-spec)
    parser.add_argument('--eval-epoch', default = 400, type = int, 
        help = 'Evaluate the model of specified epoch.')

    ## Parameters for visualization 
    parser.add_argument('--range-max', default =  0.9, type = float, 
        help = 'max value of histogram range')
    parser.add_argument('--range-min', default = -0.9, type = float, 
        help = 'min value of histogram range')

    parser.add_argument("--xlabel", default = "coefficient", type = str,
        help = "The xlabel of histogram")
    parser.add_argument("--ylabel", default = "percentage(%)", type = str,
        help = "The ylabel of histogram")
    parser.add_argument("--title", default = "Histogram of the Autoregressive Coefficients", 
        type = str, help = "The title of histogram")
    parser.add_argument("--save-name", default = "test", type = str,
        help = "The name of data file")

    main(parser.parse_args())