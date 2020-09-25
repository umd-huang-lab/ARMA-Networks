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
import utils.config as cf
from utils.AlexNet import *
from utils.VGG import *
from utils.ResNet import *

def main(args):
    ## Devices (CPU, single GPU or multiple GPU)

    # whether to use GPU (or CPU) 
    use_cuda  = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device: ", device)

    # whether to use multi-GPU (or single-GPU)
    multi_gpu = use_cuda and args.multi_gpu and torch.cuda.device_count() > 1
    num_gpus = (torch.cuda.device_count() if multi_gpu else 1) if use_cuda else 0
    print("# of GPUs: ", num_gpus)

    # enable benchmark mode in cudnn
    torch.backends.cudnn.benchmark = args.benchmark


    ## Paths (Dataset, Checkpoints, Statistics and TensorboardX)
    
    # path to the folder of all datasets
    data_path = args.data_path
    if not os.path.exists(data_path): 
        os.makedirs(data_path)

    # path to the folder of specified dataset
    dataset = args.dataset
    assert dataset in ["MNIST", "CIFAR10", "CIFAR100", "ImageNet"], \
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


    # path to the folder of the evaluation statistics
    stats_path = os.path.join(outputs_path, args.stats_path)
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)


    ## Data formats and Dataloaders 
    Dataset     = {    "MNIST": datasets.MNIST, 
                     "CIFAR10": datasets.CIFAR10,
                    "CIFAR100": datasets.CIFAR100}[dataset]

    # data format: batch_size(0) x channels(1) x height(2) x width(3)

    # batch size (0)
    batch_size  = args.batch_size
    
    # image channels, height, width (1, 2, 3)
    image_height   = args.image_height
    image_width    = args.image_width
    image_channels = args.image_channels

    # number of worker for dataloaders 
    num_workers = min(num_gpus, 1) * 4

    # data augmentation
    normalize = transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset])

    if (dataset == "MNIST" and image_channels == 3) or \
     (dataset == "CIFAR10" and image_channels == 1):

        test_transform = transforms.Compose(
            [transforms.Resize((image_height, image_width)),
             transforms.Grayscale(num_output_channels = image_channels),
             transforms.ToTensor(),
             normalize])
    else:

        test_transform = transforms.Compose(
            [transforms.Resize((image_height, image_width)),
             transforms.ToTensor(),
             normalize])


    # dataloader for the test set
    test_dataset  = Dataset(root = data_path, train = False, 
        download = True, transform = test_transform)
    test_loader  = torch.utils.data.DataLoader(test_dataset, 
        batch_size = batch_size, shuffle = False, num_workers=num_workers)


    test_samples = len(test_loader.dataset)
    print("# of test samples: ", test_samples)


    ## Models (Multi-layer Perceptron or Convolutional Neural Networks)
    print("Model: ", args.model_type)

    # multi-layer perceptron
    if args.model_type == "AlexNet":
        model = AlexNet_(                   args.arma, args.dataset, args.rf_init, 
            args.w_kernel_size, args.a_kernel_size)

    elif args.model_type == "VGG":
        model = VGG(args.model_arch,       args.arma, args.dataset, args.rf_init, 
            args.w_kernel_size, args.a_kernel_size)

    elif args.model_type == "ResNet":
        model = ResNet_(args.model_arch,   args.arma, args.dataset, args.rf_init, 
            args.w_kernel_size, args.a_kernel_size)

    # elif args.model_type == "MobileNet":
    #     model = MobileNet_(args.model_arch,args.arma, args.dataset, args.rf_init, 
    #         args.w_kernel_size, args.a_kernel_size)

    # elif args.model_type == "AllConvNet":
    #     model = AllConvNet_(               args.arma, args.dataset, args.rf_init, 
    #         args.w_kernel_size, args.a_kernel_size)

    else:
        raise NotImplementedError

    # move the model to the device (CPU, single-GPU, multi-GPU) 
    model.to(device)
    if multi_gpu: model = nn.DataParallel(model)


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

    checkpoint = torch.load(model_file)
    eval_epoch = checkpoint.get("epoch", args.eval_epoch)
    model.load_state_dict(checkpoint["model_state_dict"])

    # path to the file of the evaluation statistics
    stats_file = "stats_%d" % eval_epoch if (args.stats_file
        == "default") else args.stats_file 
    stats_file = os.path.join(stats_path, stats_file)

    # evaluation on the test set
    model.eval()
    test_nll, test_acc = 0., 0

    with torch.no_grad():
        for (inputs, targets) in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            preds = torch.max(outputs, dim = 1)[1]

            # loss_nll = F.cross_entropy(outputs, targets, reduction = "sum")
            loss_nll = F.nll_loss(outputs, targets)

            test_nll += loss_nll.item() * batch_size
            test_acc += preds.eq(targets.view_as(preds)).sum().item()

    test_nll /= test_samples
    print("Epoch {} (Validation), Avg. NLL: {:.4f}, Acc.: {}/{} ({:.2f}%)".format(
        eval_epoch, test_nll, test_acc, test_samples, 100. * test_acc / test_samples))
    test_acc =  100. * test_acc / test_samples

    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("# of params. = ", num_params)

    # save the statistics 
    np.savez(stats_file, test_acc = test_acc, test_nll_ppg = test_nll)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 
        "Testing ARMA Convolution Networks (ARMA-CNN).")

    ## Data format (Pytorch format)
    # batch size (0) x channels (1) x height (2) x width (3)

    # batch size and log interval (0)
    parser.add_argument("--batch-size",  default = 100,  type = int,
        help = "The batch size for training.")

    # image format: channels (1), height (2), width (3)
    parser.add_argument("--image-height", default =  32, type = int,
        help = "The image height of each sample.")
    parser.add_argument("--image-width",  default =  32, type = int,
        help = "The image width  of each sample.")
    parser.add_argument("--image-channels", default = 3, type = int,
        help = "The number of channels in each sample.")

    ## Device (CPU, single GPU or multiple GPUs)
    
    # whether to use GPU for testing
    parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true',
        help = 'Use GPU for training.')
    parser.add_argument('--no-cuda',  dest = 'use_cuda', action = 'store_false', 
        help = "Do not use GPU for training.")
    parser.set_defaults(use_cuda = True)

    # whether to use multi-GPU for testing 
    parser.add_argument('--multi-gpu',  dest = 'multi_gpu', action = 'store_true',
        help = 'Use multiple GPUs for training.')
    parser.add_argument('--single-gpu', dest = 'multi_gpu', action = 'store_false',
        help = 'Do not use multiple GPU for training.')
    parser.set_defaults(multi_gpu = False)

    parser.add_argument('--use-benchmark', dest = 'benchmark', action = 'store_true',
        help = 'Use benchmark for training.')
    parser.add_argument('--no-benchmark', dest = 'benchmark', action = 'store_false',
        help = 'Do not use benchmark for training.')
    parser.set_defaults(benchmark = False)


    ## Models (MLP, CNN-P, CNN-S)
    parser.add_argument("--model-type", default = "VGG", type = str,
        help = "The type of the model (options: LeNet5/AlexNet/VGG/ResNet).")
    #paras for VGG
    parser.add_argument("--model-arch", default="VGG11", type=str, 
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


    parser.add_argument("--use-bias", dest = "use_bias", action = "store_true", 
        help = "Use bias in all layers of the model.")
    parser.add_argument("--no-bias",  dest = "use_bias", action = "store_false", 
        help = "Do not use bias in all layers of the model.")
    parser.set_defaults(use_bias = True)


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
    parser.add_argument("--stats-path", default = "stats",  type = str,
        help = "The folder for the evaluation statistics.")
    parser.add_argument("--stats-file", default = "default",  type = str, 
        help = "The file name for the evaluation statistics.")

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

    parser.add_argument('--w-kernel-size', default = 3, type = int, 
        help = 'The kernel size of moving average.')
    parser.add_argument('--a-kernel-size', default = 3, type = int, 
        help = 'The kernel size of auto regressives.')

    main(parser.parse_args())