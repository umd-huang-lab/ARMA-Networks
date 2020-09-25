# system utilities
from __future__ import print_function
import os, datetime, argparse

# pytorch utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

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
    print("Device: ", device)

    # whether to use multi-GPU (or single-GPU)
    multi_gpu = use_cuda and args.multi_gpu and torch.cuda.device_count() > 1
    num_gpus = (torch.cuda.device_count() if multi_gpu else 1) if use_cuda else 0
    print("# of GPUs: ", num_gpus)

    # fix the random seed to reproductivity (if --use-seed)
    if not args.use_seed:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if use_cuda: 
            torch.cuda.manual_seed_all(args.random_seed)

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
    else: 
        dataset_path = args.dataset_path

    data_path = os.path.join(data_path, dataset_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # path to the folder of all outputs (for the dataset)
    outputs_path = args.outputs_path
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    outputs_path = os.path.join(outputs_path, dataset_path)
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    # create the name of the current network architecture
    if args.network_name == "default":
        network_name = args.model_type
    else: 
        network_name = args.network_name

    outputs_path = os.path.join(outputs_path, network_name)
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    # create the name (and time stamp) of the current model
    if args.model_name  == "default":
        model_name = str(args.random_seed)
    else: 
        model_name = args.model_name

    if args.model_stamp == "default":
        model_stamp = datetime.datetime.now().strftime("%m%d")
    else: 
        model_stamp = args.model_stamp

    model_name += '_' + model_stamp

    outputs_path = os.path.join(outputs_path, model_name)
    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    # path to the folder of checkpoints
    model_path = os.path.join(outputs_path, args.model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # path to the folder/file of the evaluation statistics
    stats_path = os.path.join(outputs_path, args.stats_path)
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)

    stats_file = os.path.join(stats_path, args.stats_file)

    # path to the folder of the tensorboardX file
    tensorboard_path = os.path.join(outputs_path, args.tensorboard_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    tensorboard_writer = SummaryWriter(tensorboard_path)


    ## Data formats and Dataloaders
    Dataset     = {    "MNIST": datasets.MNIST, 
                     "CIFAR10": datasets.CIFAR10,
                    "CIFAR100": datasets.CIFAR100}[dataset]
    
    

    # data format: batch_size(0) x channels(1) x height(2) x width(3)

    # batch size and the log intervals (0)
    batch_size  = args.batch_size
    log_samples = args.log_samples
    assert log_samples % batch_size == 0, \
        "The argument log_samples should be a multiple of batch_size."
    
    # number of image channels (1), and image height/width (2, 3)
    image_height   = args.image_height
    image_width    = args.image_width
    image_channels = args.image_channels

    # number of worker for dataloaders 
    num_workers = min(num_gpus, 1) * 4

    # preprocessing/transformation of the input images
    image_padding = args.image_padding

    # data augmentation
    normalize = transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset])

    if (dataset == "MNIST" and image_channels == 3) or \
     (dataset == "CIFAR10" and image_channels == 1):

        train_transform = transforms.Compose(
            [transforms.Resize((image_height, image_width)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomCrop((image_height, image_width), padding = image_padding),
             transforms.Grayscale(num_output_channels = image_channels), 
             transforms.ToTensor(),
             normalize])

        valid_transform = transforms.Compose(
            [transforms.Resize((image_height, image_width)),
             transforms.Grayscale(num_output_channels = image_channels),
             transforms.ToTensor(),
             normalize])

    else:

        train_transform = transforms.Compose(
            [transforms.Resize((image_height, image_width)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomCrop((image_height, image_width), padding = image_padding), 
             transforms.ToTensor(),
             normalize])

        valid_transform = transforms.Compose(
            [transforms.Resize((image_height, image_width)),
             transforms.ToTensor(),
             normalize])


    if dataset == "ImageNet":
        train_dataset = datasets.ImageFolder(root = os.path.join(data_path,'ILSVRC2012/train'),
            transform = train_transform)

        valid_dataset = datasets.ImageFolder(root = os.path.join(data_path,'ILSVRC2012/train'),
            transform = train_transform)

        test_dataset = datasets.ImageFolder(root = os.path.join(data_path,'ILSVRC2012/val'), 
            transform = valid_transform)
    else:
        # training, validation and test sets
        train_dataset = Dataset(root = data_path, train = True, 
            download = True, transform = train_transform)

        valid_dataset = Dataset(root = data_path, train = True, 
            download = True, transform = valid_transform)

        test_dataset  = Dataset(root = data_path, train = False, 
            download = True, transform = valid_transform)

    # training, validation and test dataloader
    num_samples = len(train_dataset)
    split = int(np.floor(args.train_ratio * num_samples))

    indices = list(range(num_samples))
    train_sampler = SubsetRandomSampler(indices[:split])
    valid_sampler = SubsetRandomSampler(indices[split:])

    train_loader = torch.utils.data.DataLoader(train_dataset, 
        batch_size = batch_size, sampler = train_sampler, num_workers = num_workers)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
        batch_size = batch_size, sampler = valid_sampler, num_workers = num_workers)

    test_loader  = torch.utils.data.DataLoader(test_dataset, 
        batch_size = batch_size, shuffle = False,         num_workers=num_workers)

    train_samples = len(train_loader) * batch_size
    print("# of training samples: ", train_samples)

    valid_samples = len(valid_loader) * batch_size
    print("# of validation samples: ", valid_samples)

    test_samples =  len(test_loader)  * batch_size
    print("# of test samples: ", test_samples)


    ## Models (Multi-layer Perceptron or Convolutional Neural Networks)
    print("Model: ", args.model_type)

    # multi-layer perceptron
    if args.model_type == "AlexNet":
        model = AlexNet_(                  args.arma, args.dataset, args.rf_init, 
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

    ## Main script for learning and evaluation 
    epoch_num  = args.epoch_num
    save_epoch = args.save_epoch

    valid_acc_ = np.zeros(epoch_num, dtype = np.float)
    valid_nll_ = np.zeros(epoch_num, dtype = np.float)

    # initial learning rate for the optimizer
    learning_rate = args.learning_rate

    # recover the model to resume training (if required)
    if args.start_begin:
        model_file = None
        start_epoch, total_samples = 0, 0
        min_epoch, min_valid_nll = 0, float("inf")

    else: 
        model_file = os.path.join(model_path, 'training_last.pt' 
            if args.start_last else "training_%d.pt" % args.start_epoch)
        assert os.path.exists(model_file), \
            "The specified model is not found in the folder."
        
    if model_file is not None:
        checkpoint = torch.load(model_file)

        # model parameters
        model.load_state_dict(checkpoint["model_state_dict"])

        # training progress
        start_epoch   = checkpoint["epoch"]
        total_samples = checkpoint["total_samples"]

        # best model and its negative likelihood
        min_epoch = checkpoint["min_epoch"]
        min_valid_nll = checkpoint["min_valid_nll"]

        # learning rate
        learning_rate = checkpoint["learning_rate"]

    # optimizer and corresponding scheduler
    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, 
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
         optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


    decay_epoch = list(map(int, args.decay_epoch.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_epoch, 
        gamma = args.decay_rate)


    for epoch in range(start_epoch, epoch_num):
        learning_rate = optimizer.param_groups[0]['lr']
        tensorboard_writer.add_scalar('lr', learning_rate, epoch + 1)
        print("Epoch %d, Learning rate: %2f" % (epoch + 1, learning_rate))

        ## Phase 1: Learning on training set
        model.train()
        train_nll, train_acc = 0, 0.

        # initialize the statistics
        samples = 0
        NLL, ACC = 0., 0

        for (inputs, targets) in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            total_samples += batch_size
            samples += batch_size

            # predict the outputs with probabilistic propagation
            outputs = model(inputs)
            preds = torch.max(outputs, dim = 1)[1]

            loss = F.nll_loss(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate the statistics
            NLL  += loss.item() * batch_size
            ACC  += preds.eq(targets.view_as(preds)).sum().item()

            if samples % args.log_samples == 0:

                print("Epoch: {} [{}/{} ({:.1f}%)], Avg. NLL: {:.4f}, Acc: {}/{}".format(
                    epoch + 1, samples, train_samples, 100. * samples / train_samples,
                    NLL / args.log_samples, ACC, args.log_samples))

                train_nll += NLL
                train_acc += ACC
                NLL /= log_samples
                ACC /= log_samples

                tensorboard_writer.add_scalar('train_nll',   NLL, total_samples)
                tensorboard_writer.add_scalar('train_acc',   ACC, total_samples)

                # reinitialize the statistics
                NLL, ACC = 0., 0

        train_nll /= train_samples
        print("Epoch: {} (Training), Avg. NLL: {:.4f}, Acc.: {}/{} ({:.2f}%)".format(
            epoch + 1, train_nll, train_acc, train_samples, 100. * train_acc / train_samples))
        train_acc =  100. * train_acc / train_samples 

        ## Phase 2: Evaluation on the validation set
        model.eval()
        valid_nll, valid_acc = 0., 0

        with torch.no_grad():
            for (inputs, targets) in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                preds = torch.max(outputs, dim = 1)[1]

                loss = F.nll_loss(outputs, targets)

                valid_nll += loss.item() * batch_size
                valid_acc += preds.eq(targets.view_as(preds)).sum().item()

        valid_nll /= valid_samples
        print("Epoch {} (Validation), Avg. NLL: {:.4f}, Acc.: {}/{} ({:.2f}%)".format(
            epoch + 1, valid_nll, valid_acc, valid_samples, 100. * valid_acc / valid_samples))
        valid_acc =  100. * valid_acc / valid_samples

        valid_acc_[epoch] = valid_acc
        valid_nll_[epoch] = valid_nll

        tensorboard_writer.add_scalar('valid_acc', valid_acc, epoch + 1)
        tensorboard_writer.add_scalar('valid_nll', valid_nll, epoch + 1)

        if (epoch + 1) % save_epoch == 0:
            model.eval()
            test_nll, test_acc = 0., 0

            with torch.no_grad():
                for (inputs, targets) in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    preds = torch.max(outputs, dim = 1)[1]

                    loss = F.nll_loss(outputs, targets)

                    test_nll += loss.item() * batch_size
                    test_acc += preds.eq(targets.view_as(preds)).sum().item()

            print("Epoch {} (Testing), Avg. NLL: {:.4f}, Acc.: {}/{} ({:.2f}%)".format(
                epoch + 1, test_nll / test_samples, test_acc, test_samples, 100. * test_acc / test_samples))


        ## Phase 3: Logging the learning curves and checkpoints
        if args.rate_decay: scheduler.step()

        # update the best model so far
        if valid_nll < min_valid_nll:
            min_epoch, min_valid_nll = epoch + 1, valid_nll

        # save the currrent model as a checkpoint
        checkpoint_info = {
            "epoch": epoch + 1, "total_samples": total_samples, # training progress 
            "min_epoch": min_epoch, "min_valid_nll": min_valid_nll, # best model and loss
            "learning_rate": optimizer.param_groups[0]['lr'], # current learning rate 
            "model_state_dict": model.state_dict() # model parameters
        }

        if (epoch + 1) == min_epoch:
            torch.save(checkpoint_info, os.path.join(model_path, 'training_best.pt'))

        if (epoch + 1) % save_epoch == 0:    
            torch.save(checkpoint_info, os.path.join(model_path, 'training_%d.pt' % (epoch + 1)))

        torch.save(checkpoint_info, os.path.join(model_path, 'training_last.pt'))

    # save the statistics 
    np.savez(stats_file, valid_acc = valid_acc_, valid_nll = valid_nll_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 
        "Training ARMA Convolution Neural Networks (ARMA-CNN).")

    ## Data format (Pytorch format)
    # batch size (0) x channels (1) x height (2) x width (3)

    # batch size and log interval (0)
    parser.add_argument("--train-ratio", default = 0.9, type=float,
        help="The ratio of training samples in the .")
    parser.add_argument("--log-samples", default = 4992, type = int,
        help = "Log the learning curve every log_samples.")
    parser.add_argument("--batch-size",  default = 128,  type = int,
        help = "The batch size for training.")

    # image format: channels (1), height (2), width (3)
    parser.add_argument("--image-height", default =  32, type = int,
        help = "The image height of each sample.")
    parser.add_argument("--image-width",  default =  32, type = int,
        help = "The image width  of each sample.")
    parser.add_argument("--image-channels", default = 3, type = int,
        help = "The number of channels in each sample.")
    
    # data augmentation (in learning phase)
    parser.add_argument("--image-padding",  default = 4, type = int,
        help = "The number of padded pixels along height/width.") 

    ## Device (CPU, single GPU or multiple GPUs)
    
    # whether to use GPU for training
    parser.add_argument('--use-cuda', dest = 'use_cuda', action = 'store_true',
        help = 'Use GPU for training.')
    parser.add_argument('--no-cuda',  dest = 'use_cuda', action = 'store_false', 
        help = "Do not use GPU for training.")
    parser.set_defaults(use_cuda = True)

    # whether to use multi-GPU for training 
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


    # random seed for reproducibility
    parser.add_argument('--use-seed', dest = 'use_seed', action = 'store_true', 
        help = 'Fix the random seed to reproduce the model.')
    parser.add_argument('--no-seed',  dest = 'use_seed', action = 'store_false', 
        help = 'Randomly choose the random seed.')
    parser.set_defaults(use_seed = True)
    parser.add_argument('--random-seed', default = 0, type = int, 
        help = 'The random seed number (to reproduce the model).')


    ## Models (LeNet5, AlexNet, VGG)
    parser.add_argument("--model-type", default = "VGG", type = str,
                        help = "The type of the model (options: LeNet5/AlexNet/VGG/ResNet).")
    parser.add_argument("--model-arch", default="VGG11", type=str, 
                        help="The depth of a specified Net.")
    parser.add_argument("--no-batch-norm", dest="barch_norm", action="store_false",
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


    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')

    parser.add_argument("--use-bias", dest = "use_bias", action = "store_true", 
        help = "Use bias in all layers of the model.")
    parser.add_argument("--no-bias",  dest = "use_bias", action = "store_false", 
        help = "Do not use bias in all layers of the model.")
    parser.set_defaults(use_bias = True)

    ## Paths (Data, Checkpoints, Results and TensorboardX)

    # inputs:  data
    parser.add_argument("--dataset", default = "CIFAR10", type = str,
        help = "The dataset used for training (options: MNIST/CIFAR10/CIFAR100).")
    parser.add_argument("--data-path", default = "../data", type = str,
        help = "The path to the folder stroing the data.")
    parser.add_argument("--dataset-path", default = "default", type = str,
        help = "The folder for the dataset.")

    # outputs: checkpoints, statistics and tensorboard
    parser.add_argument("--outputs-path", default = "../data/outputs", type = str,
        help = "The path to the folder storing outputs from training.")
    parser.add_argument("--network-name", default = "default", type = str, 
        help = "The architecture model (to create the folder).")

    parser.add_argument("--model-name",  default = "default", type = str,
        help = "The model name (to create the folder).")
    parser.add_argument("--model-stamp", default = "default", type = str,
        help = "The time stamp of the model (as a suffix to the its name).")

    parser.add_argument("--model-path", default = "models", type = str,
        help = "The folder for all checkpoints in training.")
    parser.add_argument("--stats-path", default = "stats",  type = str,
        help = "The folder for the evaluation statistics.")
    parser.add_argument("--stats-file", default = "curve",  type = str, 
        help = "The file name for the learning curve.")
    parser.add_argument('--tensorboard-path', default = 'tensorboard', type = str,
        help = 'The folder for the tensorboardX files.')

    ## Hyperparameters for learning
    parser.add_argument("--epoch-num", default = 300, type = int,
        help = "The total number of epochs for training.")
    parser.add_argument("--save-epoch", default = 100, type = int,
        help = "The interval of epochs to save a checkpoint.")

    parser.add_argument('--start-begin', dest = 'start_begin', action = 'store_true', 
        help = 'Start training a new model from the beginning.')
    parser.add_argument('--start-exist', dest = 'start_begin', action = 'store_false',
        help = 'Resume training from an existing model.')
    parser.set_defaults(start_begin = True)

    # if start_begin is False (--start-exist)
    parser.add_argument('--start-last', dest = 'start_last', action = 'store_true', 
        help = 'Resume training from the last available model.')
    parser.add_argument('--start-spec', dest = 'start_last', action = 'store_false', 
        help = 'Resume training from the model of the specified epoch.')
    parser.set_defaults(start_last = True)

    # if start_last is False (--start-spec)
    parser.add_argument('--start-epoch', default = 0, type = int, 
        help = 'The number of epoch to resume training.')

    # learning rate scheduling
    parser.add_argument("--learning-rate", default = 0.05, type = float,
        help = "Initial learning rate of the optimizer.")

    parser.add_argument("--learning-rate-decay", dest = "rate_decay", action = 'store_true',
        help = "Learning rate is decayed during training.")
    parser.add_argument("--learning-rate-fixed", dest = "rate_decay", action = 'store_false', 
        help = "Learning rate is fixed during training.")
    parser.set_defaults(rate_decay = True)

    # if rate_decay is True (--learning-rate-decay)
    parser.add_argument("--decay-epoch", default = "30", type = str,
        help = "The learning rate is decayed by decay_rate every decay_epoch.")
    parser.add_argument("--decay-rate", default = 0.5, type = float,
        help = "The learning rate is decayed by decay_rate every decay_epoch.")

    parser.add_argument('--w-kernel-size', default = 3, type = int, 
        help = 'The kernel size of moving average.')
    parser.add_argument('--a-kernel-size', default = 3, type = int, 
        help = 'The kernel size of auto regressives.')


    main(parser.parse_args())
