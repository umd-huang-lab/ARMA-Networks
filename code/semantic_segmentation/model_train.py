import argparse, os, datetime
from solver import Solver
from dataloader import get_loader
from tensorboardX import SummaryWriter
import utils.ISBI2012Data as ISBI
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import random
import torch

torch.backends.cudnn.benchmark = True

def main(config):

    assert config.model_type in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net']

    if config.use_seed:
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if config.use_cuda: 
            torch.cuda.manual_seed_all(config.random_seed)


    #=========================================================================================
    #data_path = '../data/dataset'
    data_path = config.data_path
    if not os.path.exists(data_path): os.makedirs(data_path)
    dataset_path = config.dataset if config.dataset_path == "default" else config.dataset_path

    #outputs_path = '../data/outputs/dataset/model_type'
    outputs_path = os.path.join(data_path, config.outputs_path)
    if not os.path.exists(outputs_path): os.makedirs(outputs_path)
    outputs_path = os.path.join(outputs_path, dataset_path)
    if not os.path.exists(outputs_path): os.makedirs(outputs_path)
    network_name = config.model_type if config.network_name == "default" else config.network_name
    outputs_path = os.path.join(outputs_path,network_name) 
    if not os.path.exists(outputs_path): os.makedirs(outputs_path)

    model_name = config.model_name
    if config.model_name == "default":
        arma = 'arma' if config.use_arma else 'no-arma'
        model_stamp = datetime.datetime.now().strftime("%m%d") if config.model_stamp == "default" else config.model_stamp
        model_name = ('%s-%s-seed%d-%.5f-%.4f' %(config.model_type, arma, config.random_seed, config.lr, config.augmentation_prob))
        model_name += '_' + model_stamp


    #outputs_path = '../data/outputs/dataset/model_type/model_name'
    outputs_path = os.path.join(outputs_path, model_name)
    if not os.path.exists(outputs_path): os.makedirs(outputs_path)
    #model_path = '../data/outputs/dataset/model_type/model_name/models'
    model_path = os.path.join(outputs_path, config.model_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    #model_path = '../data/outputs/dataset/model_type/model_name/stats'
    stats_path = os.path.join(outputs_path, config.stats_path)
    if not os.path.exists(stats_path): os.makedirs(stats_path)
    #result_path = '../data/outputs/dataset/model_type/model_name/results'
    result_path = os.path.join(outputs_path, config.result_path)
    if not os.path.exists(result_path): os.makedirs(result_path)
    #tensorboard_path = '../data/outputs/dataset/model_type/model_name/tensorboard'
    tensorboard_path = os.path.join(outputs_path, config.tensorboard_path)
    if not os.path.exists(tensorboard_path): os.makedirs(tensorboard_path)
    tensorboard_writer = SummaryWriter(tensorboard_path)

    config.model_path = model_path
    config.result_path = result_path
    config.tensorboard = tensorboard_writer
    config.model_name = model_name


    print("Mode      : ", config.mode)
    print("Model Type: ", config.model_type)
    print("Model Name: ", config.model_name)
    print("Dataset   : ", config.dataset)
    #=========================================================================================

    # print(config)
    if config.dataset == "ISIC": 
        train_loader = get_loader(image_path=config.train_path,
                                image_size=config.image_size,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode='train',
                                augmentation_prob=config.augmentation_prob)
        valid_loader = get_loader(image_path=config.valid_path,
                                image_size=config.image_size,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode='valid',
                                augmentation_prob=0. ,
                                isshuffle=False)
        test_loader = get_loader(image_path=config.test_path,
                                image_size=config.image_size,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                mode='test',
                                augmentation_prob=0.)

    if config.dataset == "ISBI":

        train_dataset = ISBI.ISBIDataset(
            os.path.join(config.data_path, "ISBI2012/Train-Volume/train-volume-*.tif"), os.path.join(config.data_path, "ISBI2012/Train-Labels/train-labels-*.tif"),
            length=22, is_pad=True, evaluate=False, totensor=True)

        valid_dataset = ISBI.ISBIDataset(
            os.path.join(config.data_path, "ISBI2012/Val-Volume/train-volume-*.tif"), os.path.join(config.data_path, "ISBI2012/Val-Labels/train-labels-*.tif"),
            length=8, is_pad=True, evaluate=True, totensor=True)

        test_dataset = ISBI.ISBIDataset(
            os.path.join(config.data_path, "ISBI2012/Test-Volume/test-volume-*.tif"), os.path.join(config.data_path, "ISBI2012/Test-Volume/test-volume-*.tif"),
            length=30, is_pad=True, evaluate=True, totensor=True)

        num_samples = len(train_dataset)
        split = int(np.floor(.7 * num_samples))

        indices = list(range(num_samples))
        train_sampler = SubsetRandomSampler(indices[:split])
        valid_sampler = SubsetRandomSampler(indices[split:])

        train_loader = torch.utils.data.DataLoader(train_dataset,
                        sampler = train_sampler, batch_size=config.batch_size,
                        num_workers=config.num_workers, pin_memory=True)

        valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                        sampler=valid_sampler, batch_size=1, 
                        num_workers=config.num_workers, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(test_dataset, 
                        batch_size=1, num_workers=config.num_workers, pin_memory=True)



    solver = Solver(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--steps',      type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch',     type=int, default = 3)
    parser.add_argument('--output_ch',  type=int, default = 1)

    parser.add_argument('--num_epochs', type=int, default = 200)
    parser.add_argument('--num_epochs_decay', type = int, default = 70)
    
    parser.add_argument('--batch_size', type = int, default = 2)
    parser.add_argument('--num_workers',type = int, default = 8)

    parser.add_argument('--lr',    type=float, default = 0.001)
    parser.add_argument('--beta1', type=float, default = 0.5)    # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default = 0.999)  # momentum2 in Adam

    parser.add_argument('--augmentation_prob', type = float, default = 0.2)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode',       type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')

    parser.add_argument('--use-arma', dest = 'use_arma', action = 'store_true',  help = 'Use ARMA convolutions in the models.')
    parser.add_argument( '--no-arma', dest = 'use_arma', action = 'store_false', help = 'Use standard convolutions in the models.')
    parser.set_defaults(use_arma = False)


    parser.add_argument('--train_path', type=str, default='../data/ISIC/train/')
    parser.add_argument('--valid_path', type=str, default='../data/ISIC/valid/')
    parser.add_argument('--test_path',  type=str, default='../data/ISIC/test/')

    parser.add_argument("--dataset",      default = "ISIC",    type = str, help = "ISIC/ISBI")
    parser.add_argument("--data-path",    default = "../data", type = str, help = "The path to data.")
    parser.add_argument("--dataset-path", default = "default", type = str, help = "The folder for the dataset.")
    parser.add_argument("--outputs-path", default = "outputs", type = str, help = "The path to outputs.")
    parser.add_argument("--network-name", default = "default", type = str, help = "The architecture model.")
    parser.add_argument("--model-name",   default = "default", type = str, help = "The model name.")
    parser.add_argument("--model-stamp",  default = "default", type = str, help = "The time stamp of the model.")
    parser.add_argument("--model-path",   default = "models",  type = str, help = "The folder for checkpoints.")
    parser.add_argument("--stats-path",   default = "stats",   type = str, help = "The folder for the evaluation statistics.")
    parser.add_argument("--result-path",  default = "results", type = str, help = "The folder for the result")
    parser.add_argument("--stats-file",   default = "curve",   type = str, help = "The file name for the learning curve.")
    parser.add_argument('--use-cuda',     dest = 'use_cuda', action = 'store_true',  help = 'Use GPU for training.')
    parser.add_argument('--no-cuda',      dest = 'use_cuda', action = 'store_false', help = "Do not use GPU for training.")
    parser.add_argument('--tensorboard-path', default = 'tensorboard', type = str,   help = 'The folder for the tensorboardX files.')
    parser.add_argument('--tensorboard',  type=str, default=None)
    parser.set_defaults(use_cuda = True)

    parser.add_argument('--cuda_idx', type=int, default=1)
    parser.add_argument('--random-seed', default = 0, type = int, help = 'The random seed number.')
    parser.add_argument('--use-seed', dest = 'use_seed', action = 'store_true', help = 'random seed.')
    parser.add_argument('--no-seed',  dest = 'use_seed', action = 'store_false', help = 'Randomly choose the random seed.')
    parser.set_defaults(use_seed = True)

    parser.add_argument("--padding",  dest = "padding", action = "store_false", help = "The number of padded pixels along height/width.")



    config = parser.parse_args()
    main(config)
