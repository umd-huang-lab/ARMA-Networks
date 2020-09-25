# system modules
import os, argparse

# basic pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# computer vision/image processing modules 
import torchvision
import skimage

# math/probability modules
import random
import numpy as np

# custom utilities
from tensorboardX import SummaryWriter 
from dataloader import MNIST_Dataset
from utils.convlstmnet import ConvLSTMNet


def main(args):
    ## Data format: batch_size(0) x time_steps(1) x 
    #  img_height(2) x img_width(3) x channels(4) 

    # batch size (0)
    batch_size  = args.batch_size
    log_samples = args.log_samples
    assert log_samples % batch_size == 0, \
        "The argument log_samples should be a multiple of batch_size."

    # frame split (1)
    input_frames  = args.input_frames
    future_frames = args.future_frames
    output_frames = args.output_frames

    total_frames = input_frames + future_frames 

    # frame format (2, 3, 4)
    img_height   = args.img_height
    img_width    = args.img_width

    img_channels = args.img_channels
    assert img_channels == 1 or img_channels == 3, \
        "The number of channels is either 1 (gray) or 3 (colored)."

    # whether the frames are colored or grayscale
    img_colored = (img_channels == 3) 


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

    # fix the random seed to reproductivity (if needed) 
    if not args.use_seed:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        if use_cuda: 
            torch.cuda.manual_seed_all(args.random_seed)

    # size of the Conv-LSTM network
    if args.model_size == "origin":  # 12-layers
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
        # model architecture
        layers_per_block, hidden_channels, skip_stride = skip_stride, 
        # input/output interfaces
        input_channels = args.img_channels, output_sigmoid = args.use_sigmoid,
        input_height = args.img_height, input_width = args.img_width, 
        # non-local blocks
        non_local = args.use_non_local, pairwise_function = args.pairwise_function,
        use_norm = args.use_norm, sub_sampling = args.use_sub_sample, 
        # convolutional layers
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


    ## Dataset Preparation (Moving-MNIST)
    dataset = args.dataset
    Dataset = {"MNIST": MNIST_Dataset}[dataset]

    # path to the dataset folder
    if args.data_path == "default":
        DATA_DIR = {"MNIST": "moving-mnist"}[dataset]
        DATA_DIR = os.path.join("../datasets", DATA_DIR)
    else: # if args.data_path != "default":
        DATA_DIR = args.data_path

    assert os.path.exists(DATA_DIR), \
        "The dataset folder does not exist."

    # number of workers for the dataloaders
    num_workers = 5 * max(num_gpus, 1)

    # dataloader for the training set
    train_data_path = os.path.join(DATA_DIR, args.train_data_file)
    assert os.path.exists(train_data_path), \
        "The training data does not exist."

    train_data = Dataset({"path": train_data_path, "unique_mode": False,
        "num_frames": total_frames, "num_samples": args.train_samples, 
        "height": img_height, "width": img_width, "channels": img_channels})

    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size = batch_size, shuffle = True, 
        num_workers = num_workers, drop_last = True)

    train_size = len(train_data_loader) * batch_size
    print("# of training samples:", train_size)

    # dataloaer for the valiation set 
    valid_data_path = os.path.join(DATA_DIR, args.valid_data_file)
    assert os.path.exists(valid_data_path), \
        "The validation set does not exist."

    valid_data = Dataset({"path": valid_data_path, "unique_mode": True,
        "num_frames": total_frames, "num_samples": args.valid_samples,
        "height": img_height, "width": img_width, "channels": img_channels})

    valid_data_loader = torch.utils.data.DataLoader(
        valid_data, batch_size = batch_size, shuffle = False,
        num_workers = num_workers, drop_last = True)

    valid_size = len(valid_data_loader) * batch_size
    print("# of valiation samples:", valid_size)


    ## Outputs (Models and Results)
    if args.output_path == "default":
        OUTPUT_DIR = {"MNIST": "./moving-mnist"}[dataset]
    else: # if args.output_path != "default":
        OUTPUT_DIR = args.output_path

    OUTPUT_DIR = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
  
    # path to the models folder
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # path to the results folder
    RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # path to the validation images folder
    RESULT_IMG = os.path.join(RESULT_DIR, "valid_images")
    if not os.path.exists(RESULT_IMG):
        os.makedirs(RESULT_IMG)

    # path to the tensorboard folder 
    RESULT_TBW = os.path.join(RESULT_DIR, "tensorboardX")
    if not os.path.exists(RESULT_TBW):
        os.makedirs(RESULT_TBW)
    tensorboard_writer = SummaryWriter(RESULT_TBW)


    ## Hyperparameters for learning algorithm

    # loss function for training
    if args.loss_function == "l1":
        loss_func = lambda pred, origin: \
            F.l1_loss( pred, origin, reduction = "mean")
    elif args.loss_function == "l2":
        loss_func = lambda pred, origin: \
            F.mse_loss(pred, origin, reduction = "mean")
    elif args.loss_function == "l1l2":
        loss_func = lambda pred, origin: \
            (F.l1_loss(pred, origin, reduction = "mean") +
            F.mse_loss(pred, origin, reduction = "mean"))
    else:
        raise NotImplementedError 

    # log the loss for decay_log_epochs for rate/ratio decay
    decay_log_epochs = args.decay_log_epochs

    # scheduled sampling ratio (the ratio is decayed linearly)
    scheduled_sampling_ratio = 0
    teacher_forcing = args.teacher_forcing
    if teacher_forcing:
        ssr_decay_mode  = False
        ssr_decay_start = args.ssr_decay_start
        ssr_decay_epoch = args.ssr_decay_epoch
        ssr_decay_ratio = args.ssr_decay_ratio
        scheduled_sampling_ratio = 1

    # learning rate (the rate is decayed exponentially)
    # learning rate is decayed after scheduled sampling
    learning_rate  = args.learning_rate
    learning_rate_decay = args.learning_rate_decay
    if learning_rate_decay:
        lr_decay_mode  = False
        lr_decay_start = args.num_epochs \
            if teacher_forcing else args.lr_decay_start
        lr_decay_epoch = args.lr_decay_epoch 
        lr_decay_rate  = args.lr_decay_rate

    # gradient clipping for gradient descent 
    gradient_clipping = args.gradient_clipping
    if gradient_clipping:
        clipping_threshold = args.clipping_threshold

    # start from beginning / resume training from checkpoint
    if not args.start_begin:
        if args.start_last:
            MODEL_FILE = os.path.join(MODEL_DIR, 'training_last.pt')
        else: # if args.start_spec:
            MODEL_FILE = os.path.join(MODEL_DIR, "training_%d.pt" % args.start_epoch)
        assert os.path.exists(MODEL_FILE), \
            "The specified model is not found in the folder."
    else: # if args.start_begin:
        MODEL_FILE = None

    if MODEL_FILE is not None:
        checkpoint = torch.load(MODEL_FILE)

        # recover the model parameters (weights)
        model.load_state_dict(checkpoint["model_state_dict"])

        # recover the information of training progress
        start_epoch = checkpoint.get("epoch", args.start_epoch)
        total_samples = checkpoint.get("total_samples", start_epoch * train_size)

        # recover the epoch/loss of the best model
        min_epoch = checkpoint.get("min_epoch", 0)
        min_loss  = checkpoint.get("min_loss", float("inf"))

        # recover the scheduled sampling ratio
        ssr_decay_mode = checkpoint.get("ssr_decay_mode", False)
        scheduled_sampling_ratio = checkpoint.get("scheduled_sampling_ratio", scheduled_sampling_ratio)
        
        # recover the learning rate
        lr_decay_mode = checkpoint.get("lr_decay_mode", False)
        learning_rate = checkpoint.get("learning_rate", learning_rate)
    else:
        start_epoch, total_samples = 0, 0
        min_epoch, min_loss = 0, float("inf")


    ## Main script for training and validation 
    num_epochs = args.num_epochs # total num of training epochs
    save_epoch = args.save_epoch # save the model per save_epoch 

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(start_epoch, num_epochs):
        # log the hyperparameters of learning sheduling 
        tensorboard_writer.add_scalar('lr',  optimizer.param_groups[0]['lr'], epoch + 1)
        tensorboard_writer.add_scalar('ssr', scheduled_sampling_ratio, epoch + 1)

        ## Phase 1: Learning on the training set
        samples, LOSS = 0, 0.

        model.train()
        for frames in train_data_loader:
            total_samples += batch_size
            samples += batch_size

            # 5-th order: batch_size(0) x total_frames(1) x channels(2) x height(3) x width(4) 
            frames = frames.permute(0, 1, 4, 2, 3).to(device)

            inputs = frames[:, :-1] if teacher_forcing else frames[:, :input_frames] 
            origin = frames[:, -output_frames:]

            pred = model(inputs, 
                input_frames  =  input_frames, 
                future_frames = future_frames, 
                output_frames = output_frames, 
                teacher_forcing = teacher_forcing, 
                scheduled_sampling_ratio = scheduled_sampling_ratio)

            # compute the loss function
            loss  = loss_func(pred, origin)
            LOSS += loss.item() * batch_size

            # compute the backpropagation
            optimizer.zero_grad()
            loss.backward()

            # compute the graident norm
            if samples % log_samples == 0:
                grad_norm = 0.
                for param in model.parameters():
                    param_grad_norm = param.grad.data.norm(2)
                    grad_norm += param_grad_norm.item() ** 2
                grad_norm = grad_norm ** (0.5)
                tensorboard_writer.add_scalar("grad", grad_norm, total_samples)
                print('-- grad: {}'.format(grad_norm))
            
            # gradient clipping and stochastic gradient descent
            if gradient_clipping: 
                nn.utils.clip_grad_norm_(model.parameters(), clipping_threshold)
            optimizer.step()

            if samples % log_samples == 0:
                LOSS /= log_samples
                print('Epoch: {}/{}, Training: {}/{}, Loss: {}'.format(
                    epoch + 1, num_epochs, samples, train_size, LOSS))
                tensorboard_writer.add_scalar('LOSS', LOSS, total_samples)
                LOSS = 0.

                # compute the gradient norm after clipping
                if gradient_clipping:
                    grad_norm = 0.
                    for param in model.parameters():
                        param_grad_norm = param.grad.data.norm(2)
                        grad_norm += param_grad_norm.item() ** 2
                    grad_norm = grad_norm ** (0.5)
                    tensorboard_writer.add_scalar("grad_clip", grad_norm, total_samples)
                    print('-- grad clip: {}'.format(grad_norm))

        # adjust the learning rate of the optimizer 
        if learning_rate_decay and lr_decay_mode and (epoch + 1) % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_rate

        # adjust the scheduled sampling ratio
        if teacher_forcing and ssr_decay_mode and (epoch + 1) % ssr_decay_epoch == 0:
            scheduled_sampling_ratio = max(scheduled_sampling_ratio - ssr_decay_ratio, 0) 

        ## Phase 2: Evaluation on the validation set
        model.eval()
        samples, LOSS = 0, 0.
        
        MSE  = [0.] * future_frames
        PSNR = [0.] * future_frames
        SSIM = [0.] * future_frames

        with torch.no_grad():    
            for frames in valid_data_loader:
                samples += batch_size

                # 5-th order: batch_size x total_frames x channels x height x width 
                frames = frames.permute(0, 1, 4, 2, 3).to(device)

                inputs = frames[:,  :input_frames]
                origin = frames[:, -output_frames:]

                pred = model(inputs, 
                    input_frames  =  input_frames, 
                    future_frames = future_frames, 
                    output_frames = output_frames, 
                    teacher_forcing = False)

                loss =  loss_func(pred, origin)
                LOSS += loss.item() * batch_size

                # clamp the output to [0, 1]
                pred = torch.clamp(pred, min = 0, max = 1)

                # save the first sample for each batch to the tensorboard
                if samples % log_samples == 0:
                    input_0 = inputs[0, -future_frames:] if input_frames >= future_frames \
                        else torch.cat([torch.zeros(future_frames - input_frames, 
                            img_channels, img_height, img_width, device = device), inputs[0]], dim = 0)

                    origin_0 = origin[0, -future_frames:]
                    pred_0   = pred[0,   -future_frames:]

                    img = torchvision.utils.make_grid(torch.cat(
                        [input_0, origin_0, pred_0], dim = 0), nrow = future_frames)

                    tensorboard_writer.add_image("img_results", img, epoch + 1)

                    RESULT_FILE = os.path.join(RESULT_IMG, "cmp_%d_%d.jpg" % (epoch + 1, samples))
                    torchvision.utils.save_image(img, RESULT_FILE)

                # accumlate the statistics
                origin = origin.permute(0, 1, 3, 4, 2).cpu().numpy()
                pred   =   pred.permute(0, 1, 3, 4, 2).cpu().numpy()
                for i in range(batch_size):
                    for t in range(-future_frames, 0):
                        origin_, pred_ = origin[i, t], pred[i, t]
                        if not img_colored:
                            origin_ = np.squeeze(origin_, axis = -1)
                            pred_   = np.squeeze(pred_,   axis = -1)

                        MSE[t]  += skimage.measure.compare_mse( origin_, pred_)
                        PSNR[t] += skimage.measure.compare_psnr(origin_, pred_)
                        SSIM[t] += skimage.measure.compare_ssim(origin_, pred_, multichannel = img_colored)

        LOSS /= valid_size
        tensorboard_writer.add_scalar("LOSS(val)", LOSS, epoch + 1)

        for t in range(future_frames):
            MSE[t]  /= valid_size
            PSNR[t] /= valid_size
            SSIM[t] /= valid_size
            tensorboard_writer.add_scalar("MSE_%d"  % (t + 1), MSE[t],  epoch + 1)
            tensorboard_writer.add_scalar("PSNR_%d" % (t + 1), PSNR[t], epoch + 1)
            tensorboard_writer.add_scalar("SSIM_%d" % (t + 1), SSIM[t], epoch + 1)

        # compute the average statistics
        MSE_AVG  = sum(MSE)  / future_frames
        PSNR_AVG = sum(PSNR) / future_frames
        SSIM_AVG = sum(SSIM) / future_frames
        tensorboard_writer.add_scalar("MSE(val)",  MSE_AVG,  epoch + 1)
        tensorboard_writer.add_scalar("PSNR(val)", PSNR_AVG, epoch + 1)
        tensorboard_writer.add_scalar("SSIM(val)", SSIM_AVG, epoch + 1)

        print("Epoch {}, LOSS: {}, MSE: {} (x1e-3); PSNR: {}, SSIM: {}".format(
            epoch + 1, LOSS, 1e3 * MSE_AVG, PSNR_AVG, SSIM_AVG))

        # automatic scheduling of (1) learning rate and (2) scheduled sampling ratio
        if not ssr_decay_mode and epoch > ssr_decay_start and epoch > min_epoch + args.decay_log_epochs:
            lr_decay_start = epoch + args.lr_decay_start
            ssr_decay_mode = True
            
        if not lr_decay_mode and epoch > lr_decay_start and epoch > min_epoch + args.decay_log_epochs:
            lr_decay_mode = True

        ## Saving the checkpoint
        if LOSS < min_loss:
            min_epoch, min_loss = epoch + 1, LOSS

        checkpoint_info = {
            'epoch': epoch + 1, 'total_samples': total_samples, # training progress 
            'min_epoch': min_epoch, 'min_loss': min_loss, # best model and loss
            'model_state_dict': model.state_dict(), # model parameters
            'lr_decay_mode': lr_decay_mode, 'learning_rate': optimizer.param_groups[0]['lr'],
            'ssr_decay_mode': ssr_decay_mode, 'scheduled_sampling_ratio': scheduled_sampling_ratio} 

        torch.save(checkpoint_info, os.path.join(MODEL_DIR, 'training_last.pt'))

        if (epoch + 1) % save_epoch == 0:
            torch.save(checkpoint_info, os.path.join(MODEL_DIR, 'training_%d.pt' % (epoch + 1)))

        if (epoch + 1) == min_epoch:
            torch.save(checkpoint_info, os.path.join(MODEL_DIR, 'training_best.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Conv-TT-LSTM Training")

    ## Data format (batch_size x time_steps x height x width x channels)

    # batch size and the logging period (0)
    parser.add_argument('--batch-size',  default =  32, type = int,
        help = 'The batch size in training phase.')
    parser.add_argument('--log-samples', default = 384, type = int,
        help = 'Log the statistics every log_samples.')

    # frame split (1)
    parser.add_argument('--input-frames',  default = 10, type = int,
        help = 'The number of input frames to the model.')
    parser.add_argument('--future-frames', default = 10, type = int,
        help = 'The number of predicted frames of the model.')
    parser.add_argument('--output-frames', default = 19, type = int,
        help = 'The number of output frames of the model.')

    # frame format (2, 3, 4)
    parser.add_argument('--img-height',  default = 64, type = int, 
        help = 'The image height of each video frame.')
    parser.add_argument('--img-width',   default = 64, type = int, 
        help = 'The image width of each video frame.')
    parser.add_argument('--img-channels', default = 1, type = int, 
        help = 'The number of channels in each video frame.')

    ## Devices (CPU, single-GPU or multi-GPU)

    # whether to use GPU for training
    parser.add_argument('--use-cuda', dest = 'use_cuda', 
        action = 'store_true',  help = 'Use GPU for training.')
    parser.add_argument( '--no-cuda', dest = 'use_cuda', 
        action = 'store_false', help = "Do not use GPU for training.")
    parser.set_defaults(use_cuda = True)

    # whether to use multi-GPU for training
    parser.add_argument( '--multi-gpu', dest = 'multi_gpu', 
        action = 'store_true',  help = 'Use multiple GPUs for training.')
    parser.add_argument('--single-gpu', dest = 'multi_gpu', 
        action = 'store_false', help = 'Do not use multiple GPU for training.')
    parser.set_defaults(multi_gpu = True)

    # random seed for reproducibility
    parser.add_argument('--use-seed', dest = 'use_seed', 
        action = 'store_true',  help = 'Fix the random seed to reproduce the model.')
    parser.add_argument( '--no-seed', dest = 'use_seed', 
        action = 'store_false', help = 'Randomly choose the random seed.')
    parser.set_defaults(use_seed = True)

    parser.add_argument('--random-seed', default = 0, type = int, 
        help = 'The random seed number (to reproduce the model).')

    ## Models (Conv-LSTM/ARMA-LSTM, non-local Conv-LSTM/ARMA-LSTM)

    # model name (with time stamp as suffix)
    parser.add_argument('--model-name',  default = "test", type = str,
        help = 'The model name is used to create the folder names.')
    parser.add_argument('--model-stamp', default = "0000", type = str, 
        help = 'The stamp is used to create the suffix to the model name.')

    # model type and size (depth and width)
    parser.add_argument('--use-arma', dest = 'use_arma', 
        action = 'store_true',  help = 'Use ARMA convolutions in the ConvLSTM model.')
    parser.add_argument( '--no-arma', dest = 'use_arma', 
        action = 'store_false', help = 'Use standard convolutions in the ConvLSTM model.')
    parser.set_defaults(use_arma = False)

    parser.add_argument('--model-size', default = 'small', type = str,
        help = 'The model size (\"small\", \"origin\", \"wide\" or \"deep\").')

    parser.add_argument('--use-sigmoid', dest = 'use_sigmoid', 
        action = 'store_true',  help = 'Used sigmoid function at the output of the model.')
    parser.add_argument( '--no-sigmoid', dest = 'use_sigmoid', 
        action = 'store_false', help = 'Use output from the last layer as the final output.')
    parser.set_defaults(use_sigmoid = False)

    # parameters of the non-local blocks
    parser.add_argument('--use-non-local', dest = 'use_non_local', 
        action = 'store_true',  help = 'Use non-local blocks in the ConvLSTM model.')
    parser.add_argument( '--no-non-local', dest = 'use_non_local', 
        action = 'store_false', help = 'Do not use non-local-blocks in the ConvLSTM model.')
    parser.set_defaults(use_non_local = False)

    parser.add_argument('--pairwise-function', default = 'embedded_gaussian', type = str, 
        help = 'The pairwise function used in the non-local blocks.')

    parser.add_argument('--use-norm', dest = 'use_norm', 
        action = 'store_true',  help = 'Use normalization in the non-local blocks.')
    parser.add_argument( '--no-norm', dest = 'use_norm', 
        action = 'store_false', help = 'No normalization in the non-local blocks.')
    parser.set_defaults(use_norm = True)

    parser.add_argument('--use-sub-sample', dest = 'use_sub_sample', 
        action = 'store_true',  help = 'Use sub-sampling in the non-local blocks.')
    parser.add_argument( '--no-sub-sample', dest = 'use_sub_sample', 
        action = 'store_false', help = 'No sub-sampling in the non-local blocks.' )
    parser.set_defaults(use_sub_sample = True)

    # parameters of the convolutional operations

    # (moving-average module)
    parser.add_argument('--w-kernel-size', default = 5, type = int, 
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

    ## Dataset (Input to the training algorithm)
    parser.add_argument('--dataset', default = "MNIST", type = str, 
        help = 'The dataset name. (Options: MNIST, KTH, KITTI)')
    parser.add_argument('--data-path', default = 'default', type = str,
        help = 'The path to the dataset folder.')

    # training set
    parser.add_argument('--train-data-file', default = 'moving-mnist-train-new1.npz', type = str,
        help = 'Name of the folder/file for training set.')
    parser.add_argument('--train-samples', default = 10000, type = int,
        help = 'Number of unique samples in training set.')

    # validation set
    parser.add_argument('--valid-data-file', default = 'moving-mnist-val-new1.npz', type = str, 
        help = 'Name of the folder/file for validation set.')
    parser.add_argument('--valid-samples', default =  3000, type = int, 
        help = 'Number of unique samples in validation set.')

    ## Results and Models (Output from the training algorithm)
    parser.add_argument('--output-path', default = 'default', type = str,
        help = "The path to the folder storing the outputs (models and results).")

    ## Learning algorithm

    # loss function for training
    parser.add_argument('--loss-function', default = 'l1l2', type = str, 
        help = 'The loss function for training.')

    # total number of epochs and the interval to save a checkpoint
    parser.add_argument('--num-epochs', default = 500, type = int, 
        help = 'Number of total epochs in training.')
    parser.add_argument('--save-epoch', default =  20, type = int, 
        help = 'Save the model parameters every save_epoch.')

    # the epoch to start/resume training
    parser.add_argument('--start-begin', dest = 'start_begin', 
        action = 'store_true',  help = 'Start training a new model from the beginning.')
    parser.add_argument('--start-exist', dest = 'start_begin', 
        action = 'store_false', help = 'Resume training an existing model.')
    parser.set_defaults(start_begin = True)

    parser.add_argument('--start-last', dest = 'start_last', 
        action = 'store_true',  help = 'Resume training from the last available model')
    parser.add_argument('--start-spec', dest = 'start_last', 
        action = 'store_false', help = 'Resume training from the model of the specified epoch')
    parser.set_defaults(start_last = True)

    parser.add_argument('--start-epoch', default = 0, type = int, 
        help = 'The number of epoch to resume training.')

    # logging for automatic scheduling
    parser.add_argument('--decay-log-epochs', default = 20, type = int, 
        help = 'The window size to determine automatic scheduling.')

    # gradient clipping
    parser.add_argument('--gradient-clipping', dest = 'gradient_clipping', 
        action = 'store_true',  help = 'Use gradient clipping in training.')
    parser.add_argument('--no-clipping', dest = 'gradient_clipping', 
        action = 'store_false', help = 'No gradient clipping in training.')
    parser.set_defaults(use_clipping = False)

    parser.add_argument('--clipping-threshold', default = 3, type = float,
        help = 'The threshold value for gradient clipping.')

    # learning rate
    parser.add_argument('--learning-rate-decay', dest = 'learning_rate_decay', 
        action = 'store_true',  help = 'Use learning rate decay in training.')
    parser.add_argument('--no-rate-decay', dest = 'learning_rate_decay', 
        action = 'store_false', help = 'Do not use learning rate decay in training.')
    parser.set_defaults(learning_rate_decay = True)

    parser.add_argument('--learning-rate', default = 1e-3, type = float,
        help = 'Initial learning rate of the Adam optimizer.')
    parser.add_argument('--lr-decay-start', default = 20, type = int,
        help = 'The minimum epoch (after scheduled sampling) to start learning rate decay.')
    parser.add_argument('--lr-decay-epoch', default = 5, type = int,
        help = 'The learning rate is decayed every decay_epoch.')
    parser.add_argument('--lr-decay-rate', default = 0.98, type = float,
        help = 'The learning rate by decayed by decay_rate every epoch.')

    # scheduled sampling ratio
    parser.add_argument('--teacher-forcing', dest = 'teacher_forcing', action = 'store_true', 
        help = 'Use teacher forcing (with scheduled sampling) in training.')
    parser.add_argument('--no-forcing', dest = 'teacher_forcing', action = 'store_false',
        help = 'Training without teacher forcing (with scheduled sampling).')
    parser.set_defaults(teacher_forcing = True)

    parser.add_argument('--ssr-decay-start', default = 20, type = int,
        help = 'The minimum epoch to start scheduled sampling.')
    parser.add_argument('--ssr-decay-epoch', default =  1, type = int, 
        help = 'Decay the scheduled sampling every ssr_decay_epoch.')
    parser.add_argument('--ssr-decay-ratio', default = 4e-3, type = float,
        help = 'Decay the scheduled sampling by ssr_decay_ratio every time.')

    main(parser.parse_args())