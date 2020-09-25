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

# perceptive quality
import PerceptualSimilarity.models as PSmodels

def main(args):
    ## Data format: batch_size(0) x time_steps(1) x 
    #  img_height(2) x img_width(3) x channels(4) 

    # batch size (0)
    assert args.log_samples % args.batch_size == 0, \
        "The argument log_samples should be a multiple of batch_size."

    # frame split (1)
    input_frames  = args.input_frames
    future_frames = args.future_frames
    total_frames  = input_frames + future_frames 

    log_frames = args.log_frames

    list_input_frames  = list(range(0, input_frames,  log_frames))
    plot_input_frames  = len(list_input_frames)

    list_future_frames = list(range(0, future_frames, log_frames))
    plot_future_frames = len(list_future_frames)

    assert args.img_channels in [1, 3], \
        "The number of channels is either 1 or 3."

    img_colored = (args.img_channels == 3) 


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

    print("Model name:", model_name)
    print("# of future frames:", future_frames)

    PSmodel = PSmodels.PerceptualLoss(model = 'net-lin', 
        net = 'alex', use_gpu = use_cuda, gpu_ids = [0])


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

    # dataloaer for test set
    test_data_path = os.path.join(DATA_DIR, args.test_data_file)
    assert os.path.exists(test_data_path), \
        "The test set does not exist."

    test_data = Dataset({"path": test_data_path, "unique_mode": True,
        "num_frames": total_frames, "num_samples": args.test_samples,
        "height": args.img_height, "width": args.img_width, "channels": args.img_channels})

    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, 
        shuffle = False, num_workers = num_workers, drop_last = True)

    test_size = len(test_data_loader) * args.batch_size
    print("# of test samples:", test_size)


    ## Outputs (Models and Results)
    if args.output_path == "default":
        OUTPUT_DIR = {"MNIST": "./moving-mnist"}[dataset]
    else: # if args.output_path != "default":
        OUTPUT_DIR = args.output_path

    OUTPUT_DIR = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)

    # path to the models
    MODEL_DIR  =  os.path.join(OUTPUT_DIR, "models")
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

    # path to the results (images and statistics)
    RESULT_DIR =  os.path.join(OUTPUT_DIR, "results")
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    RESULT_IMG =  os.path.join(RESULT_DIR, 
        "test_images_" + str(eval_epoch) + "_" + str(future_frames))
    if not os.path.exists(RESULT_IMG):
        os.makedirs(RESULT_IMG)
  
    RESULT_STAT = os.path.join(RESULT_DIR, "test_stats")
    if not os.path.exists(RESULT_STAT):
        os.makedirs(RESULT_STAT)

    RESULT_STAT = os.path.join(RESULT_STAT, 'epoch_%d' % eval_epoch)


    ## Main script for test phase 
    MSE  = [0.] * future_frames
    PSNR = [0.] * future_frames
    SSIM = [0.] * future_frames
    PIPS = [0.] * future_frames

    with torch.no_grad():
        model.eval()
        
        samples = 0
        for frames in test_data_loader:
            samples += args.batch_size

            # 5-th order: batch_size x total_frames x channels x height x width 
            frames = frames.permute(0, 1, 4, 2, 3).to(device)

            inputs = frames[:,  :input_frames]
            origin = frames[:, -future_frames:]

            pred = model(inputs, 
                input_frames  =  input_frames, 
                future_frames = future_frames, 
                output_frames = future_frames, 
                teacher_forcing = False)

            # clamp the output to [0, 1]
            pred = torch.clamp(pred, min = 0, max = 1)

            # save the first sample for each batch to the folder
            if samples % args.log_samples == 0:
                print("samples: ", samples)

                input_0  = inputs[0, list_input_frames] 
                origin_0 = origin[0, list_future_frames]
                pred_0   =   pred[0, list_future_frames]

                # pad the input with zeros (if needed)
                if plot_input_frames < plot_future_frames:
                    input_0 = torch.cat([torch.zeros(plot_future_frames - plot_input_frames, 
                        args.img_channels, args.img_height, args.img_width, device = device), input_0], dim = 0)

                img = torchvision.utils.make_grid(torch.cat(
                    [input_0, origin_0, pred_0], dim = 0), nrow = plot_future_frames)
                
                RESULT_FILE = os.path.join(RESULT_IMG, "cmp_%d_%d.jpg" % (eval_epoch, samples))
                torchvision.utils.save_image(img, RESULT_FILE)

            # accumlate the statistics per frame
            for t in range(-future_frames, 0):
                origin_, pred_ = origin[:, t], pred[:, t]
                if not img_colored:
                    origin_ = origin_.repeat([1, 3, 1, 1])
                    pred_   =   pred_.repeat([1, 3, 1, 1])

                dist = PSmodel(origin_, pred_)
                PIPS[t] += torch.sum(dist).item()

            origin = origin.permute(0, 1, 3, 4, 2).cpu().numpy()
            pred   =   pred.permute(0, 1, 3, 4, 2).cpu().numpy()
            for t in range(-future_frames, 0):
                for i in range(args.batch_size):
                    origin_, pred_ = origin[i, t], pred[i, t]
                    if not img_colored:
                        origin_ = np.squeeze(origin_, axis = -1)
                        pred_   = np.squeeze(pred_,   axis = -1)

                    MSE[t]  += skimage.measure.compare_mse( origin_, pred_)
                    PSNR[t] += skimage.measure.compare_psnr(origin_, pred_)
                    SSIM[t] += skimage.measure.compare_ssim(origin_, pred_, multichannel = img_colored)

    for t in range(future_frames):
        MSE[t]  /= test_size
        PSNR[t] /= test_size
        SSIM[t] /= test_size
        PIPS[t] /= test_size

    # compute the average statistics 
    MSE_AVG  = sum(MSE)  / future_frames
    PSNR_AVG = sum(PSNR) / future_frames
    SSIM_AVG = sum(SSIM) / future_frames
    PIPS_AVG = sum(PIPS) / future_frames

    print("Epoch {}, MSE: {} (x1e-3); PSNR: {}, SSIM: {}, PIPS: {}".format(
        eval_epoch, 1e3 * MSE_AVG, PSNR_AVG, SSIM_AVG, PIPS_AVG))

    print("PSNR:", PSNR)
    print("SSIM:", SSIM)
    print("PIPS:", PIPS)

    np.savez(RESULT_STAT, MSE = MSE, PSNR = PSNR, SSIM = SSIM, PIPS = PIPS)
    print('--------------------------------------------------------------')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Conv-TT-LSTM Test")

    ## Data format (batch_size x time_steps x height x width x channels)

    # batch size and the logging period (0)
    parser.add_argument('--batch-size',  default = 24, type = int,
        help = 'The batch size in training phase.')
    parser.add_argument('--log-samples', default = 96, type = int,
        help = 'Log the statistics every log_samples.')

    # frame split (1)
    parser.add_argument('--input-frames',  default = 10, type = int,
        help = 'The number of input frames to the model.')
    parser.add_argument('--future-frames', default = 10, type = int,
        help = 'The number of predicted frames of the model.')
    
    parser.add_argument('--log-frames', default = 1, type = int, 
        help = 'Log the frames every log_frames.')

    # frame format (2, 3, 4)
    parser.add_argument('--img-height',  default = 64, type = int, 
        help = 'The image height of each video frame.')
    parser.add_argument('--img-width',   default = 64, type = int, 
        help = 'The image width of each video frame.')
    parser.add_argument('--img-channels', default = 1, type = int, 
        help = 'The number of channels in each video frame.')

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

    ## Dataset (Input)
    parser.add_argument('--dataset', default = "MNIST", type = str, 
        help = 'The dataset name. (Options: MNIST)')
    parser.add_argument('--data-path', default = 'default', type = str,
        help = 'The path to the dataset folder.')

    parser.add_argument('--test-data-file', default = 'moving-mnist-test-d2v2.npz', type = str, 
        help = 'Name of the folder/file for test set.')
    parser.add_argument('--test-samples', default = 5000, type = int, 
        help = 'Number of unique samples in test set.')

    ## Results and Models (Output)
    parser.add_argument('--output-path', default = 'default', type = str,
        help = "The path to the folder storing the outputs (models and results).")

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
    parser.add_argument('--eval-epoch', default = 100, type = int, 
        help = 'Evaluate the model of specified epoch.')

    main(parser.parse_args())