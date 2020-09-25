import argparse, os

import torch.nn as nn
import matplotlib.pyplot as plt

from utils.ARMA_Layer import ARMA2d
from receptivefield.pytorch import PytorchReceptiveField

import tikzplotlib

class Identity(nn.Module):
    """ An identity activation function """
    def forward(self, x):
        return x

class ARMA_CNN(nn.Module):
    def __init__(self, activation = "sigmoid",
        img_channels = 3, num_features = 32, num_layers = 5, 
        w_kernel_size = 3, w_dilation = 1, a_kernel_size = 3, a_init = 0,
        dirac_after_activation = True, log_layers_interval = 1):
        """
        Initialization of a ARMA convolutional neural network.

        Arguments:
        ----------
        activation: str
            The type of activation used in the network. 
            options: sigmoid, relu, identity
            default: sigmoid

        img_channels: int
            The number of channels in the input image.
            default: 3 (colored)
        num_features: int
            The number of feature maps at each layer. 
            default: 32
        num_layers: int
            The number of convolutional layers in the network.
            default: 5

        w_kernel_size: int 
            The kernel size of the moving-average module.
            default: 3
        w_dilation: int
            The dilation of the moving-average module.
            default: 1

        a_kernel_size: int
            The kernel size of the autoregressive module.
            default: 3
        a_init: float in [0, 1)
            The strenght of initialization of the autoregressive module.
            default: 0

        dirac_after_activation: bool
            Whether to compute the gradient maps after activation.
            default: True
        log_layers_interval: 1
            Compute the gradient map every log_layers_interval layers.
            default: 1
        """
        super(ARMA_CNN, self).__init__()

        # specify the activation function 
        if   activation == "identity":
            Activation = Identity()
        elif activation == "relu":
            Activation = nn.ReLU()
        elif activation == "sigmoid":
            Activation = nn.Sigmoid()
        else:
            raise NotImplementedError 

        self.layers = nn.ModuleList()
        self.select = list()

        for l in range(num_layers):
            self.layers.append(ARMA2d(
                img_channels if l == 0 else num_features, num_features, 
                w_kernel_size = w_kernel_size, w_dilation = w_dilation,
                a_kernel_size = a_kernel_size, a_init = a_init))
            self.layers.append(Activation)

            if l % log_layers_interval == 0:
                self.select.append(2 * l + int(dirac_after_activation))

    def forward(self, inputs):
        """
        Computation of the ARMA convolutional neural network.

        Arguments:
        ----------
        inputs:  a tensor of size [batch_size, channels, height, width]
            Input to the ARMA convolutional network.

        Returns:
        -------- 
        outputs: a tensor of size [batch_size, num_features, height, width]
            Output of the ARMA convolutional network.

        Notice: in this application, the batch_size is always 1. 
        """
        self.feature_maps = list()
        for l, layer in enumerate(self.layers):
            inputs = self.layers[l](inputs)
            if l in self.select:
                self.feature_maps.append(inputs)
                
        return inputs


def main(args):

    # define the model function for ARMA convolutional neural network
    def model_fn() -> nn.Module:
        model = ARMA_CNN(
            img_channels = args.img_channels, activation = args.activation,
            num_layers = args.num_layers, num_features = args.num_features, 
            w_kernel_size = args.w_kernel_size, w_dilation = args.w_dilation,
            a_kernel_size = args.a_kernel_size, a_init = args.a_init, 
            dirac_after_activation = args.dirac_after_activation, 
            log_layers_interval = args.log_layers_interval)

        model.eval()
        return model

    # compute the parameters for the receptive field
    input_shape = [args.img_height, args.img_width, args.img_channels]
    rf = PytorchReceptiveField(model_fn)
    rf_params = rf.compute(input_shape = input_shape)

    # create the visualization of effective receptive field
    img_height_center = args.img_height // 2
    img_width_center  = args.img_width  // 2

    num_gradient_maps = (args.num_layers - 1) // args.log_layers_interval + 1

    if args.draw_multi_layers: # --plot-all-layers
        print([(img_height_center, img_width_center)] * num_gradient_maps)
        rf.plot_gradients_at(layout = (1, num_gradient_maps), 
            points = [(img_height_center, img_width_center)] * num_gradient_maps,
            figsize = (args.fig_height * num_gradient_maps, args.fig_width))
    else: # --plot-one-layer
        rf.plot_gradient_at(fm_id = args.num_layers - 1, 
            point = (img_height_center, img_width_center), 
            figsize = (args.fig_height, args.fig_width))

    if args.plot_fig: # --plot-fig
        plt.show()
    else: # --save-fig
        if args.fig_name == "default":
            fig_name = "%s_l%dr%dc%d_w%dd%da%d_%d" % (args.activation, 
                args.num_layers, args.img_height, args.num_features, 
                args.w_kernel_size, args.w_dilation, args.a_kernel_size, 
                int(args.a_init * 100))
        else: # if args.fig_name != "default":
            fig_name = args.fig_name

        if not os.path.exists(args.fig_path):
            os.makedirs(args.fig_path)

        # path to the figure 
        fig_name = os.path.join(args.fig_path, fig_name)
        if args.save_png:
            plt.savefig(fig_name + ".png")
        else:
            tikzplotlib.save(fig_name + ".tex")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Effective Receptive Field (ERF) \
        for ARMA Convolution Neural Networks (ARMA-CNN).")

    # format of the input images
    parser.add_argument("--img-height", default =  64, type = int, 
        help = "The height of the input image.")
    parser.add_argument("--img-width",  default =  64, type = int, 
        help = "The width  of the input image.")
    parser.add_argument("--img-channels", default = 3, type = int, 
        help = "Number of channels in the input image.")

    # network architecture of the ARMA CNN
    parser.add_argument("--num-layers",   default = 9, type = int, 
        help = "Number of ARMA layers in the model.")
    parser.add_argument("--num-features", default = 32, type = int, 
        help = "Number of feature maps at each layer.")
    parser.add_argument("--activation", default = "sigmoid", type = str,
        help = "The type of activations in the model.")

    # hyper-parameters of 2D-ARMA layers

    # (moving-average module)
    parser.add_argument("--w-kernel-size", default = 3, type = int,
        help = "The kernel size of the moving-average module.")
    parser.add_argument("--w-dilation", default = 1, type = int, 
        help = "The dilation of convolution in the moving-avarage module.")

    # (autoregressive module)
    parser.add_argument("--a-kernel-size", default = 3, type = int, 
        help = "The kernel size of the autoregressive module.")
    parser.add_argument("--a-init", default = 0, type = float, 
        help = "The strength of initialization of the AR module.")

    # parameters for plotting and saving 
    parser.add_argument("--draw-multi-layers", dest = "draw_multi_layers", 
        action = "store_true",  help = "Draw gradient maps for multiple layers.")
    parser.add_argument("--draw-first-layer",  dest = "draw_first_layers", 
        action = "store_false", help = "Draw gradient map for the first layer.")
    parser.set_defaults(draw_multi_layers = True)

    parser.add_argument("--dirac-after-activation",  dest = "dirac_after_activation",
        action = "store_true",  help = "Draw gradient maps after activation.")
    parser.add_argument("--dirac-before-activation", dest = "dirac_before_activation",
        action = "store_false", help = "Draw gradient maps before activation.")
    parser.set_defaults(dirac_after_activation = True)

    parser.add_argument("--log-layers-interval", default = 2, type = int, 
        help = "The interval of layers to log one gradient map.")

    parser.add_argument("--fig-height", default = 5, type = int, 
        help = "The height of the drawn figure for gradient maps.")
    parser.add_argument("--fig-width",  default = 5, type = int, 
        help = "The width  of the drawn figure for gradient maps.")

    parser.add_argument("--plot-fig", dest = "plot_fig", 
        action = "store_true",  help = "Plot the drawn figure on screen.")
    parser.add_argument("--save-fig", dest = "plot_fig", 
        action = "store_false", help = "Save the drawn figure to folder.")
    parser.set_defaults(plot_fig = True)

    parser.add_argument("--fig-path", default = "../figures/erf/", type = str, 
        help = "The path to the folder storing the figures.")
    parser.add_argument("--fig-name", default = "default", type = str, 
        help = "The name of the current visualization.")

    parser.add_argument("--save-png", dest = "save_png", 
        action = "store_true",  help = "Save the figure in png format.")
    parser.add_argument("--save-tex", dest = "save_png", 
        action = "store_false", help = "Save the figure in tex format.")
    parser.set_defaults(save_png = True)

    main(parser.parse_args())