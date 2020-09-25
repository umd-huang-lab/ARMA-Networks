import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.arma_layers import ARMA2d

## Standard Convolutional-LSTM Module
class ConvLSTMCell(nn.Module):

    def __init__(self, 
        input_channels, hidden_channels, arma = False,
        w_kernel_size = 3, w_bias = True, w_dilation = 1,
        a_kernel_size = 3, a_padding_mode = "circular"):
        """
        Construction of convolutional LSTM cell.
        
        Arguments:
        ----------
        input_channels: int
            Number of channels of the input tensor.
        hidden_channels: int
            Number of channels of the hidden/cell states.
        arma: bool
            Whether to use 2D-ARMA convoutional layer.
            default: False

        (Hyper-parameters of the moving-average module)
        w_kernel_size: int
            Size of the squared convolutional kernel.
            default: 3
        w_bias: bool
            Whether to add bias term after convolution.
            default: True
        
        (Hyper-parameters of the auto-regressive module)
        a_kernel_size: int 
            Size of the squared convolutional kernel.
            default: 3
        a_padding_mode: str ("circular", "reflect")
            Padding mode of the auto-regressive module.
            default: "circular"
        """
        super(ConvLSTMCell, self).__init__()

        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels

        if arma:
            self.conv =    ARMA2d(
                in_channels  = input_channels + hidden_channels, 
                out_channels  = 4 * hidden_channels, bias = w_bias, 
                w_kernel_size = w_kernel_size, w_dilation = w_dilation, 
                w_padding = w_dilation * (w_kernel_size - 1) // 2, 
                a_kernel_size = a_kernel_size, a_padding_mode = a_padding_mode)
        else: 
            self.conv = nn.Conv2d(
                in_channels  = input_channels + hidden_channels, 
                out_channels = 4 * hidden_channels, bias = w_bias,
                kernel_size  = w_kernel_size, dilation = w_dilation,
                padding = w_dilation * (w_kernel_size - 1) // 2)

        # Note: hidden/cell states are not intialized in construction
        self.hidden_states, self.cell_state = None, None

    def initialize(self, inputs):
        """
        Initialization of convolutional LSTM cell.
        
        Arguments: 
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, channels, height, width]
            Input tensor of convolutional-LSTM cell.
        """
        device = inputs.device # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)
        self.cell_states = torch.zeros(batch_size, 
            self.hidden_channels, height, width, device = device)

    def forward(self, inputs, first_step = False):
        """
        Computation of convolutional LSTM cell.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size
            [batch_size, input_channels, height, width] 
            Input tensor to the convolutional-LSTM cell.

        first_step: bool
            Whether the input is the first step of the sequence. 
            If so, both states are intialized to zeros tensors.
            default: False
        
        Returns:
        --------
        hidden_states: another 4-th order tensor of size 
            [batch_size, hidden_channels, height, width]
            Hidden states (and outputs) of the convolutional-LSTM cell.
        """
        if first_step: self.initialize(inputs)

        concat_conv = self.conv(torch.cat(
            [inputs, self.hidden_states], dim = 1)) # concat over channels

        cc_i, cc_f, cc_o, cc_g = torch.split(
            concat_conv, self.hidden_channels, dim = 1) # split over channels

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_states   = f * self.cell_states + i * g
        self.hidden_states = o * torch.tanh(self.cell_states)
        
        return self.hidden_states