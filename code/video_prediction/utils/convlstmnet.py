import torch
import torch.nn as nn
from utils.convlstmcell import ConvLSTMCell
from utils.non_local_blocks import NonLocalBlock2D

## Convolutional-LSTM network
class ConvLSTMNet(nn.Module):
    def __init__(self,
        # network architecture
        layers_per_block, hidden_channels, 
        skip_stride = None, arma = False,
        # interfaces of input/outputs
        input_channels = 1, output_sigmoid = False,
        input_height = 64, input_width = 64,
        # non-local blocks
        non_local = False, pairwise_function = None,
        sub_sampling = True, layer_norm = False,
        # parameters of convolutional operations
        w_kernel_size = 3, w_dilation = 1, w_bias = True,
        a_kernel_size = 3, a_padding_mode = "circular"):
        """
        Initialization of a non-local Conv-LSTM network.
        
        Arguments:
        ----------
        [Hyper-parameters of model architecture]
        layers_per_block: list of ints
            Number of Conv-LSTM layers in each block. 
        hidden_channels: list of ints
            Number of output channels.
        Note: The length of hidden_channels is equal to layers_per_block.
        skip_stride: int
            The stride (in term of blocks) of the skip connections.
            default: None, i.e. no skip connection

        arma: bool
            Whether or not to use 2D-ARMA convoutional layer.
            default: False
        
        [Hyper-parameters of input/output interfaces]
        input_channels: int 
            The number of channels for input video. 
            Note: 1 for colored video, 1 for grayscale video.
            default: 1 (i.e. colored video)
        input_height, input_width: int
            The height/width of the input video.
            default: 64
        output_sigmoid: bool
            Whether to apply sigmoid function following the last layer.
            default: False

        [Hyper-parameters for non-local blocks]
        non_local: bool
            Whether to use non-local blocks in the ConvLSTM network.
            default: None
        pairwise_function: None or str
            The pairwise function used in non-local blocks.
            options: "embedded_gaussian", "gaussian", "dot_product", "concatenation".
            default: None, i.e. "embedded_gaussian"

        layer_norm: bool
            Whether to apply normalization at the output of the non-local blocks.
            default: False
        normalized_shape: a tuple of ints
            The expected input shape of the layer normalization.
            Note: The shape must be set if layer_norm is True.
            Default: None

        [Hyper-parameters of convolutional operations]
        w_kernel_size: int
            Size of the squared convolutional kernel.
            default: 3
        w_dilation: int
            Dilation of the squared convolutional kernel.
            default: 1
        w_bias: bool
            Whether to add bias term after convolution.
            default: True

        a_kernel_size: int
            Size of the squared convolutional kernel.
            default: 3
        a_padding_mode: str ("circular", "reflect")
            The 
        """
        super(ConvLSTMNet, self).__init__()

        ## Hyper-paramters for Conv-LSTM network
        self.layers_per_block = layers_per_block
        self.hidden_channels  = hidden_channels

        self.num_blocks = len(layers_per_block)
        assert self.num_blocks == len(hidden_channels), \
            "The lengths of hidden_channels and layers_per_block are different."

        self.skip_stride = (self.num_blocks + 1) \
            if skip_stride is None else skip_stride

        self.output_sigmoid = output_sigmoid

        ## Template of each basic building block
        Cell = lambda in_channels, out_channels, arma: ConvLSTMCell(
            input_channels = in_channels,  hidden_channels = out_channels, arma = arma,
            w_kernel_size = w_kernel_size, w_bias = w_bias, w_dilation = w_dilation, 
            a_kernel_size = a_kernel_size, a_padding_mode = a_padding_mode)

        ## Construction of the Conv-LSTM network
        self.non_local = non_local
        if non_local and pairwise_function is None: 
            pairwise_function = "embedded_gaussian"

        self.layers = nn.ModuleList()
        for b in range(self.num_blocks):
            for l in range(layers_per_block[b]):

                # number of input channels to the current layer
                if l > 0: 
                    channels = hidden_channels[b]
                elif b == 0: # if l == 0 and b == 0:
                    channels = input_channels
                else: # if l == 0 and b > 0:
                    channels = hidden_channels[b-1]
                    if b > skip_stride:
                        channels += hidden_channels[b-1-skip_stride] 

                    if self.non_local:
                        self.layers.append(NonLocalBlock2D(channels, 
                            pairwise_function = pairwise_function, layer_norm = layer_norm, 
                            normalized_shape = (channels, input_height, input_width)))

                self.layers.append(Cell(channels, hidden_channels[b], arma))

        # number of input channels to the last (output) layer
        channels = hidden_channels[-1]
        if self.num_blocks >= self.skip_stride:
            channels += hidden_channels[-(self.skip_stride + 1)]

        self.layers.append(nn.Conv2d(channels, input_channels, 
            kernel_size = 1, padding = 0, bias = True))

    def forward(self, inputs, input_frames, future_frames, output_frames, 
                teacher_forcing = False, scheduled_sampling_ratio = 0):
        """
        Computation of the Conv-LSTM network.
        
        Arguments:
        ----------
        inputs: a 5-th order tensor of size 
            [batch_size, input_frames, input_channels, height, width] 
            Input tensor to the Conv-LSTM network. 
        
        input_frames: int
            The number of input frames to the model.
        future_frames: int
            The number of future frames predicted by the model.
        output_frames: int
            The number of output frames  returned by the model.

        teacher_forcing: bool
            Whether the model is trained in teacher_forcing mode.
            Note 1: In test mode, teacher_forcing should be set as False.
            Note 2: If teacher_forcing == True,  # of input frames = total_steps
                    If teacher_forcing == False, # of input frames = input_frames
        scheduled_sampling_ratio: float between [0, 1]
            The ratio of ground-truth frames used in teacher_forcing mode.
            default: 0 (i.e. no teacher forcing effectively)

        Returns:
        --------
        outputs: a 5-th order tensor of size 
            [batch_size, output_frames, hidden_channels, height, width]
            Output tensor of the Conv-LSTM network.
        """

        # compute the teacher forcing mask 
        if teacher_forcing and scheduled_sampling_ratio > 1e-6:
            # generate the teacher_forcing mask (4-th order)
            teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio * 
                torch.ones(inputs.size(0), future_frames - 1, 
                    1, 1, 1, device = inputs.device))
        else: # if not teacher_forcing or scheduled_sampling_ratio < 1e-6:
            teacher_forcing = False

        # the number of time steps in the computational graph
        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            # input_: 4-th order tensor of size 
            # [batch_size, input_channels, height, width]
            if t < input_frames: 
                input_ = inputs[:, t]
            elif not teacher_forcing:
                input_ = outputs[t-1]
            else: # if t >= input_frames and teacher_forcing:
                mask = teacher_forcing_mask[:, t - input_frames]
                input_ = inputs[:, t] * mask + outputs[t-1] * (1 - mask)

            queue, k = [], 0 # previous outputs for skip connection
            for b in range(self.num_blocks):
                if b > 0 and self.non_local:
                    input_ = self.layers[k](input_)
                    k += 1 

                for l in range(self.layers_per_block[b]):
                    input_ = self.layers[k](input_, first_step = (t == 0))
                    k += 1

                queue.append(input_)
                if b >= self.skip_stride:
                    # concatenate the inputs over the channels 
                    input_ = torch.cat([input_, queue.pop(0)], dim = 1) 

            # map the hidden states to predictive frames
            outputs[t] = self.layers[k](input_)
            if self.output_sigmoid:
                outputs[t] = torch.sigmoid(outputs[t])

        # outputs: 5-th order tensor of size 
        # [batch_size, output_frames, input_channels, height, width]
        outputs = torch.stack(outputs[-output_frames:], dim = 1)
        
        return outputs