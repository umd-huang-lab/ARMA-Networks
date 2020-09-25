import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.layers import ARMA2d

def init_weights(net, init_type = 'normal', gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)

        elif classname.find('BatchNorm2d') != -1:
            torch.init.normal_(m.weight.data, 1.0, gain)
            torch.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Attention_block(nn.Module):
    def __init__(self, gate_channels, input_channels, inter_channels):
        """
        Construction of attention block.

        Arguments:
        ----------
        input_channels: int 
            The number of channels of the input features.

        gate_channels: int
            The number of channels of the gate signal.

        inter_channels: int
            The number of intermediate channels in the block.

        """
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(input_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, g, x):
        """
        Computation of attention block.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size
            [batch_size, input_channels, height, width]

        gates: a 4-th order tensor of size
            [batch_size, gate_channels, height, width]

        Returns:
        --------
        outputs: a 4-th order tensor of size
            [batch_size, input_channels, height, width]

        """
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, arma = False):
        """
        Construction of a convolutional block.

        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.

        arma: bool
            Whether to use ARMA layers or standard convolutional layer.
            default: False, i.e. use standard convolutioal layer

        """
        super(conv_block, self).__init__()

        if arma: # 2D ARMA layer
            nn_Conv2d = lambda in_channels, out_channels: ARMA2d(
                in_channels = in_channels, out_channels = out_channels, 
                w_kernel_size = 3, w_padding = 1, a_kernel_size = 3)
        else: # standard 2D convolutional layer 
            nn_Conv2d = lambda in_channels, out_channels: nn.Conv2d(
                in_channels = in_channels, out_channels = out_channels,
                kernel_size = 3, padding = 1)

        self.conv_block = nn.Sequential(
            nn_Conv2d(in_channels,  out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn_Conv2d(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, inputs):
        """
        Computation of the convolutional block.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, in_channels,  height, width]
            Input to the convolutional block.

        Returns:
        --------
        outputs: another 4-th order tensor of size
            [batch_size, out_channels, height, width]
            Output of the convolutional block.

        """
        outputs = self.conv_block(inputs)
        return outputs


class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels, arma = False):
        """
        Construction of a upsampling convolutional block.

        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.

        arma: bool
            Whether to use ARMA layers or standard convolutional layer.
            default: False, i.e. use standard convolutioal layer
        
        """
        super(up_conv,self).__init__()

        if arma: # 2D ARMA layer
            nn_Conv2d = lambda in_channels, out_channels: ARMA2d(
                in_channels = in_channels, out_channels = out_channels, 
                w_kernel_size = 3, w_padding = 1, a_kernel_size = 3)
        else: # standard 2D convolutional layer 
            nn_Conv2d = lambda in_channels, out_channels: nn.Conv2d(
                in_channels = in_channels, out_channels = out_channels,
                kernel_size = 3, padding = 1)

        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor = 2),
            nn_Conv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, inputs):
        """
        Computation of a upsampling convolutional block.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, in_channels, in_height, in_width]
            Input to the upsampling convolutional block.

        Returns:
        --------
        outputs: another 4-th order tensor of size
            [batch_size, out_channels, out_height, out_width]
            Note: out_height = in_height // 2
                  out_width  = in_width // 2
            Output of the upsampling convolutional block.

        """
        outputs = self.up_conv(inputs)
        return outputs


class Recurrent_block(nn.Module):
    def __init__(self, channels, arma = False, steps = 2):
        """
        Construction of a recurrent convolutional block.

        Arguments:
        ----------
        channels: int
            The number of input/output channels.

        arma: bool
            Whether to use ARMA layers or standard convolutional layers.
            default: False, i.e. use standard convolutioal layers

        steps: int
            The number of recurrent steps of the block.
            default: 2 

        """
        super(Recurrent_block,self).__init__()

        if arma: # 2D ARMA layer
            nn_Conv2d = lambda in_channels, out_channels: ARMA2d(
                in_channels = in_channels, out_channels = out_channels, 
                w_kernel_size = 3, w_padding = 1, a_kernel_size = 3)
        else: # standard 2D convolutional layer 
            nn_Conv2d = lambda in_channels, out_channels: nn.Conv2d(
                in_channels = in_channels, out_channels = out_channels,
                kernel_size = 3, padding = 1)

        self.conv = nn.Sequential(
            nn_Conv2d(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace = True)
        )

        self.steps = steps

    def forward(self, inputs):
        """
        Computation of the recurrent convolutional block.

        Arguments:
        ----------
        inputs: a 4-th order tensor of size
            [batch_size, channels, height, width]
            Input to the recurrent convolutional block.

        Returns:
        --------
        outputs: another 4-th order tensor of size
            [batch_size, channels, height, width]
            Output of the recurrent convolutional block. 

        """
        outputs = self.conv(inputs)
        for _ in range(self.steps - 1):
            outputs = self.conv(outputs + inputs)

        return outputs


class RRCNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, arma = False, steps = 2):
        """
        Construction of a recurrent residual convolutional block.

        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.

        arma: bool
            Whether to use ARMA layers or standard convolutional layers.
            default: False, i.e. use standard convolutioal layers

        steps: int
            The number of recurrent steps of the block.
            default: 2
        
        """
        super(RRCNN_block, self).__init__()

        self.Conv_1x1 = nn.Conv2d(in_channels, out_channels, 1)

        self.RCNN = nn.Sequential(
            Recurrent_block(out_channels, arma = arma, steps = steps),
            Recurrent_block(out_channels, arma = arma, steps = steps)
        )
        

    def forward(self, inputs):
        """
        Computation of the recurrent residual convolutional block.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, in_channels, height, width]
            Input to the recurrent residual convolutional block.

        Returns:
        --------
        outputs: a 4-th order tensor of size
            [batch_size, out_channels, height, width]
            Output of the recurrent residual convolutional block.  

        """
        inputs = self.Conv_1x1(inputs)
        outputs = self.RCNN(inputs)
        return inputs + outputs


class single_conv(nn.Module):
    def __init__(self, in_channels, out_channels, arma = False):
        """
        Construction of a single convolutional block.

        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.

        arma: bool
            Whether to use ARMA layers or standard convolutional layers.
            default: False, i.e. use standard convolutioal layers

        """
        super(single_conv,self).__init__()

        if arma: # 2D ARMA layer
            nn_Conv2d = lambda in_channels, out_channels: ARMA2d(
                in_channels = in_channels, out_channels = out_channels, 
                w_kernel_size = 3, w_padding = 1, a_kernel_size = 3)
        else: # standard 2D convolutional layer 
            nn_Conv2d = lambda in_channels, out_channels: nn.Conv2d(
                in_channels = in_channels, out_channels = out_channels,
                kernel_size = 3, padding = 1)

        self.conv = nn.Sequential(
            nn_Conv2d(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        """
        Computation of the single convolutional block.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, input_channels, height, width]
            Input to the single convolutional block.

        Returns:
        --------
        outputs: a 4-th order tensor of size
            [batch_size, output_channels, height, width]
            Output of the single convolutional block.  

        """
        outputs = self.conv(inputs)
        return outputs



class U_Net(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, factor = 1, arma = False, arma2 =False):
        """
        Construction of U-Net.

        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.

        arma: bool
            Whether to use ARMA layers or standard convolutional layers.
            default: False, i.e. use standard convolutioal layers

        """
        super(U_Net,self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2)
        self.Conv1 = conv_block(in_channels, 64//factor, arma = arma2)

        self.Conv2 = conv_block(64//factor,  128//factor,  arma = arma)
        self.Conv3 = conv_block(128//factor, 256//factor,  arma = arma)
        self.Conv4 = conv_block(256//factor, 512//factor,  arma = arma)
        self.Conv5 = conv_block(512//factor, 1024//factor, arma = arma)

        self.Up5 = up_conv(1024//factor, 512//factor, arma = arma)
        self.Up_conv5 = conv_block(1024//factor, 512//factor, arma = arma)

        self.Up4 = up_conv(512//factor, 256//factor, arma = arma)
        self.Up_conv4 = conv_block(512//factor, 256//factor, arma = arma)
        
        self.Up3 = up_conv(256//factor, 128//factor, arma = arma)
        self.Up_conv3 = conv_block(256//factor, 128//factor, arma = arma)
        
        self.Up2 = up_conv(128//factor, 64//factor, arma = arma)
        self.Up_conv2 = conv_block(128//factor, 64//factor, arma = arma2)

        self.Conv_1x1 = nn.Conv2d(64//factor, out_channels, 1)

    def forward(self, x):
        """
        Computation of the U-Net.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, in_channels, height, width]
            Input to the U-Net.

        Returns:
        --------
        outputs: a 4-th order tensor of size
            [batch_size, out_channels, height, width]
            Output of the U-Net.  

        """

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, arma = False, steps = 2):
        """
        Construction of Recurrent Residual U-Net.

        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.

        arma: bool
            Whether to use ARMA layers or standard convolutional layers.
            default: False, i.e. use standard convolutioal layers

        steps: int
            The number of recurrent steps of each recurrent block.
            default: 2

        """
        super(R2U_Net,self).__init__()
        
        self.Maxpool  = nn.MaxPool2d(2)
        self.Upsample = nn.Upsample(2)

        self.RRCNN1 = RRCNN_block(in_channels, 64, arma = arma, steps = steps)

        self.RRCNN2 = RRCNN_block(64,  128,  arma = arma, steps = steps)
        
        self.RRCNN3 = RRCNN_block(128, 256,  arma = arma, steps = steps)
        
        self.RRCNN4 = RRCNN_block(256, 512,  arma = arma, steps = steps)
        
        self.RRCNN5 = RRCNN_block(512, 1024, arma = arma, steps = steps)
        
        self.Up5 = up_conv(1024, 512, arma = arma)
        self.Up_RRCNN5 = RRCNN_block(1024, 512, arma = arma, steps = steps)
        
        self.Up4 = up_conv(512,  256, arma = arma)
        self.Up_RRCNN4 = RRCNN_block(512,  256, arma = arma, steps = steps)
        
        self.Up3 = up_conv(256,  128, arma = arma)
        self.Up_RRCNN3 = RRCNN_block(256,  128, arma = arma, steps = steps)
        
        self.Up2 = up_conv(128,   64, arma = arma)
        self.Up_RRCNN2 = RRCNN_block(128,   64, arma = arma, steps = steps)

        self.Conv_1x1 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        """
        Computation of the Recurrent Residual U-Net.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, in_channels, height, width]
            Input to the Recurrent Residual U-Net.

        Returns:
        --------
        outputs: a 4-th order tensor of size
            [batch_size, out_channels, height, width]
            Output of the Recurrent Residual U-Net.  

        """

        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, factor=1, arma = False, arma2=False):
        """
        Construction of Attention U-Net.

        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.

        arma: bool
            Whether to use ARMA layers or standard convolutional layer.
            default: False, i.e. use standard convolutioal layer

        """
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(2)

        self.Conv1 = conv_block(in_channels, 64//factor)

        self.Conv2 = conv_block(64//factor,  128//factor,  arma = arma2)
        self.Conv3 = conv_block(128//factor, 256//factor,  arma = arma)
        self.Conv4 = conv_block(256//factor, 512//factor,  arma = arma)
        self.Conv5 = conv_block(512//factor, 1024//factor, arma = arma)

        self.Up5 = up_conv(1024//factor, 512//factor, arma = arma)
        self.Att5 = Attention_block(512//factor, 512//factor, 256//factor)
        self.Up_conv5 = conv_block(1024//factor, 512//factor, arma = arma)

        self.Up4 = up_conv(512//factor, 256//factor, arma = arma)
        self.Att4 = Attention_block(256//factor, 256//factor, 128//factor)
        self.Up_conv4 = conv_block(512//factor,  256//factor, arma = arma)
        
        self.Up3 = up_conv(256//factor, 128//factor, arma = arma)
        self.Att3 = Attention_block(128//factor, 128//factor, 64//factor)
        self.Up_conv3 = conv_block(256//factor, 128//factor, arma = arma)
        
        self.Up2 = up_conv(128//factor, 64//factor, arma = arma)
        self.Att2 = Attention_block(64//factor, 64//factor, 32//factor)
        self.Up_conv2 = conv_block(128//factor, 64//factor, arma = arma2)

        self.Conv_1x1 = nn.Conv2d(64//factor, out_channels, 1)

    def forward(self,x):
        """
        Computation of the Attention U-Net.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, in_channels, height, width]
            Input to the Attention U-Net.

        Returns:
        --------
        outputs: a 4-th order tensor of size
            [batch_size, out_channels, height, width]
            Output of the Attention U-Net.  

        """

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, arma = False, steps = 2):
        """
        Construction of Recurrent Residual Attention U-Net.

        Arguments:
        ----------
        in_channels: int
            The number of input channels.

        out_channels: int
            The number of output channels.

        arma: bool
            Whether to use ARMA layers or standard convolutional layers.
            default: False, i.e. use standard convolutioal layers

        steps: int
            The number of recurrent steps of each recurrent block.
            default: 2

        """
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(2, 2)
        self.Upsample = nn.Upsample(2)

        self.RRCNN1 = RRCNN_block(in_channels, 64, arma = arma, steps = steps)

        self.RRCNN2 = RRCNN_block(64,  128,  arma = arma, steps = steps)
        
        self.RRCNN3 = RRCNN_block(128, 256,  arma = arma, steps = steps)
        
        self.RRCNN4 = RRCNN_block(256, 512,  arma = arma, steps = steps)
        
        self.RRCNN5 = RRCNN_block(512, 1024, arma = arma, steps = steps)
        
        self.Up5 = up_conv(1024, 512, arma = arma)
        self.Att5 = Attention_block(512, 512, 256)
        self.Up_RRCNN5 = RRCNN_block(1024, 512, arma = arma, steps = steps)
        
        self.Up4 = up_conv(512,  256)
        self.Att4 = Attention_block(256, 256, 128)
        self.Up_RRCNN4 = RRCNN_block(512, 256, arma = arma, steps = steps)
        
        self.Up3 = up_conv(256, 128, arma = arma)
        self.Att3 = Attention_block(128, 128, 64)
        self.Up_RRCNN3 = RRCNN_block(256, 128, arma = arma, steps = steps)
        
        self.Up2 = up_conv(128, 64, arma = arma)
        self.Att2 = Attention_block(64, 64, 32)
        self.Up_RRCNN2 = RRCNN_block(128, 64, arma = arma, steps = steps)

        self.Conv_1x1 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        """
        Computation of the Recurrent Residual Attention U-Net.
        
        Arguments:
        ----------
        inputs: a 4-th order tensor of size 
            [batch_size, in_channels, height, width]
            Input to the Recurrent Residual Attention U-Net.

        Returns:
        --------
        outputs: a 4-th order tensor of size
            [batch_size, out_channels, height, width]
            Output of the Recurrent Residual Attention U-Net.  

        """

        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
