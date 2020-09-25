import math
import torch
import torch.nn as nn

'''
S   - in_channels
T   - out_channels
D^w - w_kernel_size
D^a - a_kernel_size
'''

class ARMA2d(nn.Module):
    def __init__(self, in_channels, out_channels,
            w_kernel_size = 3, w_padding_mode = 'zeros',    
            w_padding = 1, w_stride = 1, w_dilation = 1, w_groups = 1, bias = True,
            a_kernel_size = 3, a_padding_mode = 'circular', 
            a_padding = 0, a_stride = 1, a_dilation = 1):
        """
        Initialization of a 2D-ARMA layer.
        """
        super(ARMA2d, self).__init__()

        self.moving_average = nn.Conv2d(in_channels, out_channels, w_kernel_size,
            padding = w_padding, # padding_mode = w_padding_mode, 
            stride = w_stride, dilation = w_dilation, groups = w_groups, bias = bias) 

        self.autoregressive = AutoRegressive2d(out_channels, a_kernel_size,
                padding = a_padding, padding_mode = a_padding_mode,
            stride = a_stride, dilation = a_dilation)

    def forward(self, x):
        """
        Compuation of the 2D-ARMA layer.
        """
        # size:[M, S, I1, I2]->[M, T, I1, I2]->[M, T, I1, I2]
        x = self.moving_average(x)
        x = self.autoregressive(x)
        return x


class AutoRegressive2d(nn.Module):
    def __init__(self, channels, kernel_size = 3, 
            padding = 0, padding_mode = 'circular',
            stride = 1, dilation = 1):
        """
        Initialization of 2D-AutoRegressive layer.
        """
        super(AutoRegressive2d, self).__init__()

        if padding_mode == "circular":
            self.a = AutoRegressive_circular(
                channels, kernel_size, padding, stride, dilation)

        elif paddind_mode == "reflect":
            self.a = AutoRegressive_reflect(
                channels, kernel_size, padding, stride, dilation)
        else: 
            raise NotImplementedError

    def forward(self, x):
        """
        Computation of 2D-AutoRegressive layer.
        """
        x = self.a(x)
        return x

   
class AutoRegressive_circular(nn.Module):
    def __init__(self, channels, kernel_size = 3, 
            padding = 0, stride = 1, dilation = 1):
        """
        Initialization of a 2D-AutoRegressive layer (with circular padding).
        """
        super(AutoRegressive_circular, self).__init__()

        self.alpha = nn.Parameter(torch.Tensor(
            channels, kernel_size // 2, 4)) # size:[T, P, 4]

        self.set_parameters()

    def set_parameters(self):
        """
        Initialization of the learnable parameters.
        """
        nn.init.zeros_(self.alpha)

    def forward(self, x):
        """
        Computation of the 2D-AutoRegressive layer (with circular padding). 
        """    
        x = autoregressive_circular(x, self.alpha)
        return x


def autoregressive_circular(x, alpha):
    """
    Computation of a 2D-AutoRegressive layer (with circular padding).
    """

    if  x.size()[-2] < alpha.size()[1] * 2 + 1 or \
        x.size()[-1] < alpha.size()[1] * 2 + 1:
        return x

    # There're 4 chunks, each chunk is [T, P, 1]
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
        A_x_right, torch.zeros(chunks[0].size()[0], chunks[0].size()[1],
        x.size()[-2] - 3, device = alpha.device), A_x_left), -1)

    A_y = torch.cat((torch.ones(chunks[2].size(), device = alpha.device), 
        A_y_right, torch.zeros(chunks[2].size()[0], chunks[2].size()[1], 
        x.size()[-1] - 3, device = alpha.device), A_y_left), -1)

    # size: [T, P, I1] + [T, P, I2] -> [T, P, I1, I2]
    A = torch.einsum('tzi,tzj->tzij',(A_x, A_y))

    # Complex Division: FFT/FFT -> irFFT
    A_s = torch.chunk(A, A.size()[1], 1)
    for i in range(A.size()[1]):
        x = ar_circular.apply(x, torch.squeeze(A_s[i], 1))

    return x


def complex_division(x, A, trans_deno=False):
    a, b = torch.chunk(x, 2, -1)
    c, d = torch.chunk(A, 2, -1)

    if trans_deno: 
            # [a bj] / [c -dj] -> [ac-bd/(c^2+d^2) (bc+ad)/(c^2+d^2)j]
        res_l = (a * c - b * d) / (c * c + d * d)
        res_r = (b * c + a * d) / (c * c + d * d)
    else:   # [a bj] / [c  dj] -> [ac+bd/(c^2+d^2) (bc-ad)/(c^2+d^2)j]
        res_l = (a * c + b * d) / (c * c + d * d)
        res_r = (b * c - a * d) / (c * c + d * d)

    res = torch.cat((res_l, res_r), -1)
    return res


def complex_multiplication(x, A, trans_deno=False):
    a, b = torch.chunk(x, 2, -1)
    c, d = torch.chunk(A, 2, -1)

    if trans_deno:
            # [a bj]*[c -dj] -> [ac+bd (bc-ad)j]
        res_l = a * c + b * d
        res_r = b * c - a * d
    else:   # [a bj]*[c  dj] -> [ac-bd (ad+bc)j]
        res_l = a * c - b * d
        res_r = b * c + a * d

    res = torch.cat((res_l,res_r),  -1)
    return res

    
class ar_circular(torch.autograd.Function):

    # x size: [M, T, I1, I2]
    # a size:[T, I1, I2]
    @staticmethod
    def forward(ctx, x, a):
        X = torch.rfft(x, 2, onesided=False)  # size:[M, T, I1, I2, 2]
        A = torch.rfft(a, 2, onesided=False)  # size:[T, I1, I2, 2]
        Y = complex_division(X, A)  # size:[M, T, I1, I2, 2]
        y = torch.irfft(Y, 2, onesided=False)  # size:[M, T, I1, I2]

        ctx.save_for_backward(A, Y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        """
        {grad_a} * a^T    = - grad_y  * y^T
        [T, I1, I2]   * [T, I1, I2] = [M, T, I1, I2] * [M, T, I1, I2]

        a^T    * {grad_x}     = grad_y
        [T, I1, I2] * [M, T, I1, I2]   = [M, T, I1, I2]

        intermediate = grad_y / a^T
        """
        A, Y = ctx.saved_tensors
        grad_x = grad_a = None  

        grad_Y = torch.rfft(grad_y, 2, onesided = False)
        intermediate = complex_division(grad_Y, A, trans_deno = True)  # size: [M, T, I1, I2]
        grad_x = torch.irfft(intermediate, 2, onesided=False)

        intermediate = - complex_multiplication(intermediate, Y, trans_deno = True) # size: [M, T, I1, I2]
        grad_a = torch.irfft(intermediate.sum(0), 2, onesided = False)  # size:[T, I1, I2]
        return grad_x, grad_a