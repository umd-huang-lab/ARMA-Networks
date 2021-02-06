import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class ARMA2d(nn.Module):
    def __init__(self, in_channels, out_channels,
            w_kernel_size = 3, w_padding_mode = 'zeros',    
            w_padding = 1, w_stride = 1, w_dilation = 1, 
            w_groups = 1, bias = False, a_kernel_size = 3, 
            a_padding_mode = 'circular', a_padding = 0, 
            a_stride = 1, a_dilation = 1, a_init = 0):
        """
        Initialization of a 2D-ARMA layer.
        """
        super(ARMA2d, self).__init__()

        self.moving_average = nn.Conv2d(in_channels, out_channels, 
            w_kernel_size, padding = w_padding, padding_mode = w_padding_mode, 
            stride = w_stride, dilation = w_dilation, groups = w_groups, bias = bias) 

        self.autoregressive = AutoRegressive2d(out_channels, 
            a_kernel_size, padding = a_padding, padding_mode = a_padding_mode,
            stride = a_stride, dilation = a_dilation, init = a_init)

    def forward(self, x):
        """
        Computation of the 2D-ARMA layer.
        """

        # x = checkpoint.checkpoint(self.forward_pass, x)
        
        # size:[M, S, I1, I2]->[M, T, I1, I2]->[M, T, I1, I2]
        x = self.moving_average(x)
        # x = checkpoint.checkpoint(self.autoregressive, x)
        x = self.autoregressive(x)
        return x


class AutoRegressive2d(nn.Module):
    def __init__(self, channels, kernel_size = 3, 
            padding = 0, padding_mode = 'circular',
            stride = 1, dilation = 1, init = 0):
        """
        Initialization of a 2D-AutoRegressive layer.
        """
        super(AutoRegressive2d, self).__init__()

        if padding_mode == "circular":
            self.a = AutoRegressive_circular(channels, 
                kernel_size, padding, stride, dilation, init)

        elif padding_mode == "reflect":
            self.a = AutoRegressive_reflect( channels, 
                kernel_size, padding, stride, dilation, init)
        else: 
            raise NotImplementedError

    def forward(self, x):
        """
        Computation of the 2D-AutoRegressive layer.
        """
        x = self.a(x)
        return x


class AutoRegressive_circular(nn.Module):
    def __init__(self, channels, kernel_size = 3, 
            padding = 0, stride = 1, dilation = 1, init = 0):
        """
        Initialization of a 2D-AutoRegressive layer.
        """
        super(AutoRegressive_circular, self).__init__()

        # size: [T, P, 2, 2]
        self.alpha = nn.Parameter(torch.Tensor(channels, kernel_size // 2, 2, 2), requires_grad=True)
        self.set_parameters(init)

        # size: [2, 2*dilation + 1]
        self.alpha_weights = self.generate_alpha_weights(dilation)

        self.dummy = nn.Parameter(requires_grad=True)

    def set_parameters(self, init):
        """
        Initialization of the learnable parameters.
        """
        bound = -math.log(1 - init)
        nn.init.uniform_(self.alpha, a = -bound, b = bound)

    def generate_alpha_weights(self, dilation):
        alpha_weights = torch.zeros((2, 2 * dilation + 1), dtype=torch.float)
        alpha_weights[0, 0] = math.cos(-math.pi / 4)
        alpha_weights[1, 0] = -math.sin(-math.pi / 4)
        alpha_weights[0, -1] = math.sin(-math.pi / 4)
        alpha_weights[1, -1] = math.cos(-math.pi / 4)
        
        return alpha_weights

    def forward(self, x):
        """
        Computation of the 2D-AutoRegressive layer.
        """
        
        # alpha_weights should automatically be moved to the GPU, but it isn't for some reason
        # x = checkpoint.checkpoint(autoregressive_circular,
        #                          x, self.alpha, self.alpha_weights.to(self.alpha.device), self.dummy)
        x = autoregressive_circular(x, self.alpha, self.alpha_weights.to(self.alpha.device), self.dummy)
        
        return x


@torch.jit.script
def parameterized_weights_forward(x, alpha, alpha_weights):
    # size: [T, P, 2, 2]
    alpha = alpha.tanh()
    
    # size: [T, P, 2, 2] x [2, 3] -> [T, P, 2, 3]
    A_xy = torch.matmul(alpha, alpha_weights)
    A_xy[:, :, :, A_xy.size(2) // 2] = 1.0
    
    # size: [T, P, 2, 3] -> [T, P, 1, 3], [T, P, 1, 3]
    A_x, A_y = torch.chunk(A_xy, 2, 2)
    
    # size: [T, P, 1, 3], [T, P, 1, 3] -> [T, P, 3, 3]
    A = torch.einsum('tzai,tzbj->tzij', (A_x, A_y))
    
    # size: [T, P, 3, 3] -> [T, P, I1, I2]
    A_pad = F.pad(A, (0, x.size(-2) - 3, 0, x.size(-1) - 3))
    A_roll = torch.roll(A_pad, (-1, -1), (-2, -1))  # these numbers are specific
    
    return A_roll, alpha, alpha_weights


@torch.jit.script
def parameterized_weights_backward(alpha_tanh, alpha_weights, Agrad):
    # (forward pass recomputation)
    # size: [T, P, 2, 2] x [2, 3] -> [T, P, 2, 3] -> [T, P, 2, 2]
    A_xy = torch.matmul(alpha_tanh, alpha_weights)[..., 0:3:2]
    
    # (forward pass recomputation)
    # size: [T, P, 2, 2] -> [T, P, 1, 2], [T, P, 1, 2]
    A_x, A_y = torch.chunk(A_xy, 2, 2)
    
    # ∇Ax = Ay * (∇A)^T
    # size: [T, P, 1, 2] x [T, P, 2, 3] -> [T, P, 1, 3]
    Ax_grad = torch.matmul(A_y, Agrad.transpose(-2, -1)[:, :, 0:3:2, :])
    
    # ∇Ay = Ax * ∇A
    # size: [T, P, 1, 2] x [T, P, 2, 3] -> [T, P, 1, 3]
    Ay_grad = torch.matmul(A_x, Agrad[:, :, 0:3:2, :])
    
    # size: [T, P, 1, 3], [T, P, 1, 3] -> [T, P, 2, 3]
    Axy_grad = torch.cat((Ax_grad, Ay_grad), dim=2)
    
    # ∇α = ∇Axy * alpha_weights^T
    # size: [T, P, 2, 3] x [2, 3]^T -> [T, P, 2, 2]
    alpha_grad = torch.matmul(Axy_grad, alpha_weights.transpose(-2, -1))
    
    # size: [T, P, 2, 2]
    alpha_grad = alpha_grad * (1 - alpha_tanh ** 2)
    
    return alpha_grad


class GenerateParameterizedWeights(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, alpha, alpha_weights):
        A, alpha_tanh, alpha_weights = parameterized_weights_forward(x, alpha, alpha_weights)
        
        ctx.save_for_backward(alpha_tanh, alpha_weights)
        
        return A
    
    @staticmethod
    def backward(ctx, grad_output):
        alpha_tanh, alpha_weights = ctx.saved_tensors
        
        # size: [T, P, I1, I2] -x> [T, P, 3, 3]
        Agrad = grad_output[:, :, [[-1], [0], [1]], [[-1, 0, 1]]]
        
        alpha_grad = parameterized_weights_backward(alpha_tanh, alpha_weights, Agrad)
        
        return None, alpha_grad, None


generate_parameterized_weights = GenerateParameterizedWeights.apply


def autoregressive_circular(x, alpha, alpha_weights, dummy):
    """
    Computation of a 2D-AutoRegressive layer (with circular padding).
    """

    if x.size(-2) < alpha.size(1) * 2 + 1 or \
            x.size(-1) < alpha.size(1) * 2 + 1:
        return x

    # size: -> [T, P, I1, I2]
    A = generate_parameterized_weights(x, alpha, alpha_weights)

    for A_s in torch.chunk(A, A.size(1), 1):
        x = ar_circular_Autograd(x, torch.squeeze(A_s, 1))

    return x


def ar_circular_Autograd(x, a):
    X = torch.rfft(x, 2, onesided=False)  # size:[M, T, I1, I2, 2]
    A = torch.rfft(a, 2, onesided=False)  # size:[T, I1, I2, 2]
    Y = complex_division(X, A)            # size:[M, T, I1, I2, 2]
    y = torch.irfft(Y, 2, onesided=False) # size:[M, T, I1, I2]
    return y


class ARCircular(torch.autograd.Function):

    # x size: [M, T, I1, I2]
    # a size:[T, I1, I2]
    @staticmethod
    def forward(ctx, x, a):
        X = torch.rfft(x, 2, onesided=False)  # size:[M, T, I1, I2, 2]
        A = torch.rfft(a, 2, onesided=False)  # size:[T, I1, I2, 2]
        Y = complex_division(X, A)            # size:[M, T, I1, I2, 2]
        y = torch.irfft(Y, 2, onesided=False) # size:[M, T, I1, I2]

        ctx.save_for_backward(A, Y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        """
        {grad_a} * a^T    = - grad_y  * y^T
        [T, I1, I2]   * [T, I1, I2] = [M, T, I1, I2] * [M, T, I1, I2]

        a^T    * {grad_x}     = grad_y
        [T, I1, I2] * [M, T, I1, I2]   = [M, T, I1, I2]
        """
        A, Y = ctx.saved_tensors

        grad_Y = torch.rfft(grad_y, 2, onesided=False)
        intermediate = complex_division(grad_Y, A, trans_deno=True)               # size:[M,T,I1,I2]
        grad_x = torch.irfft(intermediate, 2, onesided=False)

        intermediate = - complex_multiplication(intermediate, Y, trans_deno=True) # size:[M,T,I1,I2]
        grad_a = torch.irfft(intermediate.sum(0), 2, onesided=False)              # size:[T,I1,I2]
        return grad_x, grad_a


@torch.jit.script
def complex_division(x, A, trans_deno: bool = False):
    a, b = torch.chunk(x, 2, -1)
    c, d = torch.chunk(A, 2, -1)

    if trans_deno:  
        res_l = (a * c - b * d) / (c * c + d * d)
        res_r = (b * c + a * d) / (c * c + d * d)
    else:  
        res_l = (a * c + b * d) / (c * c + d * d)
        res_r = (b * c - a * d) / (c * c + d * d)
        

    res = torch.zeros_like(x, device=A.device)
    i = torch.tensor([0], device=A.device)
    res.index_add_(-1,i,res_l)
    i = torch.tensor([1], device=A.device)
    res.index_add_(-1,i,res_r)

    return res


@torch.jit.script
def complex_multiplication(x, A, trans_deno: bool = False):
    a, b = torch.chunk(x, 2, -1)
    c, d = torch.chunk(A, 2, -1)

    if trans_deno:
        res_l = a * c + b * d
        res_r = b * c - a * d
    else: 
        res_l = a * c - b * d
        res_r = b * c + a * d

    res = torch.zeros_like(x, device=A.device)
    i = torch.tensor([0], device=A.device)
    res.index_add_(-1,i,res_l)
    i = torch.tensor([1], device=A.device)
    res.index_add_(-1,i,res_r)


    return res