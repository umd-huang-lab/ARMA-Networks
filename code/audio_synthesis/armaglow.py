"""ARMAGlow, ARMA layer implementation for WaveGlow speech synthesis network

Adapted from WaveGlow implementation:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/waveglow/model.py
"""
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class ARMA1d(torch.nn.Module):
    def __init__(
            self,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            a_kernel_size=3,
            a_padding_mode="circular",
            a_padding=0,
            a_stride=1,
            a_dilation=1,
    ):
        """
        Initialization of a 1D-ARMA layer.
        """
        super(ARMA1d, self).__init__()

        # Uses 1d convolutions instead of 2d
        self.moving_average = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # Uses 1d convolutions instead of 2d
        self.autoregressive = AutoRegressive1d(
            channels,
            a_kernel_size,
            padding=a_padding,
            padding_mode=a_padding_mode,
            stride=a_stride,
            dilation=a_dilation,
        )

    def forward(self, x):
        """
        Compuation of the 1D-ARMA layer.
        """
        # shape
        batch_size, group_size, n_of_groups = x.size()

        W = self.moving_average.weight.squeeze()
        # Forward computation
        log_det_W = batch_size * n_of_groups * torch.logdet(W.unsqueeze(0).float()).squeeze()

        # size:[M, S, I1, I2]->[M, T, I1, I2]->[M, T, I1, I2]
        x = self.moving_average(x)
        x = self.autoregressive(x)
        return x, log_det_W


class AutoRegressive1d(torch.nn.Module):
    def __init__(
            self,
            channels,
            kernel_size=3,
            padding=0,
            padding_mode="circular",
            stride=1,
            dilation=1,
    ):
        """
        Initialization of 1D-AutoRegressive layer.
        """
        super(AutoRegressive1d, self).__init__()

        if padding_mode == "circular":
            self.a = AutoRegressiveCircular(channels, kernel_size, padding, stride, dilation)
        elif paddind_mode == "reflect":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, x):
        """
        Computation of 1D-AutoRegressive layer.
        """
        x = self.a(x)
        return x


class AutoRegressiveCircular(torch.nn.Module):
    def __init__(self, channels, kernel_size, padding=3, stride=1, dilation=1):
        """
        Initialization of a 1D-AutoRegressive layer (with circular padding).
        """
        super(AutoRegressiveCircular, self).__init__()
        # For 1d AR circular, we can just use Conv1d with circular padding
        self.alpha = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size // 2,
            padding=padding,
            padding_mode='circular',
            stride=stride,
            dilation=dilation
        )

    def forward(self, x):
        """
        Computation of the 1D-AutoRegressive layer (with circular padding).
        """
        return self.alpha(x)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(c, c, 1)
        W = W.contiguous()
        self.conv.weight.data = W

    def forward(self, z):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        # Forward computation
        log_det_W = batch_size * n_of_groups * torch.logdet(W.unsqueeze(0).float()).squeeze()
        z = self.conv(z)
        return z, log_det_W

    def infer(self, z):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if not hasattr(self, "W_inverse"):
            # Reverse computation
            W_inverse = W.float().inverse()
            W_inverse = Variable(W_inverse[..., None])
            if z.type() == "torch.cuda.HalfTensor" or z.type() == "torch.HalfTensor":
                W_inverse = W_inverse.half()
            self.W_inverse = W_inverse
        z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
        return z


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary
    difference from WaveNet is the convolutions need not be causal.  There is
    also no dilation size reset.  The dilation only doubles on each layer
    """

    def __init__(
        self,
        n_in_channels,
        n_mel_channels,
        n_layers,
        n_channels,
        kernel_size,
    ):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert n_channels % 2 == 0
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.cond_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name="weight")
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                    n_channels,
                    2 * n_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * n_channels, 1)
            cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")
            self.cond_layers.append(cond_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect = forward_input
        audio = self.start(audio)

        for i in range(self.n_layers):
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                self.cond_layers[i](spect),
                torch.IntTensor([self.n_channels]),
            )

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = res_skip_acts[:, : self.n_channels, :] + audio
                skip_acts = res_skip_acts[:, self.n_channels :, :]
            else:
                skip_acts = res_skip_acts

            if i == 0:
                output = skip_acts
            else:
                output = skip_acts + output
        return self.end(output)


class WaveGlow(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        n_flows,
        n_group,
        n_early_every,
        n_early_size,
        WN_config,
        arma=False,
    ):
        super(WaveGlow, self).__init__()

        self.upsample = torch.nn.ConvTranspose1d(n_mel_channels, n_mel_channels, 1024, stride=256)
        assert n_group % 2 == 0
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(n_group / 2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        if arma:
            Conv = lambda channels: ARMA1d(channels)
        else:
            Conv = lambda channels: Invertible1x1Conv(channels)
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Conv(n_remaining_channels))
            self.WN.append(WN(n_half, n_mel_channels * n_group, **WN_config))
        self.n_remaining_channels = n_remaining_channels

    def forward(self, forward_input):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        spect, audio = forward_input

        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        assert spect.size(2) >= audio.size(1)
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, : audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1)
        spect = spect.permute(0, 2, 1)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, : self.n_early_size, :])
                audio = audio[:, self.n_early_size :, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) // 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:]

            output = self.WN[k]((audio_0, spect))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list
