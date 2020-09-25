import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, input_channels, inter_channels = None, dimension = 2, 
                pairwise_function = "embedded_gaussian",
                sub_sampling = True, kernel_size = None,
                norm_type = None, norm_args = None):
        """
        Intialization of non-local block.

        Arguments:
        ----------
        input_channels: int
            The number of input channels to the non-local block.
            Note: The number of output channels is equal to the input one.

        inter_channels: int
            The number of intermediate channels within the non-local block.
            Note: If set to None, inter_channels = (input_channels + 1) // 2
            Default: None

        pairwise_function: str
            The pairwise function for the attention mechanism.
            Options: "embedded_gaussian", "gaussian", "dot_product", "concatenation".
            Default: "embedded_gaussian"

        dimension: int
            The dimension of the input/output features.
            Default: 2

        sub_sampling: bool
            Whether to apply sub-sampling (i.e. pooling) in the non-local block.
            Default: True

        kernel_size: int or a tuple of ints
            The kernel size for sub-sampling (i.e. pooling).
            Note: if set to None, it will be intialized as (2, ) * dimension
            Default: None 

        norm_type: None or str 
            The type of normalization layer following the linear embedding.
            Options: "batch", "instance", "layer", "group"
            Default: None, i.e. no normalization layer is used.

        norm_args: int or a tuple of ints
            The arguments to the normalization layer.
            The expected input shape of the layer normalization.
            Note: The argument is required for group or layer normalization.
            If norm_type == "layer", norm_args --> normalized_shape
                The expected shape for the input tensor.
            If norm_type == "group", norm_args --> num_groups
                The number of groups to separate the channels into.

        """
        super(_NonLocalBlockND, self).__init__()

        if inter_channels is None:
            inter_channels = (input_channels + 1) // 2

        assert dimension in [1, 2, 3]

        if dimension == 3:
            conv_nd, pool_nd = nn.Conv3d, nn.MaxPool3d
        elif dimension == 2:
            conv_nd, pool_nd = nn.Conv2d, nn.MaxPool2d
        else: # if dimension == 1:
            conv_nd, pool_nd = nn.Conv1d, nn.MaxPool1d

        assert pairwise_function in ["embedded_gaussian", 
            "gaussian", "dot_product", "concatenation"]

        self.g = conv_nd(input_channels, inter_channels, 1)
        if pairwise_function != "gaussian":
            self.phi   = conv_nd(input_channels, inter_channels, 1)
            self.theta = conv_nd(input_channels, inter_channels, 1)
        else: # if pairwise_function == "gaussian":
            self.phi, self.theta = nn.Identity(), nn.Identity()

        if sub_sampling:
            if kernel_size is None:
                kernel_size = (2, ) * dimension
            else: # if kernel_size is not None:
                assert isinstance(kernel_size, int) or len(kernel_size) == dimension

            self.g   = nn.Sequential(self.g,   pool_nd(kernel_size))
            self.phi = nn.Sequential(self.phi, pool_nd(kernel_size))

        if pairwise_function in ["embedded_gaussian", "gaussian"]:
            self.pairwise_function = lambda phi, theta: \
                F.softmax(torch.matmul(theta.permute(0, 2, 1), phi), dim = -1)
        elif pairwise_function == "dot_product":
            self.pairwise_function = lambda phi, theta: \
                torch.matmul(theta.permute(0, 2, 1), phi) / phi.size(-1)
        else: # if pairwise == "concatenation":
            concat_project = nn.Sequential(
                nn.Conv2d(inter_channels * 2, 1, 1, bias = False),
                nn.ReLU())

            def pairwise_function(phi, theta):
                batch_size = phi.size(0)

                theta = theta.view(batch_size, inter_channels, -1, 1)
                phi = phi.view(batch_size, inter_channels, 1, -1)
                theta = theta.repeat(1, 1, 1, phi.size(3))
                phi = phi.repeat(1, 1, theta.size(2), 1)

                nl_map = concat_project(torch.cat([theta, phi], dim = 1))
                nl_map = torch.squeeze(nl_map, dim = 1) / nl_map.size(-1)
                return nl_map

            self.pairwise_function = pairwise_function

        self.W = conv_nd(inter_channels, input_channels, 1)
        nn.init.zeros_(self.W.weight)
        nn.init.zeros_(self.W.bias)

        if norm_type is not None:
            assert norm_type in ["batch", "instance", "layer", "group"]

            if norm_type == "batch":
                if dimension == 3:
                    norm_nd = nn.BatchNorm3d(input_channels)
                elif dimension == 2:
                    norm_nd = nn.BatchNorm2d(input_channels)
                else: # if dimension == 1:
                    norm_nd = nn.BatchNorm1d(input_channels)
            elif norm_type == "instance":
                if dimension == 3:
                    norm_nd = nn.InstanceNorm3d(input_channels)
                elif dimension == 2:
                    norm_nd = nn.InstanceNorm2d(input_channels)
                else: # if dimension == 1:
                    norm_nd = nn.InstanceNorm1d(input_channels)
            elif norm_type == "layer":
                norm_nd = nn.LayerNorm(norm_args)
            else: # if norm_type == "group":
                norm_nd = nn.GroupNorm(norm_args, input_channels)
                
            self.W = nn.Sequential(self.W, norm_nd)

    def forward(self, inputs, return_nl_map = False):
        """
        Computation of the non-local block. 

        Arguments:
        ----------
        inputs: a (d+2)-th order tensor of size 
            [batch_size, input_channels, feature_size_1, ..., feature_size_d].
            Input to the non-local block.

        return_nl_map: bool
            Whether to return the non-local map as output.
            Default: False

        Returns:
        --------
        outputs: a (d+2)-th order tensor of size
            [batch_size, input_channels, feature_size_1, ..., feature_size_d].
            Output of the non-local block.

        nl_map: a 3-rd orde tensor of size 
            [batch_size, prod(feature_size_d), prod(feature_size_d/kernel_size_d)]
            The non-local map for the input.

        """
        batch_size, input_shape = inputs.size(0), inputs.size()[2:]

        # g: [batch_size, inter_channels, prod(feature_size_d/kernel_size_d)]
        g = self.g(inputs)
        g = g.view(batch_size, g.size(1), -1)
        g = g.permute(0, 2, 1)

        # phi: [batch_size, inter_channels, prod(feature_size_d/kernel_size_d)]
        phi = self.phi(inputs)
        phi = phi.view(batch_size, phi.size(1), -1)

        # theta: [batch_size, prod(feature_size_d), inter_channels]
        theta = self.theta(inputs)
        theta = theta.view(batch_size, theta.size(1), -1)

        # nl_map: [batch_size, prod(feature_size_d), prod(feature_size_d/kernel_size_d)]
        nl_map = self.pairwise_function(phi, theta)

        # y: [batch_size, inter_channels, feature_size_1, ..., feature_size_d]
        y = torch.matmul(nl_map, g)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, y.size(1), *input_shape)

        # outputs: [batch_size, input_channels, feature_size_1, .., feature_size_d]
        outputs = self.W(y) + inputs

        return (outputs, nl_map) if return_nl_map else outputs


class NonLocalBlock1D(_NonLocalBlockND):
    def __init__(self, input_channels, inter_channels = None,
                pairwise_function = "embedded_gaussian",
                sub_sampling = True, kernel_size = None, 
                norm_type = True, norm_args = None):
        """
        Initialization of 1D non-local block.

        """
        super(NonLocalBlock1D, self).__init__(
            input_channels, inter_channels = inter_channels,
            dimension = 1, pairwise_function = pairwise_function,
            sub_sampling = sub_sampling, kernel_size = kernel_size,
            norm_type = norm_type, norm_args = norm_args)


class NonLocalBlock2D(_NonLocalBlockND):
    def __init__(self, input_channels, inter_channels = None,
                pairwise_function = "embedded_gaussian",
                sub_sampling = True, kernel_size = None,
                norm_type = True, norm_args = None):
        """
        Initialization of 2D non-local block.
    
        """
        super(NonLocalBlock2D, self).__init__(
            input_channels, inter_channels = inter_channels, 
            dimension = 2, pairwise_function = pairwise_function,
            sub_sampling = sub_sampling, kernel_size = kernel_size,
            norm_type = norm_type, norm_args = norm_args)


class NonLocalBlock3D(_NonLocalBlockND):
    def __init__(self, input_channels, inter_channels = None,
                pairwise_function = "embedded_gaussian",
                sub_sampling = True, kernel_size = None, 
                norm_type = True, norm_args = None):
        """
        Initialization of 3D non-local block.
        """
        super(NonLocalBlock3D, self).__init__(
            input_channels, inter_channels = inter_channels,
            dimension = 3, pairwise_function = pairwise_function,
            sub_sampling = sub_sampling, kernel_size = kernel_size,
            norm_type = norm_type, norm_args = norm_args)


if __name__ == '__main__':

    for pairwise_function in ["embedded_gaussian", 
        "gaussian", "dot_product", "concatenation"]: 
        print("pairwise function:", pairwise_function)

        for (norm_type, norm_args) in [(None, None), ("batch", None), 
            ("instance", None), ("group", 3), ("layer", (20, ))]:
            print("normalizaton layer:", norm_type)

            for (sub_sampling, kernel_size) in [(True, 2), (False, None)]:
                print("sub_sampling =", sub_sampling, "; kernel_size =", kernel_size)

                # 1D non-local block
                img = torch.zeros(2, 3, 20)
                net = NonLocalBlock1D(3, 
                    pairwise_function = pairwise_function,
                    sub_sampling = sub_sampling, kernel_size = kernel_size, 
                    norm_type = norm_type, norm_args = norm_args)
                out = net(img)
                print(out.size())

                # 2D non-local block
                img = torch.zeros(2, 3, 20, 20)
                net = NonLocalBlock2D(3,
                    pairwise_function = pairwise_function,
                    sub_sampling = sub_sampling, kernel_size = kernel_size, 
                    norm_type = norm_type, norm_args = norm_args)
                out = net(img)
                print(out.size())

                # 3D non-local block
                img = torch.randn(2, 3, 8, 20, 20)
                net = NonLocalBlock3D(3,
                    pairwise_function = pairwise_function,
                    sub_sampling = sub_sampling, kernel_size = kernel_size, 
                    norm_type = norm_type, norm_args = norm_args)
                out = net(img)
                print(out.size())
