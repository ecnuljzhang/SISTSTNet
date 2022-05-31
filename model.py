import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from attention import PAM_CAM_Layer
from model_utils import (
    get_same_padding_conv2d,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)

class MBConvBlock(nn.Module):
    
    """Mobile Inverted Residual Bottleneck Block.
    Args:
         (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    
    num_repeat=int(options['r']),
    kernel_size=int(options['k']),
    stride=[int(options['s'][0])],
    expand_ratio=int(options['e']),
    input_filters=int(options['i']),
    output_filters=int(options['o']),
    se_ratio=float(options['se']) if 'se' in options else None,
    id_skip=('noskip' not in block_string))
    """

    def __init__(self, 
                 kernel_size = 3,
                 stride=(1,1),
                 expand_ratio = 1,
                 input_filters = None,
                 output_filters = None,
                 se_ratio = 0.25,
                 id_skip = True,
                 image_size=None):
        super().__init__()
        # self._bn_mom = 1 - global_params.batch_norm_momentum # pytorch's difference from tensorflow
        # self._bn_eps = global_params.batch_norm_epsilon
        self.kernel_size = kernel_size
        self.input_filters = input_filters
        self.has_se = (se_ratio is not None) and (0 < se_ratio <= 1)
        self.id_skip = id_skip  # whether to use skip connection and drop connect
        # Expansion phase (Inverted Bottleneck)
        self.expand_ratio = expand_ratio
        self.output_filters = output_filters
        self.stride = stride
        self.se_ratio = se_ratio
        inp = self.input_filters  # number of input channels
        oup = self.input_filters * self.expand_ratio  # number of output channels
        if self.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._in0 = nn.InstanceNorm2d(num_features=oup, affine=True)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self.kernel_size
        s = self.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._in1 = nn.InstanceNorm2d(num_features=oup, affine=True)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self.input_filters * self.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=True)
        self._in2 = nn.InstanceNorm2d(num_features=final_oup, affine=True)
        self._swish = nn.LeakyReLU(0.2)

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """


        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._in0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._in1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._in2(x)
        x = self._swish(x)
        
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


# model = MBConvBlock(kernel_size = 3,
#                  stride=1,
#                  expand_ratio = 1,
#                  input_filters = 32,
#                  output_filters = 16,
#                  se_ratio = 0.25,
#                  id_skip = False,
#                  image_size=None)

# z = torch.randn(1,32,256,256)

# z_out = model(z)

class ConvLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 relu = False,
                 sigmoid= False,
                 tanh = False):
        super().__init__()
        self._conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride)
        self._pad = nn.ReflectionPad2d(padding=kernel_size // 2)
        self._in = nn.InstanceNorm2d(num_features=out_channels, affine=True)
        if relu:
            self._act = nn.ReLU()
        elif tanh:
            self._act = nn.Tanh()
        else:
            self._act = nn.LeakyReLU(0.2)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        x = self._conv(x)
        x = self._in(x)
        x = self._act(x)
        return x

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int, upsample: None):
        super().__init__()
        self._upsample = None if upsample is None else nn.UpsamplingBilinear2d(scale_factor=upsample)
        self._conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride)
        self._pad = nn.ReflectionPad2d(padding=kernel_size // 2)
        self._in = nn.InstanceNorm2d(num_features=out_channels, affine=True)
        self._lrelu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._upsample is not None:
            x = self._upsample(x)
        x = self._pad(x)
        x = self._conv(x)
        x = self._in(x)
        x = self._lrelu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super(ResidualBlock, self).__init__()
        self._conv1 = ConvLayer(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, stride=1)
        self._in1 = nn.InstanceNorm2d(num_features=channels, affine=True)
        self._lrelu = nn.LeakyReLU(0.2)
        self._conv2 = ConvLayer(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, stride=1)
        self._in2 = nn.InstanceNorm2d(num_features=channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self._lrelu(self._in1(self._conv1(x)))
        out = self._in2(self._conv2(out))
        out = out + residual
        out = self._lrelu(out)
        return out


class EncoderModel(nn.Module):
    def __init__(self, input_channels = 1, base_output_channels = 32 ):
        super(EncoderModel, self).__init__()
        
        self.conv0 = ConvLayer(input_channels, base_output_channels, kernel_size= 3,stride = 1)
        self.mbconv0 = MBConvBlock(input_filters=int(base_output_channels), output_filters=int(base_output_channels*2))
        self.conv1 = ConvLayer(int(base_output_channels*2), int(base_output_channels*2), kernel_size= 3,stride = 1)
        self.mbconv1 = MBConvBlock(stride=2, input_filters=int(base_output_channels*2), output_filters=int(base_output_channels*4))
        self.conv2 = ConvLayer(int(base_output_channels*4), int(base_output_channels*4), kernel_size= 3,stride = 1)
        # self.mbconv2 = MBConvBlock(stride=2, input_filters=int(base_output_channels*4), output_filters=int(base_output_channels*8))
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv0(x)
        x = self.mbconv0(x)
        x = self.conv1(x)
        x = self.mbconv1(x)
        x = self.conv2(x)
        # x = self.mbconv2(x)
        return x

# em =EncoderModel()

# z = torch.randn(1,1,64,64)

# [z1, z2, z3] = em(z)

class GeneratorModel(nn.Module):
    def __init__(self, target_channels = 1, source_channels = 3, output_channels = 3, base_output_channels = 16, num_res_blocks = 5, activ_tanh = False):
        super(GeneratorModel, self).__init__()
        
        self.TEN = EncoderModel(input_channels = target_channels, base_output_channels= base_output_channels)
        self.SPN = EncoderModel(input_channels = source_channels, base_output_channels= base_output_channels)
        
        self.conv0 = ConvLayer(int(base_output_channels*4*2), int(base_output_channels*4), kernel_size= 3,stride = 1)
        self.pam = PAM_CAM_Layer(int(base_output_channels*2), use_pam = True)
        self.cam = PAM_CAM_Layer(int(base_output_channels*2), use_pam = False)
        
        
        
        self.mbconv0 = MBConvBlock(stride=2, input_filters=int(base_output_channels*4), output_filters=int(base_output_channels*8))
        
        self.resblocks = nn.Sequential( *[ResidualBlock(int(base_output_channels*8)) for i in range(num_res_blocks) ]) 
            
        self.upsample1 = UpsampleConvLayer(int(base_output_channels*8), int(base_output_channels*4), kernel_size= 3,stride = 1, upsample=2)
        self.conv1 = ConvLayer(int(base_output_channels*4), int(base_output_channels*2), kernel_size= 3,stride = 1)
        
        self.fuse = ConvLayer(int(base_output_channels*2), int(base_output_channels*2), kernel_size= 3,stride = 1)
        
        
        self.upsample2 = UpsampleConvLayer(int(base_output_channels*2), int(base_output_channels*2), kernel_size= 3,stride = 1, upsample=2)
        self.conv2 = ConvLayer(int(base_output_channels*2), int(base_output_channels*2), kernel_size= 3,stride = 1)
        
        # self.upsample3 = UpsampleConvLayer(int(base_output_channels), int(base_output_channels//2), kernel_size= 3,stride = 1, upsample=2)
        # self.conv3 = ConvLayer(int(base_output_channels//2), int(base_output_channels//2), kernel_size= 3,stride = 1)
        
        self.conv31 = ConvLayer(int(base_output_channels*2), base_output_channels, kernel_size= 3,stride = 1)
        # self.conv41 = ConvLayer(int(base_output_channels), output_channels, kernel_size= 3,stride = 1)
       
        # self.conv32 = ConvLayer(int(base_output_channels*2), base_output_channels, kernel_size= 3,stride = 1)
        
        # if activ_tanh:
        # self.conv42 = ConvLayer(int(base_output_channels), 1, kernel_size= 3,stride = 1, relu = True)
        # else:
        #     self.conv42 = ConvLayer(int(base_output_channels), 1, kernel_size= 3,stride = 1, relu = True)
        # if activ_tanh:
        self.conv5 = ConvLayer(base_output_channels, output_channels, kernel_size= 3, stride = 1, tanh = True)
        # else:
        #     self.conv5 = ConvLayer(output_channels + 1, output_channels, kernel_size= 3, stride = 1, relu = True)
        
    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
         
         x1 = self.TEN(target)
         
         x2 = self.SPN(source)
         
         y = torch.cat([x1, x2], 1)
         
         x = self.conv0(y)
         
         # print(x.shape)
         pam = self.pam(x)
         cam = self.cam(x)
         
         sum_pam_cam = pam + cam
         
         x = self.mbconv0(x)
         
         x = self.resblocks(x)
         
         x = self.upsample1(x)
         x = self.conv1(x)
         
         x = self.fuse(x*sum_pam_cam)
         
         x = self.upsample2(x)
         x = self.conv2(x)
         
         # x = self.upsample3(x)
         x = self.conv31(x)
         # x_ = self.conv41(x_)
         
         # xb =  self.conv32(x)
         # xb =  self.conv42(xb)
         
         x = self.conv5(x)
         # x = self.conv4(x)
         return x
     
    def init_networks(self, weights_init):
        self.TEN.apply(weights_init)
        self.SPN.apply(weights_init)
        self.conv0.apply(weights_init)
        self.mbconv0.apply(weights_init)
        self.resblocks.apply(weights_init)
        self.upsample1.apply(weights_init)
        self.conv1.apply(weights_init)
        self.upsample2.apply(weights_init)
        self.conv2.apply(weights_init)
        # self.upsample3.apply(weights_init)
        # self.conv31.apply(weights_init)
        # self.conv32.apply(weights_init)
        # self.conv41.apply(weights_init)
        # self.conv42.apply(weights_init)
        self.conv5.apply(weights_init)
        self.fuse.apply(weights_init)
            
            

GM = GeneratorModel(base_output_channels = 32)
#################
# total = sum([par.nelement() for par in GM.parameters()])
# print("Number of parameter: %.2fM" % (total/1e6))
#################
target = torch.randn(64,1,64,64)
source = torch.randn(64,3,64,64)

z = GM(target, source)


         
        
       
        
