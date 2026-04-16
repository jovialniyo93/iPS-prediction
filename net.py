import torch
from torch import nn

'''UNet model'''

class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvOut(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvOut, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Unet(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Unet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Restoring the number of filters to the original values, starting from 32
        self.conv1 = DoubleConv(ch_in, 32)   # Restored from 16 to 32
        self.conv2 = DoubleConv(32, 64)      # Restored from 32 to 64
        self.conv3 = DoubleConv(64, 128)     # Restored from 64 to 128
        self.conv4 = DoubleConv(128, 256)    # Restored from 128 to 256
        self.conv5 = DoubleConv(256, 512)    # Restored from 256 to 512

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) 
        self.conv6 = DoubleConv(512, 256)    # Input: 512 from concat of upsampled and conv4
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(64, 32)
        self.conv10 = ConvOut(32, ch_out)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.maxpool(c1)
        c2 = self.conv2(p1)
        p2 = self.maxpool(c2)
        c3 = self.conv3(p2)
        p3 = self.maxpool(c3)
        c4 = self.conv4(p3)
        p4 = self.maxpool(c4)
        c5 = self.conv5(p4)

        up6 = self.up6(c5)
        concat6 = torch.cat([up6, c4], dim=1)
        c6 = self.conv6(concat6)

        up7 = self.up7(c6)
        concat7 = torch.cat([up7, c3], dim=1)
        c7 = self.conv7(concat7)

        up8 = self.up8(c7)
        concat8 = torch.cat([up8, c2], dim=1)
        c8 = self.conv8(concat8)

        up9 = self.up9(c8)
        concat9 = torch.cat([up9, c1], dim=1)
        c9 = self.conv9(concat9)

        result = self.conv10(c9)
        return result



'''FCN8s model'''
class FCN8s(nn.Module):

    def __init__(self, in_ch,n_class):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(in_ch, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)


    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        return h

'''DeepSeg model'''
import torch
import torch.nn.functional as F
from torch import nn

class Upsampling(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super(Upsampling, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 1x1 convolution for shortcut
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.conv1d(x)
        x = x1 + x2  # Shortcut connection
        return x

class DeepSeg(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DeepSeg, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Increase the number of channels in each layer for better performance
        self.res1 = ResBlock(n_channels, 32)  # First down block, increased from 16 to 32
        self.down1 = nn.MaxPool2d(2)
        self.res2 = ResBlock(32, 64)  # Second down block, increased from 32 to 64
        self.down2 = nn.MaxPool2d(2)
        self.res3 = ResBlock(64, 128)  # Third down block, increased from 64 to 128
        self.down3 = nn.MaxPool2d(2)
        self.res4 = ResBlock(128, 256)  # Bottleneck, increased from 128 to 256

        self.up1 = Upsampling(256, bilinear)  # First up block
        self.res5 = ResBlock(256 + 128, 128)  # Adjusted input channels
        self.up2 = Upsampling(128, bilinear)  # Second up block
        self.res6 = ResBlock(128 + 64, 64)   # Adjusted input channels
        self.up3 = Upsampling(64, bilinear)  # Third up block
        self.res7 = ResBlock(64 + 32, 32)    # Adjusted input channels
        self.conv_out = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Down-sampling path
        x1 = self.res1(x)
        x2 = self.down1(x1)
        x2 = self.res2(x2)
        x3 = self.down2(x2)
        x3 = self.res3(x3)
        x4 = self.down3(x3)
        x4 = self.res4(x4)

        # Up-sampling path
        x5 = self.up1(x4, x3)
        x5 = self.res5(x5)
        x6 = self.up2(x5, x2)
        x6 = self.res6(x6)
        x7 = self.up3(x6, x1)
        x7 = self.res7(x7)

        # Output layer
        result = self.conv_out(x7)
        return result

'''TransDeepSeg model'''
import torch
import torch.nn as nn

# Residual Block (RB)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # To match dimensions if needed
    
    def forward(self, x):
        residual = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

# Transformer Block (TB)
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4, dim_feedforward=512):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.linear1 = nn.Linear(in_channels, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, in_channels)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Self-attention mechanism
        x2 = self.norm1(x)
        attn_output, _ = self.self_attn(x2, x2, x2)
        x = x + attn_output
        # Feedforward network
        x2 = self.norm2(x)
        x2 = self.linear2(self.relu(self.linear1(x2)))
        x = x + x2
        return x

# Down-Sampling (DS)
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

# Up-Sampling (US)
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.conv(x)

# TransDeepSeg Model (with Residual Blocks and Transformer Blocks)
class TransDeepSeg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransDeepSeg, self).__init__()

        # Encoder part (down-sampling)
        self.rb1 = ResidualBlock(in_channels, 64)
        self.rb2 = ResidualBlock(64, 128)
        self.rb3 = ResidualBlock(128, 256)
        self.rb4 = ResidualBlock(256, 512)

        self.maxpool = nn.MaxPool2d(2)

        # Bottleneck with Transformer Block
        self.transformer_block = TransformerBlock(512)

        # Decoder part (up-sampling)
        self.up3 = UpSample(512, 256)
        self.rb5 = ResidualBlock(512, 256)  # Skip connection from rb3

        self.up2 = UpSample(256, 128)
        self.rb6 = ResidualBlock(256, 128)  # Skip connection from rb2

        self.up1 = UpSample(128, 64)
        self.rb7 = ResidualBlock(128, 64)  # Skip connection from rb1

        # Final Output Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder path
        c1 = self.rb1(x)
        p1 = self.maxpool(c1)

        c2 = self.rb2(p1)
        p2 = self.maxpool(c2)

        c3 = self.rb3(p2)
        p3 = self.maxpool(c3)

        c4 = self.rb4(p3)

        # Get spatial dimensions before flattening
        b, c, h, w = c4.size()

        # Bottleneck with transformer block
        c4 = c4.flatten(2).permute(2, 0, 1)  # Prepare for multi-head attention (seq_len, batch, channels)
        c4 = self.transformer_block(c4)
        c4 = c4.permute(1, 2, 0).view(b, c, h, w)  # Reshape back to feature map with dynamic spatial dims

        # Decoder path
        up3 = self.up3(c4)
        concat3 = torch.cat([up3, c3], dim=1)  # Skip connection from encoder
        c5 = self.rb5(concat3)

        up2 = self.up2(c5)
        concat2 = torch.cat([up2, c2], dim=1)  # Skip connection from encoder
        c6 = self.rb6(concat2)

        up1 = self.up1(c6)
        concat1 = torch.cat([up1, c1], dim=1)  # Skip connection from encoder
        c7 = self.rb7(concat1)

        # Final output layer
        output = self.final_conv(c7)
        return output

'''StarDistModel model'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Reduce memory: Apply 1x1 convolution to match input to output size only if needed
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_conv:
            residual = self.residual_conv(residual)

        x += residual
        x = self.relu(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, skip_connection):
        x = self.up(x)
        x = self.conv(x)
        
        # Padding to match dimensions if necessary
        diffY = skip_connection.size()[2] - x.size()[2]
        diffX = skip_connection.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenation along channel dimension
        x = torch.cat([x, skip_connection], dim=1)
        return x


class StarDistUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(StarDistUNet, self).__init__()
        
        # Downsampling path (light version to save memory)
        self.res1 = ResBlock(n_channels, 32)  # Reduced channels
        self.down1 = nn.MaxPool2d(2)
        
        self.res2 = ResBlock(32, 64)  # Reduced channels
        self.down2 = nn.MaxPool2d(2)
        
        self.res3 = ResBlock(64, 128)  # Reduced channels
        self.down3 = nn.MaxPool2d(2)
        
        self.res4 = ResBlock(128, 256)  # Reduced channels
        self.down4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResBlock(256, 512)  # Reduced channels

        # Upsampling path
        self.up1 = UpsampleBlock(512, 256)
        self.res_up1 = ResBlock(512, 256)  # Reduced channels

        self.up2 = UpsampleBlock(256, 128)
        self.res_up2 = ResBlock(256, 128)  # Reduced channels

        self.up3 = UpsampleBlock(128, 64)
        self.res_up3 = ResBlock(128, 64)  # Reduced channels

        self.up4 = UpsampleBlock(64, 32)
        self.res_up4 = ResBlock(64, 32)  # Reduced channels

        # Final output layer
        self.output_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Downsampling path
        x1 = self.res1(x)
        x2 = self.down1(x1)
        
        x3 = self.res2(x2)
        x4 = self.down2(x3)
        
        x5 = self.res3(x4)
        x6 = self.down3(x5)
        
        x7 = self.res4(x6)
        x8 = self.down4(x7)

        # Bottleneck
        x_bottleneck = self.bottleneck(x8)

        # Upsampling path
        x = self.up1(x_bottleneck, x7)
        x = self.res_up1(x)

        x = self.up2(x, x5)
        x = self.res_up2(x)

        x = self.up3(x, x3)
        x = self.res_up3(x)

        x = self.up4(x, x1)
        x = self.res_up4(x)

        # Output
        output = self.output_conv(x)
        return output


'''Cellpose'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def batchconv(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1d = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.conv1d(x)
        x = x1 + x2
        x = self.bn(x)
        x = self.relu(x)
        return x

class Upsampling(nn.Module):
    def __init__(self, in_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x

class Cellpose(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(Cellpose, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Define the downsampling layers (same as before)
        self.res1 = ResBlock(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.res2 = ResBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.res3 = ResBlock(128, 256)
        
        # Define the upsampling layers
        self.up1 = Upsampling(256, bilinear=bilinear)
        self.res4 = ResBlock(384, 128)
        self.up2 = Upsampling(128, bilinear=bilinear)
        self.res5 = ResBlock(192, 64)
        
        # Single output (instead of logits and edges)
        self.conv_out = nn.Conv2d(64, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        # Downsampling path
        x1 = self.res1(x)
        x2 = self.down1(x1)
        x2 = self.res2(x2)
        x3 = self.down2(x2)
        x3 = self.res3(x3)

        # Upsampling path
        x4 = self.up1(x3, x2)
        x4 = self.res4(x4)
        x5 = self.up2(x4, x1)
        x6 = self.res5(x5)

        # Single output
        output = self.conv_out(x6)
        return output


'''AttU_Net'''
import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        # Reduce the number of filters to save memory
        n1 = 64  # Halved the base number of filters
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        # Downsampling path
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # Upsampling path
        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=16)  # Reduce internal channels
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Downsample
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # Upsample
        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        # Final output
        out = self.Conv(d2)
        return out

'''NestedUnet'''
import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvOut(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvOut, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class NestedUNet(nn.Module):
    def __init__(self, in_channel, out_channel, deepsupervision=False):
        super(NestedUNet, self).__init__()

        self.deepsupervision = deepsupervision

        # Reduce the number of filters to save memory
        nb_filter = [32, 64, 128, 256, 512]  # Reduce the base number of filters

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Down-sampling path
        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        # Up-sampling path with concatenation
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

    def forward(self, input):
        # Downsample path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return (output1 + output2 + output3 + output4) / 4
        else:
            output = self.final(x0_4)
            return output
