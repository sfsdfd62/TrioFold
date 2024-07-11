import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

CH_FOLD2 = 1

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel*4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel*4, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        #y = self.fc(y).view(b, c, 1, 1)
        y = self.fc(y)
        #div_y = y/y.sum(axis = 1,keepdims = True)
        #y = (div_y).view(b,c,1,1)
        y = y.view(b,c,1,1)
        channel_sum = torch.sum(x * y.expand_as(x), dim=1)
        #channel_sum = channel_sum*0.999

        return channel_sum
        #return x * y.expand_as(x)
        
        



class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=4):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel*reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel*reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        #output = max_out + avg_out
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=9,reduction=4,kernel_size=3):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
        self.conv2d = nn.Conv2d(in_channels=9,out_channels=1,kernel_size=5,padding = 2)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)

        channel_sum = self.conv2d(out)
        #channel_sum = torch.sum(out, dim=1)/9
        
        #one = torch.ones_like(channel_sum)
        #zero = torch.zeros_like(channel_sum)
        
        #channel_sum = torch.where(channel_sum > 1, one , channel_sum)
        #channel_sum = torch.where(channel_sum < 0, zero , channel_sum)

        channel_sum = torch.squeeze(channel_sum)
        

        #channel_sum = channel_sum*0.999
        
        return torch.transpose(channel_sum, -1, -2) * channel_sum
        
        
        
        
class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        #y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return torch.sum(x*y.expand_as(x), dim=1)
        

        
class CBAMBlock_less(nn.Module):

    def __init__(self, channel=4,reduction=4,kernel_size=3):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)
        self.conv2d = nn.Conv2d(in_channels=4,out_channels=1,kernel_size=5,padding = 2)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)

        channel_sum = self.conv2d(out)
        #channel_sum = torch.sum(out, dim=1)/9
        
        #one = torch.ones_like(channel_sum)
        #zero = torch.zeros_like(channel_sum)
        
        #channel_sum = torch.where(channel_sum > 1, one , channel_sum)
        #channel_sum = torch.where(channel_sum < 0, zero , channel_sum)

        channel_sum = torch.squeeze(channel_sum)
        

        #channel_sum = channel_sum*0.999
        
        return torch.transpose(channel_sum, -1, -2) * channel_sum     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        