from torch import nn
from torch.nn import functional as F
import torch


class SPADE(nn.Module):
    def __init__(self, inchannel,nfilt):
        super(SPADE,self).__init__()


        self.norm=nn.InstanceNorm2d(nfilt,affine=True)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(inchannel, nfilt, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )
        self.mlp_gamma= nn.Conv2d(nfilt, nfilt, 3, 1, 1)
        self.mlp_beta = nn.Conv2d(nfilt, nfilt, 3, 1, 1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out




class Multi_SPADE1(nn.Module):
    def __init__(self, ch1,nfilt):
        super(Multi_SPADE1,self).__init__()


        self.spade1=SPADE(ch1,nfilt)
        self.last=nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(nfilt, nfilt, 3, 1, 1)
        )
    def forward(self, x, label1):
        x=self.spade1(x,label1)
        out=self.last(x)
        return out


class Multi_SPADE_resblk1(nn.Module):
    def __init__(self, ch1,nfilt):
        super(Multi_SPADE_resblk1,self).__init__()
        self.MSBlk1=Multi_SPADE1(ch1,nfilt)
        self.MSBlk2=Multi_SPADE1(ch1,nfilt)
        self.MSBlk_res=Multi_SPADE1(ch1,nfilt)

    def forward(self, x, label1):

        x2=self.MSBlk1(x,label1)
        x2=self.MSBlk2(x2,label1)
        x_res=self.MSBlk_res(x,label1)
        out=x2+x_res
        return out




class Multi_SPADE3(nn.Module):
    def __init__(self, ch1,ch2,ch3,nfilt):
        super(Multi_SPADE3,self).__init__()


        self.spade1=SPADE(ch1,nfilt)
        self.spade2=SPADE(ch2,nfilt)
        self.spade3=SPADE(ch3,nfilt)
        self.last=nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(nfilt, nfilt, 3, 1, 1)
        )

    def forward(self, x, label1,label2,label3):

        x=self.spade1(x,label1)
        x=self.spade2(x,label2)
        x=self.spade3(x,label3)
        out=self.last(x)
        return out



class Multi_SPADE_resblk3(nn.Module):
    def __init__(self, ch1,ch2,ch3,nfilt):
        super(Multi_SPADE_resblk3,self).__init__()


        self.MSBlk1=Multi_SPADE3(ch1,ch2,ch3,nfilt)
        self.MSBlk2=Multi_SPADE3(ch1,ch2,ch3,nfilt)
        self.MSBlk_res=Multi_SPADE3(ch1,ch2,ch3,nfilt)

    def forward(self, x, label1,label2,label3):

        x2=self.MSBlk1(x,label1,label2,label3)
        x2=self.MSBlk2(x2,label1,label2,label3)
        x_res=self.MSBlk_res(x,label1,label2,label3)
        out=x2+x_res

        return out

class Multi_SPADE4(nn.Module):
    def __init__(self, ch1,ch2,ch3,ch4,nfilt):
        super(Multi_SPADE4,self).__init__()


        self.spade1=SPADE(ch1,nfilt)
        self.spade2=SPADE(ch2,nfilt)
        self.spade3=SPADE(ch3,nfilt)
        self.spade4=SPADE(ch4,nfilt)
        self.last=nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(nfilt, nfilt, 3, 1, 1)
        )

    def forward(self, x, label1,label2,label3,label4):

        x=self.spade1(x,label1)
        x=self.spade2(x,label2)
        x=self.spade3(x,label3)
        x=self.spade4(x,label4)
        out=self.last(x)
        return out



class Multi_SPADE_resblk4(nn.Module):
    def __init__(self, ch1,ch2,ch3,ch4,nfilt):
        super(Multi_SPADE_resblk4,self).__init__()

        self.MSBlk1=Multi_SPADE4(ch1,ch2,ch3,ch4,nfilt)
        self.MSBlk2=Multi_SPADE4(ch1,ch2,ch3,ch4,nfilt)
        self.MSBlk_res=Multi_SPADE4(ch1,ch2,ch3,ch4,nfilt)

    def forward(self, x, label1,label2,label3,label4):

        x2=self.MSBlk1(x,label1,label2,label3,label4)
        x2=self.MSBlk2(x2,label1,label2,label3,label4)
        x_res=self.MSBlk_res(x,label1,label2,label3,label4)
        out=x2+x_res

        return out



class Image_decoder(nn.Module):
    def __init__(self, nfilt=64, out_channel=1):
        super(Image_decoder, self).__init__()

        self.relu = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Sequential(nn.Linear(256, 512),
                                self.relu,
                                nn.Linear(512, 8*8*512),
                                self.relu,
                                )


        self.output = nn.Sequential(nn.Conv2d(nfilt, nfilt, 3, 1, 1),
                                    nn.PReLU(),
                                    nn.Conv2d(nfilt, nfilt, 3, 1, 1),
                                    nn.PReLU(),
                                    nn.Conv2d(nfilt, nfilt, 3, 1, 1),
                                    nn.Conv2d(nfilt, out_channel, 1, 1, 0),
                                    nn.Tanh()
                                    )

        self.MSpade2 = Multi_SPADE_resblk1(nfilt, nfilt * 2)
        self.CSpade2 = Multi_SPADE_resblk1(nfilt, nfilt * 2)

        self.deconv2 = nn.Sequential(nn.Conv2d(nfilt * 2, nfilt * 2, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 2, affine=True),
                                     self.relu,
                                     nn.ConvTranspose2d(nfilt * 2, nfilt * 1, kernel_size=4, stride=2,
                                                        padding=1),
                                     self.relu)  # 512+512, 16,16-># 512, 32,32

        self.MSpade3 = Multi_SPADE_resblk1(nfilt * 2, nfilt * 4)
        self.CSpade3 = Multi_SPADE_resblk1(nfilt * 2, nfilt * 4)

        self.deconv3 = nn.Sequential(nn.Conv2d(nfilt * 4, nfilt * 4, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 4, affine=True),
                                     self.relu,
                                     nn.ConvTranspose2d(nfilt * 4, nfilt * 2, kernel_size=4, stride=2,
                                                        padding=1),
                                     self.relu)  # 512+512, 16,16-># 512, 32,32

        self.MSpade4 = Multi_SPADE_resblk1(nfilt * 4, nfilt * 8)
        self.CSpade4 = Multi_SPADE_resblk1(nfilt * 4, nfilt * 8)

        self.deconv4 = nn.Sequential(nn.Conv2d(nfilt * 8, nfilt * 8, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 8, affine=True),
                                     self.relu,
                                     nn.ConvTranspose2d(nfilt * 8, nfilt * 4, kernel_size=4, stride=2,
                                                        padding=1),
                                     self.relu)  # 512+512, 16,16-># 512, 32,32

        self.MSpade5 = Multi_SPADE_resblk1(nfilt * 8, nfilt * 8)
        self.CSpade5 = Multi_SPADE_resblk1(nfilt * 8, nfilt * 8)

        self.deconv5 = nn.Sequential(nn.Conv2d(nfilt * 8, nfilt * 8, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 8, affine=True),
                                     self.relu,
                                     nn.ConvTranspose2d(nfilt * 8, nfilt * 8, kernel_size=4, stride=2,
                                                        padding=1),
                                     self.relu)  # 512+512, 16,16-># 512, 32,32

        self.MSpade6 = Multi_SPADE_resblk1(nfilt * 8, nfilt * 8)
        self.CSpade6 = Multi_SPADE_resblk1(nfilt * 8, nfilt * 8)

        self.deconv6 = nn.Sequential(nn.Conv2d(nfilt * 8, nfilt * 8, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 8, affine=True),
                                     self.relu,
                                     nn.ConvTranspose2d(nfilt * 8, nfilt * 8, kernel_size=4, stride=2,
                                                        padding=1),
                                     self.relu)  # 512+512, 16,16-># 512, 32,32

    def forward(self, z, x1,x2,x3,x4,x5,c1,c2,c3,c4,c5):

        y=self.fc1(z).view(-1,512,8,8)

        y = self.MSpade6(y, x5)  # 8
        y = self.CSpade6(y, c5)  # 8
        y = self.deconv6(y)  # 16

        y = self.MSpade5(y, x4)  # 16
        y = self.CSpade5(y, c4)  # 16
        y = self.deconv5(y)  # 32

        y = self.MSpade4(y, x3)
        y = self.CSpade4(y, c3)
        y = self.deconv4(y)

        y = self.MSpade3(y, x2)
        y = self.CSpade3(y, c2)
        y = self.deconv3(y)

        y = self.MSpade2(y, x1)
        y = self.CSpade2(y, c1)
        y = self.deconv2(y)

        out_img = self.output(y)

        return out_img


class Label_Encoder(nn.Module):
    def __init__(self, inchannel=5, num_filters=64):

        super(Label_Encoder, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        # self.relu = nn.ReLU(inplace=True)
        self.relu= nn.LeakyReLU(0.2)


        self.conv1= nn.Sequential(nn.Conv2d(11,num_filters, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters, num_filters, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters, affine=True),
                                  self.relu,
                                  nn.Conv2d(num_filters, num_filters, 4, 2, 1),
                                  nn.InstanceNorm2d(num_filters, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128


        self.conv1_1= nn.Sequential(nn.Conv2d(inchannel,num_filters, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters, num_filters, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters, affine=True),
                                  self.relu,
                                  nn.Conv2d(num_filters, num_filters, 4, 2, 1),
                                  nn.InstanceNorm2d(num_filters, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128


        self.conv2= nn.Sequential(nn.Conv2d(num_filters,num_filters*2, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters*2,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters*2, num_filters*2, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 2, affine=True),
                                  self.relu,
                                  nn.Conv2d(num_filters*2, num_filters*2, 4, 2, 1),
                                  nn.InstanceNorm2d(num_filters * 2, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128

        self.conv3= nn.Sequential(nn.Conv2d(num_filters*2,num_filters*4, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters*4,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters*4, num_filters*4, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 4, affine=True),
                                  self.relu,
                                  nn.Conv2d(num_filters*4, num_filters*4, 4, 2, 1),
                                  nn.InstanceNorm2d(num_filters * 4, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128


        self.conv4= nn.Sequential(nn.Conv2d(num_filters*4,num_filters*8, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters*8,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                  nn.Conv2d(num_filters*8, num_filters*8, 4, 2, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128

        self.conv5= nn.Sequential(nn.Conv2d(num_filters*8,num_filters*8, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters*8,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                  nn.Conv2d(num_filters*8, num_filters*8, 4, 2, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128




    def forward(self, x,sel):

        x1=self.conv1_1(x)
        x2=self.conv2(x1)
        x3=self.conv3(x2)
        x4=self.conv4(x3)
        x5=self.conv5(x4)

        if sel==1:
            return x5
        else:
            return x1,x2,x3,x4,x5

class Image_Encoder(nn.Module):
    def __init__(self, inchannel=5, num_filters=64):

        super(Image_Encoder, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        # self.relu = nn.ReLU(inplace=True)
        self.relu= nn.LeakyReLU(0.2)


        self.conv1= nn.Sequential(nn.Conv2d(inchannel,num_filters, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters, num_filters, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters, affine=True),
                                  self.relu,
                                  nn.Conv2d(num_filters, num_filters, 4, 2, 1),
                                  nn.InstanceNorm2d(num_filters, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128


        self.conv2= nn.Sequential(nn.Conv2d(num_filters,num_filters*2, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters*2,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters*2, num_filters*2, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 2, affine=True),
                                  self.relu,
                                  nn.Conv2d(num_filters*2, num_filters*2, 4, 2, 1),
                                  nn.InstanceNorm2d(num_filters * 2, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128

        self.conv3= nn.Sequential(nn.Conv2d(num_filters*2,num_filters*4, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters*4,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters*4, num_filters*4, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 4, affine=True),
                                  self.relu,
                                  nn.Conv2d(num_filters*4, num_filters*4, 4, 2, 1),
                                  nn.InstanceNorm2d(num_filters * 4, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128


        self.conv4= nn.Sequential(nn.Conv2d(num_filters*4,num_filters*8, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters*8,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                  nn.Conv2d(num_filters*8, num_filters*8, 4, 2, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128

        self.conv5= nn.Sequential(nn.Conv2d(num_filters*8,num_filters*8, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters*8,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                  nn.Conv2d(num_filters*8, num_filters*8, 4, 2, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128

        self.fc1 = nn.Sequential(nn.Linear(8*8*512, 512),
                                self.relu,
                                )
        self.mu = nn.Linear(512,256)
        self.std = nn.Linear(512,256)
        #
        # self.apply(lambda x: glorot(x, 0.2))
        # glorot(self.mu, 1.0)
        # glorot(self.std, 1.0)


    def forward(self, x):

        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x).view(-1, 8*8*512)

        x=self.fc1(x)
        mu=self.mu(x)
        std=self.std(x)

        return mu * 0.1, std * 0.01,
#




