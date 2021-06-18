from torch import nn
import torch


class Cloth_decoder(nn.Module):
    def __init__(self, nfilt=64,out_channel=1):
        super(Cloth_decoder, self).__init__()

        self.relu = nn.LeakyReLU(0.2)
        self.pool = nn.MaxPool2d(2, 2)


        self.output = nn.Sequential(nn.Conv2d(nfilt, nfilt, 3, 1, 1),
                                    nn.PReLU(),
                                    nn.Conv2d(nfilt, nfilt, 3, 1, 1),
                                    nn.PReLU(),
                                    nn.Conv2d(nfilt, nfilt, 3, 1, 1),
                                    nn.Conv2d(nfilt,out_channel, 1, 1, 0),
                                    )





        self.conv2=nn.Sequential(nn.Conv2d(nfilt * 2, nfilt * 1, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt*1),
                                     self.relu,
                                     nn.Conv2d(nfilt * 1, nfilt * 1, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 1),
                                     self.relu,
                                     )  # 512+512, 16,16-># 512, 32,32

        self.deconv2 = nn.Sequential(nn.Conv2d(nfilt * 1, nfilt * 1, 3, 1, 1),
                                     self.relu,
                                     nn.ConvTranspose2d(nfilt * 1, nfilt * 1, kernel_size=4, stride=2,
                                                        padding=1),
                                     self.relu)  # 512+512, 16,16-># 512, 32,32




        self.conv3=nn.Sequential(nn.Conv2d(nfilt * 2*2, nfilt * 2, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt*2),
                                     self.relu,
                                     nn.Conv2d(nfilt * 2, nfilt * 2, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 2),
                                     self.relu,
                                     )  # 512+512, 16,16-># 512, 32,32

        self.deconv3 = nn.Sequential(nn.Conv2d(nfilt * 2, nfilt * 2, 3, 1, 1),
                                     self.relu,
                                     nn.ConvTranspose2d(nfilt * 2, nfilt * 1, kernel_size=4, stride=2,
                                                        padding=1),
                                     self.relu)  # 512+512, 16,16-># 512, 32,32





        self.conv4=nn.Sequential(nn.Conv2d(nfilt * 4*2, nfilt * 4, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt*4),
                                     self.relu,
                                     nn.Conv2d(nfilt * 4, nfilt * 4, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 4),
                                     self.relu,
                                     )  # 512+512, 16,16-># 512, 32,32
        self.deconv4 = nn.Sequential(nn.Conv2d(nfilt * 4, nfilt * 4, 3, 1, 1),
                                     self.relu,
                                     nn.ConvTranspose2d(nfilt * 4, nfilt * 2, kernel_size=4, stride=2,
                                                        padding=1),
                                     self.relu)  # 512+512, 16,16-># 512, 32,32


        self.conv5=nn.Sequential(nn.Conv2d(nfilt * 8*2, nfilt * 8, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt*8),
                                     self.relu,
                                     nn.Conv2d(nfilt * 8, nfilt * 8, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 8),
                                     self.relu,
                                     )  # 512+512, 16,16-># 512, 32,32

        self.deconv5 = nn.Sequential(nn.Conv2d(nfilt * 8, nfilt * 8, 3, 1, 1),
                                     self.relu,
                                     nn.ConvTranspose2d(nfilt * 8, nfilt * 4, kernel_size=4, stride=2,
                                                        padding=1),
                                     self.relu)  # 512+512, 16,16-># 512, 32,32



        self.conv6=nn.Sequential(nn.Conv2d(nfilt * 8*2, nfilt * 8, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt*8),
                                     self.relu,
                                     nn.Conv2d(nfilt * 8, nfilt * 8, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 8),
                                     self.relu,
                                     )  # 512+512, 16,16-># 512, 32,32
        self.res61=nn.Sequential(nn.Conv2d(nfilt * 8, nfilt * 8, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt*8),
                                     self.relu,
                                     nn.Conv2d(nfilt * 8, nfilt * 8, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 8),
                                     )  # 512+512, 16,16-># 512, 32,32
        self.res62=nn.Sequential(nn.Conv2d(nfilt * 8, nfilt * 8, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt*8),
                                     self.relu,
                                     nn.Conv2d(nfilt * 8, nfilt * 8, 3, 1, 1),
                                     nn.InstanceNorm2d(nfilt * 8),
                                     )  # 512+512, 16,16-># 512, 32,32



        self.deconv6 = nn.Sequential(nn.Conv2d(nfilt * 8, nfilt * 8, 3, 1, 1),
                                     self.relu,
                                     nn.ConvTranspose2d(nfilt * 8, nfilt * 8, kernel_size=4, stride=2,
                                                        padding=1),
                                     self.relu)  # 512+512, 16,16-># 512, 32,32





    def forward(self, y,x1,x2,x3,x4,x5):

        y=self.conv6(torch.cat([y,x5],1))
        y=y+self.res61(y)
        y=y+self.res62(y)
        y=self.deconv6(y)

        y=self.conv5(torch.cat([y,x4],1))
        y=self.deconv5(y)

        y=self.conv4(torch.cat([y,x3],1))
        y=self.deconv4(y)

        y=self.conv3(torch.cat([y,x2],1))
        y=self.deconv3(y)

        y=self.conv2(torch.cat([y,x1],1))
        y=self.deconv2(y)

        out_img=self.output(y)

        return out_img



class Cloth_Encoder(nn.Module):
    def __init__(self, inchannel=6, num_filters=64):

        super(Cloth_Encoder, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        # self.relu = nn.ReLU(inplace=True)
        self.relu= nn.LeakyReLU(0.2)


        self.conv1= nn.Sequential(nn.Conv2d(inchannel,num_filters, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters, num_filters, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128


        self.conv2= nn.Sequential(nn.Conv2d(num_filters,num_filters*2, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters*2,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters*2, num_filters*2, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 2, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128

        self.conv3= nn.Sequential(nn.Conv2d(num_filters*2,num_filters*4, 3, 1, 1),
                                   nn.InstanceNorm2d(num_filters*4,affine=True),
                                   self.relu,
                                   nn.Conv2d(num_filters*4, num_filters*4, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 4, affine=True),
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128


        self.conv4= nn.Sequential(nn.Conv2d(num_filters*4,num_filters*8, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                   nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128

        self.conv5= nn.Sequential(nn.Conv2d(num_filters*8,num_filters*8, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                   nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1),
                                  nn.InstanceNorm2d(num_filters * 8, affine=True),
                                  self.relu,
                                  )  # 70, 256,256 -># 64, 256,256 -> pool -> 64, 128, 128




    def forward(self, x,sel):
        x1=self.pool(self.conv1(x))
        x2=self.pool(self.conv2(x1))
        x3=self.pool(self.conv3(x2))
        x4=self.pool(self.conv4(x3))
        x5=self.pool(self.conv5(x4))

        if sel==1:
            return x5
        else:
            return x1,x2,x3,x4,x5



