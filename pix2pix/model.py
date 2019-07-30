#This file contains code that defines the network architecture

import torch.nn as nn
import torch


#BuildConv class is utility class to help build our Convolution layers
class BuildConv(nn.Module):

    def __init__(self , input_size , output_size , kernel_size = 4 , stride = 2 , padding = 1 , batch_norm = True, activation = True):
        super(BuildConv , self).__init__()

        self.Conv = nn.Conv2d(input_size,  output_size , kernel_size , stride , padding)
        self.activation = activation
        self.leaky_relu = nn.LeakyReLU(0.2 , True)
        self.batch_norm = batch_norm
        self.BatchNorm = nn.BatchNorm2d(output_size)

    def forward(self , x):

        if self.activation:
            op = self.Conv(self.leaky_relu(x))
        else:
            op = self.Conv(x)

        if self.batch_norm:

            return self.BatchNorm(op)
        else:
            return op




#BuildDeconv is a utility class to help build deconv layers

class BuildDeconv(nn.Module):

    def __init__(self , input_size , output_size , kernel_size = 4 , stride = 2 , padding = 1 , batch_norm = True , dropout = False):

        super(BuildDeconv ,self).__init__()

        self.DeConv = nn.ConvTranspose2d(input_size , output_size , kernel_size , stride , padding)
        self.BatchNorm = nn.BatchNorm2d(output_size)
        self.Dropout = nn.Dropout2d(0.5)
        self.relu = nn.ReLU(True)
        self.dropout = dropout
        self.batch_norm = batch_norm

    def forward(self , x):

        if self.batch_norm:

            op = self.BatchNorm(self.DeConv(self.relu(x)))
        else:
            op = self.DeConv(self.relu(x))
        
        if self.dropout:

            return self.Dropout(op)
        else:
            return op
    

    
#Let's define our Generator class (U-net architecture)

class Generator(nn.Module):

    def __init__(self ,input_dim , num_filters , output_dim):

        super(Generator , self).__init__()

        #Encoder
        #self.conv1 
        self.conv1 = BuildConv(input_dim , num_filters , batch_norm = True  , activation = False)
        self.conv2 = BuildConv(num_filters , num_filters*2)
        self.conv3 = BuildConv(num_filters*2 , num_filters*4)
        self.conv4 = BuildConv(num_filters*4 , num_filters*8)
        self.conv5 = BuildConv(num_filters*8 , num_filters*8)
        self.conv6 = BuildConv(num_filters*8 , num_filters*8)
        self.conv7 = BuildConv(num_filters*8 , num_filters*8)
        self.conv8 = BuildConv(num_filters*8 , num_filters*8 , batch_norm= False)

        #Decoder

        self.deconv1 = BuildDeconv(num_filters*8 , num_filters*8 , dropout= True)
        self.deconv2 = BuildDeconv(num_filters*8*2 , num_filters*8 , dropout= True)
        self.deconv3 = BuildDeconv(num_filters*8*2 , num_filters*8 , dropout= True)
        self.deconv4 = BuildDeconv(num_filters*8*2 , num_filters*8)
        self.deconv5 = BuildDeconv(num_filters*8*2 , num_filters*4)
        self.deconv6 = BuildDeconv(num_filters*4*2 , num_filters*2)
        self.deconv7 = BuildDeconv(num_filters*2*2 , num_filters)
        self.deconv8 = BuildDeconv(num_filters*2 , output_dim , batch_norm= False)

    def forward(self , x):

        #Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        enc7 = self.conv7(enc6)
        enc8 = self.conv8(enc7)

        #Decoder with skip connections
        dec1 = self.deconv1(enc8)
        dec1 = torch.cat([dec1, enc7], 1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc6], 1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc5], 1)
        dec4 = self.deconv4(dec3)
        dec4 = torch.cat([dec4, enc4], 1)
        dec5 = self.deconv5(dec4)
        dec5 = torch.cat([dec5, enc3], 1)
        dec6 = self.deconv6(dec5)
        dec6 = torch.cat([dec6, enc2], 1)
        dec7 = self.deconv7(dec6)
        dec7 = torch.cat([dec7, enc1], 1)
        dec8 = self.deconv8(dec7)
        out = torch.nn.Tanh()(dec8)
        return out

        

    def normal_weight_init(self , mean = 0.0 ,std = 0.02):
        ''' This function initializes weights of layers'''
        for m in self.children():

            if isinstance(m , BuildConv):
                nn.init.normal_(m.Conv.weight , mean , std)
            
            if isinstance(m , BuildDeconv):
                nn.init.normal_(m.DeConv.weight , mean , std)


#Lets define class Discriminator

class Discriminator(nn.Module):

    def __init__(self , input_dim , num_filters , output_dim):

        super(Discriminator , self).__init__()

        self.conv1 = BuildConv(input_dim , num_filters , batch_norm= False , activation= False)
        self.conv2 = BuildConv(num_filters , num_filters*2)
        self.conv3 = BuildConv(num_filters*2 , num_filters*4)
        self.conv4 = BuildConv(num_filters*4 , num_filters*8 , stride = 1)
        self.conv5 = BuildConv(num_filters*8 , output_dim , stride = 1 , batch_norm= False)

    def forward(self , x , label):

        x = torch.cat([x , label] , 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        op = torch.nn.Sigmoid()(x)

        return op

    def normal_weight_init(self , mean = 0.0 , std = 0.02):

        ''' This function initializes the weights for layers'''
        for m in self.children():

            if isinstance(m , BuildConv):
                nn.init.normal_(m.Conv.weight , mean , std)
