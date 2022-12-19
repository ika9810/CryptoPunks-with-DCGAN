from fastapi import FastAPI
from pydantic import BaseModel


import numpy as np
import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL
from PIL import Image
import base64
import io

app = FastAPI()

class Msg(BaseModel):
    msg: str

weights = torch.load("cryptopunks_generator.pth",map_location=torch.device('cpu'))
device = torch.device('cpu')

z_dim = 100       #noise
beta_1 = 0.5      #as specified in the original DCGAN paper
beta_2 = 0.999 
lr = 0.0002       #as specified in the original DCGAN paper
n_epochs = 100
batch_size = 128
image_size = 64

class Generator(nn.Module):       #signals neural network
    def __init__(self, 
                 z_dim=100,      #noise vector
                 im_chan=3,      #color chanel, 3 for red green blue
                 hidden_dim=64): #spatial size of feature map (conv)
        
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.im_chan = im_chan
        self.hidden_dim = hidden_dim
        
        self.generator_cnn = nn.Sequential(self.make_gen_block(z_dim, hidden_dim*8, stride=1, padding=0),   
                                           #(64*8) x 4 x 4
                                           self.make_gen_block(hidden_dim*8, hidden_dim*4),                           
                                           #(64*4) x 8 x 8
                                           self.make_gen_block(hidden_dim*4, hidden_dim*2),                           
                                           #(64*2) x 16 x 16
                                           self.make_gen_block(hidden_dim*2, hidden_dim),                             
                                           #(64) x 32 x 32
                                           self.make_gen_block(hidden_dim, im_chan, final_layer=True))
    
    def make_gen_block(self, 
                       im_chan,     #image dimension
                       op_chan,     #output dimension
                       kernel_size=4, 
                       stride=2, 
                       padding=1, 
                       final_layer=False): 
        
        layers = []
        #de-convolutional layer
        layers.append(nn.ConvTranspose2d(im_chan,     
                                         op_chan, 
                                         kernel_size, 
                                         stride, 
                                         padding, 
                                         bias=False))
        
        if not final_layer:
            layers.append(nn.BatchNorm2d(op_chan))
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)
    
    def forward(self,noise):
        x = noise.view(-1,self.z_dim,1,1)
        return self.generator_cnn(x)

    def get_noise(n_samples, 
                  z_dim, 
                  device='cpu'):
        return torch.randn(n_samples, 
                           z_dim, 
                           device=device)



def print_tensor_images(images_tensor):
    
    '''
    Function for visualizing images: Given a tensor of images, prints the images.
    '''
        
    plt.rcParams['figure.figsize'] = (15, 15)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    images_tensor = images_tensor.to('cpu')
    npimgs = images_tensor.detach().numpy()
    
    no_plots = len(images_tensor)

    for idx,image in enumerate(npimgs):
        plt.subplot(1, 8, idx+1)
        plt.axis('off')
        #dnorm
        image = image * 0.5 + 0.5     
        plt.imshow(np.transpose(image, (1, 2, 0)))

    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes,  format='png')
    pic_IObytes.seek(0)

    pic_hash = base64.b64encode(pic_IObytes.read())
    
    #base64 to PIL image
    msg = base64.b64decode(pic_hash)
    buf = io.BytesIO(msg)
    img = Image.open(buf)
    print(type(img))

    #crop image
    #135,485,240,590
    left = 188
    top = 685
    right = 334
    #right = 240
    #bottom = 590
    bottom = 830
    im1 = img.crop((left, top, right, bottom))

    #PIL image to base64
    buffered = io.BytesIO()
    im1.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    print(type(img_str))
    return img_str


@app.get("/")
async def root():
    #initialize generator
    generator = Generator(z_dim, 
                        im_chan=3, 
                        hidden_dim=64).to(device)



    generator.load_state_dict(weights)

    generator.eval()  

    sample_size=1

    for i in range(1):    
        
        #generate latent vectors
        fixed_z = Generator.get_noise(n_samples=sample_size, 
                                    z_dim=z_dim, 
                                    device=device)    
        
        #generate samples
        sample_image = generator(fixed_z)

        #display samples
        result = print_tensor_images(sample_image)

    return {"message": result}


@app.get("/path")
async def demo_get():
    return {"message": "This is /path endpoint, use a post request to transform the text to uppercase"}


@app.post("/path")
async def demo_post(inp: Msg):
    return {"message": inp.msg.upper()}


@app.get("/path/{path_id}")
async def demo_get_path_id(path_id: int):
    return {"message": f"This is /path/{path_id} endpoint, use post request to retrieve result"}