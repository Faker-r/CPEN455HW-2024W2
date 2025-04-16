import time
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import wandb
from utils import *
from model import * 
from dataset import *
from tqdm import tqdm
from pprint import pprint
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths




def get_fid():
    print('......sampling......')

    # args.obs = (3, 32, 32)
    # input_channels = args.obs[0]
    input_channels = 3
    
    # loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    # sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

    model = PixelCNN(nr_resnet=2, nr_filters=64, 
                input_channels=input_channels, nr_logistic_mix=5)

    # model = PixelCNN(nr_resnet=1, nr_filters=40, 
    #             input_channels=input_channels, nr_logistic_mix=5)

    if not torch.cuda.is_available():
        print("no cuda!!")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    
    model.load_state_dict(torch.load("models/pcnn_cpen455_from_scratch_2_99.pth"))
    print('model parameters loaded')
    
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, 5)
    sample_t = sample(model, 16, (3, 32, 32), sample_op)
    sample_t = rescaling_inv(sample_t)
    save_images(sample_t, "samples")


    # sample_t = sample(model, args.sample_batch_size, args.obs, sample_op)
    # sample_t = rescaling_inv(sample_t)
    # save_images(sample_t, args.sample_dir)
    # sample_result = wandb.Image(sample_t, caption="epoch {}".format(epoch))

    gen_data_dir = "samples"
    ref_data_dir = "data" +'/test'
    paths = [gen_data_dir, ref_data_dir]
    try:
        fid_score = calculate_fid_given_paths(paths, 32, device, dims=192)
        print("Dimension {:d} works! fid score: {}".format(192, fid_score))
    except:
        print("Dimension {:d} fails!".format(192))

get_fid()