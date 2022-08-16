import os
import sys

import torch
import torch.nn as nn
import torchvision

def load_pretrained_model(ckpt_path):
    # create model
    model = torchvision.models.__dict__['resnet50']()

    # Load weights
    if os.path.isfile(ckpt_path):
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # remove online_encoder prefix
        state_dict = checkpoint['state_dict']
        #print(state_dict)
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.net.') and not k.startswith('module.net.fc'):
                # remove prefix
                state_dict[k[len("module.net."):]] = state_dict[k]
            # delete renamed or unused k
            if k.startswith('net.') and not k.startswith('net.fc'):
                # remove prefix
                state_dict[k[len("net."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(ckpt_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(ckpt_path))
   
    model.eval()
    return model
