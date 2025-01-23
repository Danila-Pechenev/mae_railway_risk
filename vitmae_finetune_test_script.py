import sys

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('..')
import models_vit

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = models_vit.__dict__[arch](
        num_classes=2,
        global_pool=False,
    )
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    y = model(x.float())
    return y 


import argparse
import numpy as np
import torch

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning test for image classification', add_help=False)
    return parser


def main():

    #Charge model
    #################################################
    chkpt_dir_railsem = './finetune/finetune_railsem4_balanced_10000/best_model.pth'
    model_mae_classif_railsem = prepare_model(chkpt_dir_railsem, 'vit_base_patch16')
    #Charge image
    img_path = "../../data/images/2vsall_bl_2/test/safe/rs05862.jpg"#railsem balanced
    #Run one image
    img_true= Image.open(img_path)
    img_resized = img_true.resize((224, 224))
    img_float = np.array(img_resized) / 255.
    img_jpg = img_float[:,:,:3]

    assert img_jpg.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img_jpg - imagenet_mean
    img_tsf = img / imagenet_std
    
    result = run_one_image(img_tsf, model_mae_classif_railsem)
    # Optionally compute probabilities
    probabilities = torch.softmax(result, dim=1)
    print("Probabilities:", probabilities)

    # Get the indices of the max logits
    predicted_class_indices = torch.argmax(result, dim=1)
    print("Predicted class indices:", predicted_class_indices)

    # Assuming you know your class labels
    class_labels = ['risky', 'safe']
    predicted_classes = [class_labels[idx] for idx in predicted_class_indices]
    print("Predicted classes:", predicted_classes)



if __name__ == '__main__':
    main()
