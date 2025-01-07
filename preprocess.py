import numpy as np


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def process_images(image_a, image_b, type_a, type_b):
    if type_a == 'opt' and type_b == 'opt':
        # If a and b are both optical images, use a specific preprocessing method
        image_a = normalize_img(image_a )
        image_b = normalize_img(image_b)
    elif (type_a == 'sar' and type_b == 'opt') or (type_a == 'opt' and type_b == 'sar'):
        # If one is sar and the other is opt, use a custom preprocessing method
        image_a = preprocess_img(image_a, type_a)
        image_b = preprocess_img(image_b, type_b)
    else:
        raise ValueError("Unknown image type")
    return image_a, image_b


def normalize_img(image):
    min_val = torch.min(image)
    max_val = torch.max(image)
    if max_val - min_val != 0:
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        small_constant = 1e-10
        normalized_image = (image - min_val) / (max_val - min_val + small_constant)
    return normalized_image
def preprocess_img(data, d_type):
    pps_data = data.detach().to(device).float()

    if d_type == 'opt':
        pps_data = stad_img(pps_data, channel_first=False)
    elif d_type == 'sar':
        pps_data[torch.abs(pps_data) <= 0] = torch.min(pps_data[torch.abs(pps_data) > 0]).float()
        pps_data = torch.log(pps_data + 1.0).float()
        pps_data = stad_img(pps_data, channel_first=False)
    return pps_data.float()


def stad_img(img, channel_first):
    img = img.float()  # Make sure the input image is of type float
    if channel_first:
        channel, img_height, img_width = img.shape
        img = img.view(channel, -1).float()
        mean = torch.mean(img, dim=1, keepdim=True).float()
        center = (img - mean).float()
        var = torch.sum(center**2, dim=1, keepdim=True).float() / (img_height * img_width)
        std = torch.sqrt(var).float()
        nm_img = (center / std).float()
        nm_img = nm_img.view(channel, img_height, img_width).float()
    else:
        img_height, img_width, channel = img.shape
        img = img.view(-1, channel).float()
        mean = torch.mean(img, axis=0, keepdim=True).float()
        center = (img - mean).float()
        var = torch.sum(center**2, axis=0, keepdim=True).float() / (img_height * img_width)
        std = torch.sqrt(var).float()
        nm_img = (center / std).float()
        nm_img = nm_img.view(img_height, img_width, channel).float()
    return nm_img.float()





