
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def process_images_1(image_a, image_b, type_a, type_b):
    if type_a == 'opt' and type_b == 'opt':
        image_a = normalize_img(image_a )
        image_b = normalize_img(image_b)
    elif (type_a == 'sar' and type_b == 'opt') or (type_a == 'opt' and type_b == 'sar'):
        image_a = preprocess_img_1(image_a, type_a)
        image_b = preprocess_img_1(image_b, type_b)
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
def preprocess_img_1(data, d_type):
    data = data.clone().detach().to(device).float()
    if d_type == 'opt':
        pps_data = stad_img(data)
    elif d_type == 'sar':
        data[torch.abs(data) <= 0] = torch.min(data[torch.abs(data) > 0])
        pps_data = torch.log(data + 1.0)
        pps_data = stad_img(pps_data)
    return pps_data

def stad_img(img):
    mean = torch.mean(img)
    std = torch.std(img)
    nm_img = (img - mean) / std
    return nm_img

