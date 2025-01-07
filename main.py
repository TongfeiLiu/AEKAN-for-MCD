import torch
import argparse
import os
import numpy as np
from skimage import io
from skimage.segmentation import slic, mark_boundaries
from skimage.filters.thresholding import threshold_otsu
from model.KAN import *
from loss_py import AEKAN_Loss
from model.KAN_model import Encoder, Decoder_a, Decoder_b
from preprocess import process_images
from preprocess_1 import process_images_1
from evaluation import Evaluation

# Setting command line parameters
parser = argparse.ArgumentParser(description="Image Segmentation and Model Training")
parser.add_argument('--root_dir', type=str, default='/opt/data/private/RSTeam/xj/GCN_KAN', help='Root directory path')
parser.add_argument('--load_data_dir', type=str, default='/data/dataset', help='Directory containing image datasets')
parser.add_argument('--result_folder', type=str, default='shuguang', help='Folder to save the result')
parser.add_argument('--T1_img', type=str, default='T1.png', help='Pre-event image name')
parser.add_argument('--T2_img', type=str, default='T2.png', help='Post-event image name')
parser.add_argument('--GT_img', type=str, default='GT.png', help='Reference Image Name')
parser.add_argument('--N_SEG', type=int, default=1100, help='Number of superpixels (N_SEG)')
parser.add_argument('--Com', type=float, default=20, help='Compactness parameter for SLIC')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer')
parser.add_argument('--img_t1_type1', type=str, default='sar', help='Pre-event image type')
parser.add_argument('--img_t2_type2', type=str, default='opt', help='Post-event image type')

args = parser.parse_args()
# Use equipment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loading image data
def load_images(data_path):
    a = io.imread(os.path.join(data_path, args.T1_img))
    b = io.imread(os.path.join(data_path, args.T2_img))
    ref = io.imread(os.path.join(data_path, args.GT_img))

    # Preventing reference image problems
    unique_values = np.unique(ref)
    if not np.array_equal(unique_values, [0, 255]):
        ref[ref < 200] = 0
        ref[ref >= 200] = 255
    else:
        ref = ref
    return torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), ref


# 训练和测试模型
def train(epoch):
    encoder.train()
    decoder_a.train()
    decoder_b.train()
    optimizer_e.zero_grad()
    optimizer_d_a.zero_grad()
    optimizer_d_b.zero_grad()
    total_loss = 0
    for _iter in range(obj_nums):
        node_t1 = node_set_t1[_iter].float().to(device)
        node_t2 = node_set_t2[_iter].float().to(device)
        x_a1, x_a2 = encoder(node_t1)
        rec_a1, rec_a2 = decoder_a(x_a2)
        x_b1, x_b2 = encoder(node_t2)
        rec_b1, rec_b2 = decoder_b(x_b2)
        loss = criterion(node_t1, node_t2, x_a1, x_a2, x_b1, x_b2, rec_a2, rec_b2, encoder, decoder_a, decoder_b)
        loss.backward()
        optimizer_e.step()
        optimizer_d_a.step()
        optimizer_d_b.step()
        total_loss += loss.item()
    avg_loss = total_loss / obj_nums
    weight_path = os.path.join(file_path, 'weight_epoch_%d.pth' % (epoch))
    model_dict = {
        'encoder': encoder.state_dict(),
        'decoder_a': decoder_a.state_dict(),
        'decoder_b': decoder_b.state_dict()
    }
    torch.save(model_dict, weight_path)
    print(f'Epoch [{epoch}], Loss: {avg_loss}')


def test(epoch):
    encoder.eval()
    decoder_a.eval()
    decoder_b.eval()
    with torch.no_grad():
        diff_set = []
        for _iter in range(obj_nums):
            node_t1 = node_set_t1[_iter].float().to(device)
            node_t2 = node_set_t2[_iter].float().to(device)
            x_a1, x_a2 = encoder(node_t1)
            x_b1, x_b2 = encoder(node_t2)
            difference_map_x = F.mse_loss(x_a2, x_b2)
            diff_set.append(difference_map_x)

    if ref.shape[-1] == 3:
        diff_map = torch.zeros((height, width, channel_t1), device=device)
    else:
        diff_map = torch.zeros((height, width), device=device)

    for i in range(0, obj_nums):
        diff_map[objects == i] = diff_set[i]

    diff_map_min = diff_map.min()
    diff_map_max = diff_map.max()
    if diff_map_max - diff_map_min != 0:
        diff_map_nor = (diff_map - diff_map_min) / (diff_map_max - diff_map_min)
    else:
        small_constant = 1e-10
        diff_map_nor = (diff_map - diff_map_min) / (diff_map_max - diff_map_min + small_constant)

    filename1 = f"DI_iter_{epoch}.png"
    filename2 = f"CM_iter_{epoch}.png"
    diff_map_save = diff_map_nor * 255
    diff_map_save = diff_map_save.cpu().numpy()
    io.imsave(os.path.join(file_path, filename1), diff_map_save.astype(np.uint8))
    thre1 = threshold_otsu(diff_map_save)
    CM1 = (diff_map_save >= thre1) * 255
    io.imsave(os.path.join(file_path, filename2), (CM1).astype(np.uint8))
    Indicators1 = Evaluation(ref, CM1)
    OA1, kappa1, AA1 = Indicators1.Classification_indicators()
    P1, R1, F11 = Indicators1.ObjectExtract_indicators()
    TP1, TN1, FP1, FN1 = Indicators1.matrix()

    file_name = "eval.txt"
    # Write the contents to a file
    file_path_txt = os.path.join(file_path, file_name)
    val_acc = open(file_path_txt, 'a')
    val_acc.write('===============================Parameters settings==============================\n')
    val_acc.write('=== epoch={} || superpixel Num={} || compact ={} ===\n'.format(epoch, args.N_SEG, args.Com))
    val_acc.write('Domain t1:\n')
    val_acc.write('TP={} || TN={} || FP={} || FN={}\n'.format(TP1, TN1, FP1, FN1))
    val_acc.write("\"OA\":\"" + "{}\"\n".format(OA1))
    val_acc.write("\"Kappa\":\"" + "{}\"\n".format(kappa1))
    val_acc.write("\"AA\":\"" + "{}\"\n".format(AA1))
    val_acc.write("\"Precision\":\"" + "{}\"\n".format(P1))
    val_acc.write("\"Recall\":\"" + "{}\"\n".format(R1))
    val_acc.write("\"F1\":\"" + "{}\"\n".format(F11))
    val_acc.close()


# Main training and testing process
for N_SEG in [args.N_SEG]:
    for Com in [args.Com]:
        for item in range(5):
            load_data = os.path.join(args.root_dir + args.load_data_dir)
            img_t1, img_t2, ref = load_images(load_data)

            if img_t1.shape[-1] == 3:
                height, width, channel_t1 = img_t1.shape
            else:
                height, width = img_t1.shape

            objects = slic(img_t2.cpu().numpy(), n_segments=N_SEG, compactness=Com, start_label=0)
            objects = torch.from_numpy(objects).to(device)
            di_sp_b = mark_boundaries(img_t2.cpu().numpy(), objects.cpu().numpy())
            file_path = os.path.join(args.result_folder, 'di_sp_%d_%f_%d' % (N_SEG, Com, item))

            if not os.path.exists(file_path):
                os.makedirs(file_path)
            io.imsave(os.path.join(file_path, 'di_sp_%s_slic%d_%f.bmp' % (args.result_folder, N_SEG, Com)),
                      (di_sp_b * 255).astype(np.uint8))
            img_t1_type1 = args.img_t1_type1
            img_t2_type2 = args.img_t2_type2
            if img_t1.dim()==3:
                # Three-channel image normalization
                img_t1, img_t2 = process_images(img_t1, img_t2, img_t1_type1, img_t2_type2)
            else:
                # Single channel image normalization
                img_t1, img_t2 = process_images_1(img_t1, img_t2, img_t1_type1, img_t2_type2)


            obj_nums = torch.max(objects) + 1
            node_set_t1 = []
            node_set_t2 = []
            for idx in range(obj_nums):
                obj_idx = objects == idx
                if img_t1.shape[-1] == 3:
                    node_set_t1.append(img_t1[obj_idx])
                    node_set_t2.append(img_t2[obj_idx])
                else:
                    node_set_t1.append(img_t1[obj_idx].reshape(-1, 1))
                    node_set_t2.append(img_t2[obj_idx].reshape(-1, 1))

            # Set the model according to the number of input image channels
            if img_t1.shape[-1] == 3:
                encoder = Encoder(in_channels=3, hidden_channels=3, out_channels=2).to(device)
                decoder_a = Decoder_a(in_channels=2, hidden_channels=3, out_channels=3).to(device)
                decoder_b = Decoder_b(in_channels=2, hidden_channels=3, out_channels=3).to(device)
            else:
                encoder = Encoder(in_channels=1, hidden_channels=3, out_channels=2).to(device)
                decoder_a = Decoder_a(in_channels=2, hidden_channels=3, out_channels=1).to(device)
                decoder_b = Decoder_b(in_channels=2, hidden_channels=3, out_channels=1).to(device)

            optimizer_e = torch.optim.AdamW(encoder.parameters(), lr=args.learning_rate, weight_decay=0.0001)
            optimizer_d_a = torch.optim.AdamW(decoder_a.parameters(), lr=args.learning_rate, weight_decay=0.0001)
            optimizer_d_b = torch.optim.AdamW(decoder_b.parameters(), lr=args.learning_rate, weight_decay=0.0001)
            criterion = AEKAN_Loss(lambda_reg=0.001)
            # model_dict = torch.load(
            #     '/shuguang/weight_epoch_4.pth') #Load our trained weights
            # encoder.load_state_dict(model_dict['encoder'])
            # decoder_a.load_state_dict(model_dict['decoder_a'])
            # decoder_b.load_state_dict(model_dict['decoder_b'])

            # Training and Testing
            for epoch in range(args.epochs):
                train(epoch)
                test(epoch)


