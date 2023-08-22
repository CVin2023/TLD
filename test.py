import torch
import sys

sys.path.append('./models')
import os
import cv2
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PL_dataset import PL_dataset
# from model.Student_Net import SNet
# from model.Student_Net_Segformer_b0 import SNet
# from model.Student_pvt import SNet
# from model.VGG_4layer import Net
# from model.VGG_4teacher import Net
# from model.VGG_net import Net
# from model.HAINet.model.HAI_models import HAIMNet_VGG
# from model_others.MobileSal.model import MobileSal
# from model_others.HRTransNet.model import HR_SwinNet
# from model_others.EGANet.network import Segment
# from model_others.LSNet.LSNet import UNet
from A_TLD.Model.BBS.BBSNet_model import BBSNet
# from proposed.model0 import Model
cfg = "train"
model = BBSNet()
# model = Segment()
# model = SNet()
# model = UNet()
# model = HAIMNet_VGG()
# model = Net()
# model = PvtNet()
# model = RGBD_sal()
from config import opt
# set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print('USE GPU:', opt.gpu_id)
# device = torch.device('cuda:1')
def get_files_list(raw_dir):
    files_list = []
    for filepath, dirnames, filenames in os.walk(raw_dir):
        for filename in filenames:
            files_list.append(filepath+'/'+filename)
    return files_list

test_dataset_path = opt.test_path

image_root = get_files_list(test_dataset_path + 'vl')
ti_root = get_files_list(test_dataset_path + 'ir')
gt_root = get_files_list(test_dataset_path + 'gt')

test_dataset = PL_dataset(image_root, ti_root, gt_root, is_train=False)

test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

print('test_dataset', len(test_dataset))
print('test_loader_size', len(test_loader))
# load the model
# model = UTA(cfg="train")
# Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
# model.load_state_dict(torch.load('/media/wby/F426D6F026D6B2BA/best_epoch.pth'))  ##/media/maps/shuju/osrn999et/
#
model.load_state_dict(torch.load('/Users/chuanmingji/Downloads/ImageFusion/A_TLD/BBS/Net_epoch_200.pth', map_location='cuda'))   #163nei 100  153
# model.load_state_dict(torch.load('/home/guoxiaodong/code/TLD/save_path/dilation_res/Net_epoch_180.pth', map_location='cuda'))   #163nei 100  153
# model.load_state_dict(torch.load('/home/jcm/PycharmProject/TLD/run1/Net_epoch_190.pth', map_location='cuda'))   #163nei 100  153

model.cuda()
model.eval()
print("==> Total params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6))

test_datasets = ['1']  # 'VT5000'
# test
test_mae = []
for dataset in test_datasets:
    mae_sum = 0
    save_path = './result/model0/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for n_iter, batch_data in enumerate(test_loader):
        with torch.no_grad():
            # print(len(batch_data))
            image, ti, labels, name = batch_data
            image = image.cuda()
            ti = ti.cuda()
            labels = labels.cuda()
            res = model(image, ti).cuda()
            name = str(name).replace('\'', "").replace('[', '').replace(']', '')
            predict = torch.sigmoid(res).cuda()
            predict = (predict - predict.min()) / (predict.max() - predict.min() + 1e-8)
            mae = torch.sum(torch.abs(predict - labels)) / torch.numel(labels)
            mae_sum = mae.item() + mae_sum
        predict = predict.data.cpu().numpy().squeeze()
        # print(predict.shape)
        print('save img to: ', save_path + name)
        cv2.imwrite(save_path + name, predict * 256)
    test_mae.append(mae_sum / len(test_loader))
print('Test Done!', 'MAE', test_mae)
