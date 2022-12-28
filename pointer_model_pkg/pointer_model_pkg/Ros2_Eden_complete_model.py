#!/usr/bin/env /home/inbarm/dev_ws/src/HRI_ROS2/pointer_model_pkg/hri_ros2/bin/python3

import numpy as np
import torch
import cv2
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import random_split
import torchvision
from torchvision import models, transforms
from torch.nn import functional as func
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

PathToModle = '/home/inbarm/dev_ws/src/HRI_ROS2/pointer_model_pkg/pointer_model_pkg/PointerModel'

sys.path.insert(1, PathToModle+'/')
from L_Realtime_Seg_Depth_Main import get_ytest, getcoloredMask, make_image_types
from Lisa_Seg_model import UnetPlusPlus
import Lisa_Seg_util as utils
import warnings
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from model_msg.msg import ModleData

warnings.filterwarnings("ignore")

def countdown(time_factor):
    time.sleep(5)
    print(" ")
    print("Starts in:")
    for i in range(0, 3):
        print(3 - i)
        time.sleep(time_factor)
    print(" ")
    time.sleep(2)


class CNN(nn.Module):
    def __init__(self, device):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(12800, 128),
            nn.Sigmoid(),
            nn.Linear(128, 32),
            nn.Sigmoid()
        )  # 12800 , 4608
        self.fc2 = nn.Sequential(
            nn.Linear(12800, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.lin1 = nn.Linear(32, 2)
        self.lin2 = nn.Linear(128, 3)
        self.sigmoid = nn.Sigmoid()
        self.to(device)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # Flattening
        x = torch.flatten(x, 1)
        output1 = self.fc1(x)
        output1 = self.lin1(output1)
        output1 = self.sigmoid(output1)
        output1 = 360 * (output1 - 0.5)  # Limit output to be at [-180, 180]
        output2 = self.fc2(x)
        output2 = self.lin2(output2)

        return output1, output2

def pde(in_frame, main_model, pre_model, mask_model, transform, device):
    original_size = in_frame.shape[0:2]
    # frame = cv2.flip(in_frame, 1)
    img_T, _ = make_image_types(in_frame)
    mask = get_ytest(img_T, mask_model, original_size)

    # only_seg = getcoloredMask(in_frame, mask)

    t_frame = transform(in_frame).to(device)
    with torch.no_grad():
        prediction = pre_model(t_frame)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=in_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    # only_depth = depth_map
    depth_map = getcoloredMask(depth_map, mask)

    transform2 = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    depth_map_T = cv2.cvtColor(depth_map, cv2.COLOR_BGR2RGB)
    depth_map_T = Image.fromarray(depth_map_T.astype(np.uint8), mode='RGB')
    depth_map_T = transform2(depth_map_T)
    depth_map_T = depth_map_T[None, :]
    depth_map_T = depth_map_T.to(device)

    with torch.no_grad():
        ang_pred, pos_pred = main_model(depth_map_T)

    ang_pred, pos_pred = ang_pred.cpu().numpy(), pos_pred.cpu().numpy()
    ang = ang_pred[0]
    xyz = pos_pred[0]
    result = np.concatenate((ang, xyz), axis=None)
    np.set_printoptions(suppress=True)
    return result, depth_map

class PublishModlePointerDirection(Node):                
    def __init__(self):
        super().__init__('PublishModlePointerDirection')
        self.publisher_ = self.create_publisher(ModleData, 'pub_model_data', 10)
        self.PublishModelData()

    def PublishModelData(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # # Main model:
        weights_path = PathToModle+'/checkpiont/POS_and_ANG_new2_raytuned_overlay_3ch.pth'
        main_model = CNN(device)
        main_model.load_state_dict(torch.load(weights_path, map_location=device))
        main_model.eval()

        # # Finger mask model:
        # Model, loss, optimizer
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        ENCODER = "vgg19_bn"
        # ENCODER_WEIGHTS = 'imagenet'
        ENCODER_WEIGHTS = None
        CLASSES = ['hand']
        ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
        # DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        in_channels = 3

        # create segmentation model with pretrained encoder
        mask_model = UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),
            activation=ACTIVATION,
            in_channels=in_channels)
        preprocessing_fn = utils.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
        path_to_lisas_weights = PathToModle+"/checkpiont/edge_best_checkpoint.pt"
        ENCODER_WEIGHTS = utils.get_state_dict(path_to_lisas_weights)
        mask_model.load_state_dict(ENCODER_WEIGHTS)
        mask_model.eval()

        # # Pre-processing (MiDAS):
        pre_model_type = 'DPT_Hybrid'
        pre_model = torch.hub.load("intel-isl/MiDaS", pre_model_type)
        pre_model.to(device)
        pre_model.eval()

        # # Pre-processing transform:
        pre_model_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if pre_model_type == "DPT_Large" or pre_model_type == "DPT_Hybrid":
            transform = pre_model_transforms.dpt_transform
        else:
            transform = pre_model_transforms.small_transform

        # Uncomment to open camera or input a video:
        cap = cv2.VideoCapture(0)
        # cap1 = cv2.VideoCapture(2)
        # cap = cv2.VideoCapture('pointing.mp4')
        # used to record the time when we processed last frame
        prev_frame_time = 0
        # used to record the time at which we processed current frame
        new_frame_time = 0
        start = 0
        datapoints = 10000
        while True:
        # for i in range(datapoints):
            success, frame = cap.read()
            # success1, frame1 = cap1.read()

            # if not success or not success1:
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break
            if start >= 10:
                # print(type(frame))
                result, md_frame = pde(frame, main_model, pre_model, mask_model, transform, device)
                # cv2.imshow('Frame',frame1)
                print(result)
                msg = ModleData()
                # define the linear x-axis velocity of /cmd_vel Topic parameter to 0.5
                msg.x_cord = float(result[2])
                msg.y_cord = float(result[3])
                msg.z_cord = float(result[4])
                msg.pitch = float(result[0])
                msg.yaw = float(result[1])
                self.publisher_.publish(msg)
                print(msg)
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps = int(fps)
                fps = str(fps)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(md_frame,
                            str([result[0], result[1]]),
                            (40, 45),
                            font, 0.7,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA)
                cv2.putText(md_frame,
                            str([result[2], result[3], result[4]]),
                            (40, 85),
                            font, 0.7,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA)
                cv2.imshow('my model', md_frame)
            start += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    puinter_direction_node = PublishModlePointerDirection()
    rclpy.spin(puinter_direction_node)
    puinter_direction_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()