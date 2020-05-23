import torch
import torch.nn.functional as F
import numpy as np
import cv2 as cv
from torch.distributions import Beta


def horisontal_flip(images, targets):
    # imgshow(targets, images,'1.jpg')
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    # imgshow(targets, images,'2.jpg')
    return images, targets

def vertical_flip(images, targets):
    # imgshow(targets, images,'1.jpg')
    images = torch.flip(images, [-2])
    targets[:, 3] = 1 - targets[:, 3]
    # imgshow(targets, images,'2.jpg')
    return images, targets

def rot(images, targets):
    # imgshow(targets, images,'1.jpg')
    images = torch.rot90(images, 1,[1,2])
    x = targets[:, 3]
    y = 1-targets[:, 2]
    targets[:, 2] = x
    targets[:, 3] = y
    term = targets[:, 4].clone()
    targets[:, 4] = targets[:, 5]
    targets[:, 5] = term
    # imgshow(targets, images,'2.jpg')
    return images, targets

def imgshow(targets,images,name):

    img = images.clone().permute(1,2,0)
    w = img.size(0)
    img = img.numpy().squeeze()*255
    img = img.astype('int32')

    for i in range(targets.size(0)):
        point = targets[i, 2:].numpy()*w
        img = cv.rectangle(img, (int(point[0] - (point[2] / 2)), int(point[1] - int(point[3] / 2))),(int(point[0] + (point[2] / 2)), int(point[1] + (point[3] / 2))), (0, 0, 255), 2)
    cv.imwrite(name, img)
    return

def imgshow_output(output,img,name):

    img = img.permute(1,2,0)
    w = img.size(0)
    img = img.cpu().numpy().squeeze()*255
    img = img.astype('int32')

    for i in range(output.size(0)):
        point = output[i, :4].numpy()
        img = cv.rectangle(img, (int(point[0]), int(point[1])),(int(point[2]), int(point[3] )), (0, 0, 255), 2)
    cv.imwrite(name, img)
    return
