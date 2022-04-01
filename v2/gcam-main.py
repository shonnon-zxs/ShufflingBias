# -*- coding: utf-8 -*-
"""
Created on 2019/8/4 上午9:53

@author: mick.yi

入口类

"""
import argparse
import os
import re
import opts
import cv2
import numpy as np
import torch
from skimage import io
from torch import nn
from torchvision import models
from base_model import Model
from interpretability.grad_cam import GradCAM, GradCamPlusPlus
from interpretability.guided_back_propagation import GuidedBackPropagation
from dataset_vqacp import Dictionary, VQAFeatureDataset
from torch.utils.data import DataLoader
import utils


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


@torch.no_grad()
def get_logits(model, dataloader):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    K = 36
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N or None).start()
    for v, b, q, a, i in iter(dataloader):
        bar.update(idx)
        batch_size = v.size(0)
        v = v.cuda()
        b = b.cuda()
        q = q.cuda()
        logits, att = model(q, v, None, None, False)
        pred[idx:idx + batch_size, :].copy_(logits.data)
        qIds[idx:idx + batch_size].copy_(i)
        idx += batch_size

    bar.update(idx)
    return pred, qIds


def get_net(net_name, weight_path=None):
    """
    根据网络名称获取模型
    :param net_name: 网络名称
    :param weight_path: 与训练权重路径
    :return:
    """
    pretrain = weight_path is None  # 没有指定权重路径，则加载默认的预训练权重
    if net_name in ['vgg', 'vgg16']:
        net = models.vgg16(pretrained=pretrain)
    elif net_name == 'vgg19':
        net = models.vgg19(pretrained=pretrain)
    elif net_name == 'best_model':
        net = Model(opt)
    elif net_name in ['resnet', 'resnet50']:
        net = models.resnet50(pretrained=pretrain)
    elif net_name == 'resnet101':
        net = models.resnet101(pretrained=pretrain)
    elif net_name in ['densenet', 'densenet121']:
        net = models.densenet121(pretrained=pretrain)
    elif net_name in ['inception']:
        net = models.inception_v3(pretrained=pretrain)
    elif net_name in ['mobilenet_v2']:
        net = models.mobilenet_v2(pretrained=pretrain)
    elif net_name in ['shufflenet_v2']:
        net = models.shufflenet_v2_x1_0(pretrained=pretrain)
    else:
        raise ValueError('invalid network name:{}'.format(net_name))
    # 加载指定路径的权重参数
    if weight_path is not None and net_name.startswith('densenet'):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(weight_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        net.load_state_dict(state_dict)
    elif weight_path is not None:
        net.load_state_dict(torch.load(weight_path))
    return net


def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name


def prepare_input(image):
    image = image.copy()

    # 归一化
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image -= means
    image /= stds

    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))  # channel first
    image = image[np.newaxis, ...]  # 增加batch维

    return torch.tensor(image, requires_grad=True)


def gen_cam(image, mask):
    """
    生成CAM图
    :param image: [H,W,C],原始图像
    :param mask: [H,W],范围0~1
    :return: tuple(cam,heatmap)
    """
    # mask转为heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb

    # 合并heatmap到原始图像
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)


def norm_image(image):
    """
    标准化图像
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)


def gen_gb(grad):
    """
    生guided back propagation 输入图像的梯度
    :param grad: tensor,[3,H,W]
    :return:
    """
    # 标准化
    grad = grad.data.numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb


def save_image(image_dicts, input_image_name, network, output_dir):
    prefix = os.path.splitext(input_image_name)[0]
    for key, image in image_dicts.items():
        io.imsave(os.path.join(output_dir, '{}-{}-{}.jpg'.format(prefix, network, key)), image)


def main(args):

    # 输入
    img = io.imread(args.image_path)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    inputs = prepare_input(img)
    q = 'How many people can be seen?'
    # 输出图像
    image_dict = {}
    n_device = torch.cuda.device_count()
    batch_size = opt.batch_size * n_device
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(opt.dataroot + 'dictionary.pkl')
    opt.ntokens = dictionary.ntoken
    eval_dset = VQAFeatureDataset('test', dictionary, opt.dataroot, opt.img_root, 1.0, adaptive=False)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)
    # 网络
    model = Model(opt)
    net = model.cuda()
    model_data = torch.load(args.weight_path)
    net.load_state_dict(model_data.get('model_state', model_data), False)
    # net = get_net(args.network, args.weight_path)
    # Grad-CAM

    layer_name = get_last_conv_name(net) if args.layer_name is None else args.layer_name
    
    grad_cam = GradCAM(net, layer_name)
    mask = grad_cam(args.class_id, eval_loader)  # cam mask
    image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
    grad_cam.remove_handlers()
    # Grad-CAM++
    grad_cam_plus_plus = GradCamPlusPlus(net, layer_name)
    mask_plus_plus = grad_cam_plus_plus(inputs, args.class_id)  # cam mask
    image_dict['cam++'], image_dict['heatmap++'] = gen_cam(img, mask_plus_plus)
    grad_cam_plus_plus.remove_handlers()

    # GuidedBackPropagation
    gbp = GuidedBackPropagation(net)
    inputs.grad.zero_()  # 梯度置零
    grad = gbp(inputs)

    gb = gen_gb(grad)
    image_dict['gb'] = norm_image(gb)
    # 生成Guided Grad-CAM
    cam_gb = gb * mask[..., np.newaxis]
    image_dict['cam_gb'] = norm_image(cam_gb)

    save_image(image_dict, os.path.basename(args.image_path), args.network, args.output_dir)


if __name__ == '__main__':
    opt = opts.parse_opt()
    main(opt)
