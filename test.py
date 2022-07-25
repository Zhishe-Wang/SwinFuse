# test phase
import os
import torch
from torch.autograd import Variable
from net import SwinFuse
import utils
from args_fusion import args
import numpy as np
import time
import cv2


def load_model(path, in_chans, out_chans):

    SwinFuse_model = SwinFuse(in_chans=in_chans, out_chans=out_chans)
    SwinFuse_model.load_state_dict(torch.load(path), False)

    para = sum([np.prod(list(p.size())) for p in SwinFuse_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(SwinFuse_model._get_name(), para * type_size / 1000 / 1000))

    SwinFuse_model.eval()
    SwinFuse_model.cuda()

    return SwinFuse_model


def run_demo(model, infrared_path, visible_path, output_path_root, index, f_type):
    img_ir, h, w, c = utils.get_test_images(infrared_path)
    img_vi, h, w, c = utils.get_test_images(visible_path)

    if c is 0:
        if args.cuda:
            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()
        img_ir = Variable(img_ir, requires_grad=False)
        img_vi = Variable(img_vi, requires_grad=False)
        # encoder
        tir3 = model.encoder(img_ir)
        tvi3 = model.encoder(img_vi)
        # fusion
        f = model.fusion(tir3, tvi3, f_type)
        # decoder
        img_fusion = model.up_x4(f)
        img_fusion = ((img_fusion / 2) + 0.5) * 255
    else:
        img_fusion_blocks = []
        for i in range(c):
            img_vi_temp = img_vi[i]
            img_ir_temp = img_ir[i]
            if args.cuda:
                img_vi_temp = img_vi_temp.cuda()
                img_ir_temp = img_ir_temp.cuda()
            img_vi_temp = Variable(img_vi_temp, requires_grad=False)
            img_ir_temp = Variable(img_ir_temp, requires_grad=False)
            # encoder
            tir3 = model.encoder(img_ir_temp)
            tvi3 = model.encoder(img_vi_temp)
            # fusion
            f = model.fusion(tir3, tvi3, f_type)
            # decoder
            img_fusion = model.up_x4(f)
            img_fusion = ((img_fusion / 2) + 0.5) * 255
            img_fusion_blocks.append(img_fusion)
        if 224 < h < 448 and 224 < w < 448:
            img_fusion_list = utils.recons_fusion_images1(img_fusion_blocks, h, w)
        if 448 < h < 672 and 448 < w < 672:
            img_fusion_list = utils.recons_fusion_images2(img_fusion_blocks, h, w)
        if 448 < h < 672 and 672 < w < 896:
            img_fusion_list = utils.recons_fusion_images3(img_fusion_blocks, h, w)
        if 224 < h < 448 and 448 < w < 672:
            img_fusion_list = utils.recons_fusion_images4(img_fusion_blocks, h, w)
        if 672 < h < 896 and 896 < w < 1120:
            img_fusion_list = utils.recons_fusion_images5(img_fusion_blocks, h, w)
        if 0 < h < 224 and 224 < w < 448:
            img_fusion_list = utils.recons_fusion_images6(img_fusion_blocks, h, w)
        if 0 < h < 224 and 448 < w < 672:
            img_fusion_list = utils.recons_fusion_images7(img_fusion_blocks, h, w)
        if h == 224 and 448 < w < 672:
            img_fusion_list = utils.recons_fusion_images8(img_fusion_blocks, h, w)
    ############################ multi outputs ##############################################
    output_count = 0
    for img_fusion in img_fusion_list:
        file_name = 'fusion' + '_' + str(index) + '_swinfuse_' + str(output_count) + '_' + 'f_type' + '.png'
        output_path = output_path_root + file_name
        output_count += 1
        # save images
        utils.save_image_test(img_fusion, output_path)
        print(output_path)


def main():
    # run demo
    test_path = "imgs/test-TNO/"
    # test_ir_path = "D:/Transformer  224Unet/imgs road sence/thermal/"
    # test_vis_path = "D:/Transformer  224Unet/imgs road sence/visual/"
    # test_ir_path = "D:/Transformer  224Unet/INO_TreesAndRunner/INO_TreesAndRunner_T/"
    # test_vis_path = "D:/Transformer  224Unet/INO_TreesAndRunner/INO_TreesAndRunner_Gray/"
    # test_ir_path = "D:/Transformer  224Unet/video/thermal/"
    # test_vis_path = "D:/Transformer  224Unet/video/visual/"

    network_type = 'SwinFuse'
    fusion_type = ['l1_mean']

    output_path = './outputs/'

    # in_c = 3 for RGB imgs; in_c = 1 for gray imgs
    in_chans = 1

    num_classes = in_chans
    mode = 'L'
    model_path = args.model_path_gray

    with torch.no_grad():
        print('SSIM weight ----- ' + args.ssim_path[1])
        ssim_weight_str = args.ssim_path[1]
        f_type = fusion_type[0]

        model = load_model(model_path, in_chans, num_classes)
        # begin = time.time()
        # for a in range(10):
        for i in range(25):
        # for i in range(1000, 1221):
        # for i in range(1000, 1040):
            index = i + 1
            infrared_path = test_path + 'IR' + str(index) + '.png'
            visible_path = test_path + 'VIS' + str(index) + '.png'
            # infrared_path = test_ir_path + 'roadscene' + '_' + str(index) + '.png'
            # visible_path = test_vis_path + 'roadscene' + '_' + str(index) + '.png'
            # infrared_path = test_ir_path + 'video' + '_' + str(index) + '.png'
            # visible_path = test_vis_path + 'video' + '_' +str(index) + '.png'
            run_demo(model, infrared_path, visible_path, output_path, index, f_type)
        # end = time.time()
        # print("consumption time of generating:%s " % (end - begin))
    print('Done......')


if __name__ == '__main__':
    main()