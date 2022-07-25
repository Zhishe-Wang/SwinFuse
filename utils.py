import os
import torch
import torch.nn as nn
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from args_fusion import args
from scipy.misc import imread, imsave, imresize
import matplotlib as mpl
import imageio


def list_images(dimgectory):
    images = []
    names = []
    dimg = listdir(dimgectory)
    dimg.sort()
    for file in dimg:
        name = file.lower()
        images.append(join(dimgectory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def make_floor(path1,path2):
    path = os.path.join(path1,path2)
    if os.path.exists(path) is False:
        os.makedirs(path)
    return path


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 1).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 1).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U, D, V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    print(img_fusion.shape)
    imsave(output_path, img_fusion)


def get_image(path, height=224, width=224, flag=False):
    image = imread(path, mode='L')
    image = (image-127.5)/127.5

    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')
    return image


def get_train_images_auto(paths, height=224, width=224, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width,  flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    images = (images-127.5)/127.5
    return images


def get_test_images(paths, height=None, width=None, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width,  flag)
        if height is not None and width is not None:
            image = imresize(image, [height, width], interp='nearest')
        base_size = 224
        h = image.shape[0]
        w = image.shape[1]
        if 1 * base_size < h < 2 * base_size and 1 * base_size < w < 2 * base_size:
            c = 4
            images = get_img_parts1(image, h, w)
        if 2*base_size<h < 3*base_size and 2*base_size< w < 3*base_size:
            c = 9
            images = get_img_parts2(image, h, w)
        if 2 * base_size < h < 3 * base_size and 3 * base_size < w < 4 * base_size:
            c = 12
            images = get_img_parts3(image, h, w)
        if 1 * base_size < h < 2 * base_size and 2 * base_size < w < 3 * base_size:
            c = 6
            images = get_img_parts4(image, h, w)
        if 3 * base_size < h < 4 * base_size and 4 * base_size < w < 5 * base_size:
            c = 20
            images = get_img_parts5(image, h, w)
        if 0 * base_size < h < 1 * base_size and 1 * base_size < w < 2 * base_size:
            c = 2
            images = get_img_parts6(image, h, w)
        if 0 * base_size < h < 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts7(image, h, w)
        if h == 1 * base_size and 2 * base_size < w < 3 * base_size:
            c = 3
            images = get_img_parts8(image, h, w)

    return images, h, w, c


def get_img_parts1(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 448-w, 0, 448-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[224:448, 0: 224]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[224:448, 224: 448]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    return images


def get_img_parts2(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 672-w, 0, 672-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[224:448, 0: 224]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[224:448, 224: 448]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[224:448, 448: 672]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[448:672, 0: 224]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[448:672, 224: 448]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[448:672, 448: 672]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    return images


def get_img_parts3(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 896-w, 0, 672-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:224, 672: 896]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[224:448, 0: 224]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[224:448, 224: 448]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[224:448, 448: 672]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[224:448, 672: 896]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[448:672, 0: 224]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    img10 = image[448:672, 224: 448]
    img10 = torch.reshape(img10, [1, 1, img10.shape[0], img10.shape[1]])
    img11 = image[448:672, 448: 672]
    img11 = torch.reshape(img11, [1, 1, img11.shape[0], img11.shape[1]])
    img12 = image[448:672, 672: 896]
    img12 = torch.reshape(img12, [1, 1, img12.shape[0], img12.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    images.append(img10.float())
    images.append(img11.float())
    images.append(img12.float())
    return images


def get_img_parts4(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 672-w, 0, 448-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[224:448, 0: 224]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[224:448, 224: 448]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[224:448, 448: 672]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    return images


def get_img_parts5(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 1120-w, 0, 896-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    img4 = image[0:224, 672: 896]
    img4 = torch.reshape(img4, [1, 1, img4.shape[0], img4.shape[1]])
    img5 = image[0:224, 896: 1120]
    img5 = torch.reshape(img5, [1, 1, img5.shape[0], img5.shape[1]])
    img6 = image[224:448, 0: 224]
    img6 = torch.reshape(img6, [1, 1, img6.shape[0], img6.shape[1]])
    img7 = image[224:448, 224: 448]
    img7 = torch.reshape(img7, [1, 1, img7.shape[0], img7.shape[1]])
    img8 = image[224:448, 448: 672]
    img8 = torch.reshape(img8, [1, 1, img8.shape[0], img8.shape[1]])
    img9 = image[224:448, 672: 896]
    img9 = torch.reshape(img9, [1, 1, img9.shape[0], img9.shape[1]])
    img10 = image[224:448, 896: 1120]
    img10 = torch.reshape(img10, [1, 1, img10.shape[0], img10.shape[1]])
    img11 = image[448:672, 0: 224]
    img11 = torch.reshape(img11, [1, 1, img11.shape[0], img11.shape[1]])
    img12 = image[448:672, 224: 448]
    img12 = torch.reshape(img12, [1, 1, img12.shape[0], img12.shape[1]])
    img13 = image[448:672, 448: 672]
    img13 = torch.reshape(img13, [1, 1, img13.shape[0], img13.shape[1]])
    img14 = image[448:672, 672: 896]
    img14 = torch.reshape(img14, [1, 1, img14.shape[0], img14.shape[1]])
    img15 = image[448:672, 896: 1120]
    img15 = torch.reshape(img15, [1, 1, img15.shape[0], img15.shape[1]])
    img16 = image[672:896, 0: 224]
    img16 = torch.reshape(img16, [1, 1, img16.shape[0], img16.shape[1]])
    img17 = image[672:896, 224: 448]
    img17 = torch.reshape(img17, [1, 1, img17.shape[0], img17.shape[1]])
    img18 = image[672:896, 448: 672]
    img18 = torch.reshape(img18, [1, 1, img18.shape[0], img18.shape[1]])
    img19 = image[672:896, 672: 896]
    img19 = torch.reshape(img19, [1, 1, img19.shape[0], img19.shape[1]])
    img20 = image[672:896, 896: 1120]
    img20 = torch.reshape(img20, [1, 1, img20.shape[0], img20.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    images.append(img4.float())
    images.append(img5.float())
    images.append(img6.float())
    images.append(img7.float())
    images.append(img8.float())
    images.append(img9.float())
    images.append(img10.float())
    images.append(img11.float())
    images.append(img12.float())
    images.append(img13.float())
    images.append(img14.float())
    images.append(img15.float())
    images.append(img16.float())
    images.append(img17.float())
    images.append(img18.float())
    images.append(img19.float())
    images.append(img20.float())

    return images


def get_img_parts6(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 448-w, 0, 224-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    return images


def get_img_parts7(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 672-w, 0, 224-h), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    return images


def get_img_parts8(image, h, w):
    pad = nn.ConstantPad2d(padding=(0, 672-w, 0, 224), value=0)
    image = torch.from_numpy(image)
    image = pad(image)
    images = []
    img1 = image[0:224, 0: 224]
    img1 = torch.reshape(img1, [1, 1, img1.shape[0], img1.shape[1]])
    img2 = image[0:224, 224: 448]
    img2 = torch.reshape(img2, [1, 1, img2.shape[0], img2.shape[1]])
    img3 = image[0:224, 448: 672]
    img3 = torch.reshape(img3, [1, 1, img3.shape[0], img3.shape[1]])
    images.append(img1.float())
    images.append(img2.float())
    images.append(img3.float())
    return images


def recons_fusion_images1(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:224, 0: 224] += img1
        img_f[:, 0:224, 224: w] += img2[:, 0:224, 0:w-224]
        img_f[:, 224:h, 0: 224] += img3[:, 0:h-224, 0:224]
        img_f[:, 224:h, 224: w] += img4[:, 0:h-224, 0:w-224]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images2(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:224, 0: 224] += img1
        img_f[:, 0:224, 224: 448] += img2
        img_f[:, 0:224, 448: w] += img3[:, 0:224, 0:w-448]
        img_f[:, 224:448, 0: 224] += img4
        img_f[:, 224:448, 224: 448] += img5
        img_f[:, 224:448, 448: w] += img6[:, 0:224, 0:w-448]
        img_f[:, 448:h, 0: 224] += img7[:, 0:h-448, 0:224]
        img_f[:, 448:h, 224: 448] += img8[:, 0:h-448, 0:224]
        img_f[:, 448:h, 448: w] += img9[:, 0:h-448, 0:w - 448]
        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images3(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:224, 0: 224] += img1
        img_f[:, 0:224, 224: 448] += img2
        img_f[:, 0:224, 448: 672] += img3
        img_f[:, 0:224, 672: w] += img4[:, 0:224, 0:w-672]
        img_f[:, 224:448, 0: 224] += img5
        img_f[:, 224:448, 224: 448] += img6
        img_f[:, 224:448, 448: 672] += img7
        img_f[:, 224:448, 672: w] += img8[:, 0:224, 0:w-672]
        img_f[:, 448:h, 0: 224] += img9[:, 0:h-448, 0:224]
        img_f[:, 448:h, 224: 448] += img10[:, 0:h - 448, 0:224]
        img_f[:, 448:h, 448: 672] += img11[:, 0:h - 448, 0:224]
        img_f[:, 448:h, 672: w] += img12[:, 0:h - 448, 0:w-672]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images4(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]

        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:224, 0: 224] += img1
        img_f[:, 0:224, 224: 448] += img2
        img_f[:, 0:224, 448: w] += img3[:, 0:224, 0:w-448]
        img_f[:, 224:h, 0: 224] += img4[:, 0:h-224, 0:224]
        img_f[:, 224:h, 224: 448] += img5[:, 0:h - 224, 0:224]
        img_f[:, 224:h, 448: w] += img6[:, 0:h - 224, 0:w-448]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images5(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]
        img4 = img_lists[3][i]
        img5 = img_lists[4][i]
        img6 = img_lists[5][i]
        img7 = img_lists[6][i]
        img8 = img_lists[7][i]
        img9 = img_lists[8][i]
        img10 = img_lists[9][i]
        img11 = img_lists[10][i]
        img12 = img_lists[11][i]
        img13 = img_lists[12][i]
        img14 = img_lists[13][i]
        img15 = img_lists[14][i]
        img16 = img_lists[15][i]
        img17 = img_lists[16][i]
        img18 = img_lists[17][i]
        img19 = img_lists[18][i]
        img20 = img_lists[19][i]
        img_f = torch.zeros(1, h, w).cuda()

        img_f[:, 0:224, 0: 224] += img1
        img_f[:, 0:224, 224: 448] += img2
        img_f[:, 0:224, 448: 672] += img3
        img_f[:, 0:224, 672: 896] += img4
        img_f[:, 0:224, 896: w] += img5[:, 0:224, 0:w-896]
        img_f[:, 224:448, 0: 224] += img6
        img_f[:, 224:448, 224: 448] += img7
        img_f[:, 224:448, 448: 672] += img8
        img_f[:, 224:448, 672: 896] += img9
        img_f[:, 224:448, 896: w] += img10[:, 0:224, 0:w-896]
        img_f[:, 448:672, 0: 224] += img11
        img_f[:, 448:672, 224: 448] += img12
        img_f[:, 448:672, 448: 672] += img13
        img_f[:, 448:672, 672: 896] += img14
        img_f[:, 448:672, 896: w] += img15[:, 0:224, 0:w - 896]
        img_f[:, 672:h, 0: 224] += img16[:, 0:h-672, 0:224]
        img_f[:, 672:h, 224: 448] += img17[:, 0:h-672, 0:224]
        img_f[:, 672:h, 448: 672] += img18[:, 0:h-672, 0:224]
        img_f[:, 672:h, 672: 896] += img19[:, 0:h-672, 0:224]
        img_f[:, 672:h, 896: w] += img20[:, 0:h-672, 0:w - 896]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images6(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: 224] += img1[:, 0:h, 0:224]
        img_f[:, 0:h, 224: w] += img2[:, 0:h, 0:w-224]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images7(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: 224] += img1[:, 0:h, 0:224]
        img_f[:, 0:h, 224: 448] += img2[:, 0:h, 0:224]
        img_f[:, 0:h, 448: w] += img3[:, 0:h, 0:w - 448]

        img_f_list.append(img_f)
    return img_f_list


def recons_fusion_images8(img_lists, h, w):
    img_f_list = []
    for i in range(len(img_lists[0])):

        img1 = img_lists[0][i]
        img2 = img_lists[1][i]
        img3 = img_lists[2][i]

        img_f = torch.zeros(1, h, w).cuda()
        print(img_f.size())

        img_f[:, 0:h, 0: 224] += img1[:, 0:h, 0:224]
        img_f[:, 0:h, 224: 448] += img2[:, 0:h, 0:224]
        img_f[:, 0:h, 448: w] += img3[:, 0:h, 0:w - 448]

        img_f_list.append(img_f)
    return img_f_list



def save_image_test(img_fusion, output_path):
    img_fusion = img_fusion.float()
    if args.cuda:
        img_fusion = img_fusion.cpu().data[0].numpy()
    else:
        img_fusion = img_fusion.clamp(0, 255).data[0].numpy()

    img_fusion = (img_fusion - np.min(img_fusion)) / (np.max(img_fusion) - np.min(img_fusion))
    img_fusion = img_fusion * 255
    print(img_fusion.shape)
    img_fusion = img_fusion.reshape([1,img_fusion.shape[0], img_fusion.shape[1]])#有些出来的是二维，需要变成3维
    img_fusion = img_fusion.transpose(1, 2, 0).astype('uint8')
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    imageio.imwrite(output_path, img_fusion)



def save_imgs(path, img_fusion):
    if img_fusion.shape[2] == 1:
        img_fusion = img_fusion.reshape([img_fusion.shape[0], img_fusion.shape[1]])
    imsave(path, img_fusion)


def save_image_scales(img_fusion, output_path):
    img_fusion = img_fusion.float()
    img_fusion = img_fusion.cpu().data[0].numpy()
    imageio.imwrite(output_path, img_fusion)

