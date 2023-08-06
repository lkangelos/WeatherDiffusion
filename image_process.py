import numpy as np
from PIL import Image
import os


def create_blk_img(src_path, save_path):
    img_black = Image.new('RGB', (1024, 1024), (0, 0, 0))
    for filename in os.listdir(src_path):
        img_black.save(os.path.join(save_path, filename))


def cover_image(cloud_img, dota_img, cloud_dota):
    img_cloud = Image.open(cloud_img)
    img_dota = Image.open(dota_img)
    box = (100, 100, 612, 612)
    img_dota = img_dota.crop(box)

    img_dota = np.array(img_dota)
    img_cloud = np.array(img_cloud)
    for i in range(img_cloud.shape[0]):
        for j in range(img_cloud.shape[1]):
            (b, g, r) = img_cloud[i, j]
            if (b, g, r) != (0, 0, 0):
                img_dota[i, j] = (b, g, r)

    img_dota = Image.fromarray(img_dota)
    img_dota.save(cloud_dota)


def background_multiplied_by_mask(src_path, mask_path, dest_path):
    for filename in os.listdir(src_path):
        src_file = os.path.join(src_path, filename)
        mask_file = os.path.join(mask_path, filename)
        dest_file = os.path.join(dest_path, filename)

        img_mask = Image.open(mask_file)
        img_src = Image.open(src_file)

        img_mask = np.array(img_mask)
        img_src = np.array(img_src)
        for i in range(img_mask.shape[0]):
            for j in range(img_mask.shape[1]):
                (b, g, r) = img_mask[i, j]
                if (b, g, r) != (0, 0, 0):
                    img_mask[i, j] = (1, 1, 1)
        img_src = img_src * img_mask
        Image.fromarray(img_src).save(dest_file)


def main():

    # cloud_img = '/home/louanqi/pycharmp/data/rice2/train/input/576.png'
    # mask_img = '/home/louanqi/pycharmp/data/rice2/train/mask/576.png'
    # only_cloud = '/home/louanqi/pycharmp/WeatherDiffusion/results/images/cloud.png'
    # cloud_mask(cloud_img, mask_img, only_cloud)

    # cloud_img = '/home/louanqi/pycharmp/WeatherDiffusion/results/images/cloud.png'
    # dota_img = '/home/louanqi/pycharmp/data/dota/val_split/images/P1542__1__2472___1648.png'
    # cover_image(cloud_img, dota_img)

    # src_path = '/home/louanqi/pycharmp/data/rice2/val/input'
    # save_path = '/home/louanqi/pycharmp/data/rice2/val/black'
    # create_blk_img(src_path, save_path)

    src_path = '/home/louanqi/pycharmp/data/rice2/train/gt'
    mask_path = '/home/louanqi/pycharmp/data/rice2/train/mask'
    dest_path = '/home/louanqi/pycharmp/data/rice2/train/cloud_gt'
    background_multiplied_by_mask(src_path, mask_path, dest_path)


if __name__ == '__main__':
    main()
