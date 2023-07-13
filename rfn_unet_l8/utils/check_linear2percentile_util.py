import os

import numpy as np
import matplotlib.pyplot as plt

from reconstruct_util import reconstruct
from tfrecord_util import load_tfrecord

def linear_2_percent_stretch(image, truncated_percentile=2, minout=0., maxout=1.):
    def ch_stretch(ch):
        truncated_down = np.percentile(ch, truncated_percentile)
        truncated_up = np.percentile(ch, 100 - truncated_percentile)

        new_ch = ((maxout - minout) / (truncated_up - truncated_down)) * ch
        new_ch[new_ch < minout] = minout
        new_ch[new_ch > maxout] = maxout

        return new_ch

    n_chs = image.shape[-1]
    new_image = np.zeros_like(image)
    for i in range(n_chs):
        new_image[..., i] = ch_stretch(image[..., i])

    return new_image


def main():
    data_path = "../../../tfrecord_l8/"
    crop_height = 512
    crop_width = 512
    img_channel_list = [4, 3, 2]
    n_bands = 7

    test_names = os.listdir(data_path)
    for test_name in test_names:
        save_img_name = test_name.replace("train.tfrecords", "falsecolor.jpg")

        test_name = [os.path.join(data_path, test_name)]

        test_image = load_tfrecord(
                    test_name,
                    inp_shape=[crop_height, crop_width, n_bands],
                    img_channel_list=img_channel_list,
                )
        images = []

        for record in test_image.take(-1):
            img_patch = record['image']
            images += [img_patch[0].numpy()]

        images = np.stack(images, axis=0)
        image = reconstruct(images, crop_height=crop_height, crop_width=crop_width, n_channels=len(img_channel_list))

        image = linear_2_percent_stretch(image)

        plt.imsave(os.path.join(save_img_name), image)


if __name__ == "__main__":
    main()