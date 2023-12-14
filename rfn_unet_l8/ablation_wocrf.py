"""
Ablation study for perm. rfn. unet by disabling CRF.
Import parameters from ``backbone``.
Required by r1 of IEEE J-STARS. 
"""

import os, time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.reconstruct_util import reconstruct
from utils.tfrecord_util import load_tfrecord
# from utils.linear2percentile_util import linear_2_percent_stretch # disable linear 2% stretching

# from model.pydensecrf_wobilateral import PyDenseCRF # disable crf
from config.config import Config


def main():
    # ->> Instantiate config entity
    config = Config()

    # ->> Output all attributes of the current config entity
    print(config.__dict__)

    # ->> Create output dir
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
        print('Create output dir `{}`.'.format(config.save_path))

    # ->> Load all fnames
    test_names = os.listdir(config.data_path)
    print(test_names)

    # ->> Load backbone
    model = tf.saved_model.load(config.model_path)

    with open(os.path.join(config.save_path, config.save_info_fname), "w") as fp:
        fp.writelines("name, theta_alpha, theta_beta, theta_gamma, duration\n")

        for test_name in test_names:
            # Names
            save_npz_name = test_name.replace("train.tfrecords", "pred.npz")
            save_png_name = test_name.replace("train.tfrecords", "pred.png")
            save_img_name = test_name.replace("train.tfrecords", "falsecolor.jpg")

            # Load a test case
            test_name = [os.path.join(config.data_path, test_name)]
            test_image = load_tfrecord(
                test_name,
                inp_shape=[config.crop_height, config.crop_width, config.n_bands],
                img_channel_list=config.img_channel_list,
            )
            preds = []
            # images = []

            # Predict with the pretrained UNet
            start = time.time()
            print('UNet predicting...')
            i = 0
            for record in test_image.take(-1):
                x_norm, img_patch = record["x_norm"], record["image"]
                pred_patches = model(x_norm)

                preds += [pred_patches[0].numpy()]
                # images += [img_patch[0].numpy()]

                i += 1

            preds = np.stack(preds, axis=0)
            # images = np.stack(images, axis=0)
            prediction = reconstruct(preds, crop_height=config.crop_height, crop_width=config.crop_width, n_channels=None)
            # image = reconstruct(images, crop_height=config.crop_height, crop_width=config.crop_width, n_channels=len(config.img_channel_list))

            # Linearly stretch image with linear 2% algorithm
            # image = linear_2_percent_stretch(image, truncated_percentile=2, minout=0., maxout=1.)

            # # ->> Refinement
            # # Sizes and dims of spatial and color features
            # height, width = image.shape[:2]
            # d_spatial, d_image = 2, image.shape[-1]
            # n_feats = height * width
            # d_bifeats = d_spatial + d_image
            # d_spfeats = d_spatial

            # # Create DCRF
            # dcrf = PyDenseCRF(
            #     H=height,
            #     W=width,
            #     n_classes=config.n_classes,
            #     d_bifeats=d_bifeats,
            #     d_spfeats=d_spfeats,
            #     theta_alpha=config.theta_alpha,
            #     theta_beta=config.theta_beta,
            #     theta_gamma=config.theta_gamma,
            #     bilateral_compat=config.bilateral_compat,
            #     spatial_compat=config.spatial_compat,
            #     n_iterations=config.n_iterations,
            # )

            # # Compute
            # print("DCRF refining...")
            # unary_1d = unary.reshape((-1,))
            # image_1d = image.reshape((-1,))
            # out_1d = np.zeros_like(unary_1d, dtype=np.float32)

            # dcrf.inference(unary_1d, image_1d, out_1d)
            # refinement = out_1d.reshape((height, width, config.n_classes))
            # refinement = np.argmax(refinement, axis=-1)

            duration = time.time() - start

            # Save
            # vis_image = image if config.vis_channel_list == None else image[..., config.vis_channel_list]
            np.savez(os.path.join(config.save_path, save_npz_name), prediction)
            plt.imsave(os.path.join(config.save_path, save_png_name), prediction)
            # plt.imsave(os.path.join(config.save_path, save_img_name), vis_image)

            fp.writelines(
                "{}, {}, {}, {}, {}\n".format(
                    test_name, config.theta_alpha, config.theta_beta, config.theta_gamma, duration
                )
            )

            print("{} Done.".format(test_name))


if __name__ == "__main__":
    main()
