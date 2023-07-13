import os, time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.tfrecord_util import load_tfrecord
# from utils.linear2percentile_util import linear_2_percent_stretch

from model.pydensecrf_wobilateral import PyDenseCRF

# theta_alpha = 60.
theta_alpha = 80.
# theta_alpha = 100.
# theta_alpha = 120.

# theta_beta = .25
# theta_beta = .125
# theta_beta = .0625
theta_beta = .03125

theta_gamma = 3.0
output_path = "../../result/rice/"

n_bands = 3
n_classes = 4
data_path = "../../data/rice/test/"
model_path = "unary_model/"

crop_height = 512
crop_width = 512
img_channel_list = [0, 1, 2]
bilateral_compat = 10.0
spatial_compat = 3.0
n_iterations = 10


def inference(model, dcrf, x_norm, img):
    unaries = model(x_norm)
    unary = unaries[0].numpy()
    image = img[0].numpy()

    image = linear_2_percent_stretch(
        image, truncated_percentile=2, minout=0.0, maxout=1.0
    )

    unary_1d = unary.reshape((-1,))
    image_1d = image.reshape((-1,))
    out_1d = np.zeros_like(unary_1d, dtype=np.float32)

    dcrf.inference(unary_1d, image_1d, out_1d)
    refinement = out_1d.reshape((crop_height, crop_width, n_classes))
    refinement = np.argmax(refinement, axis=-1)

    return refinement


def main():
    # ->> Names of test images
    test_names = os.listdir(data_path)
    test_fullnames = [os.path.join(data_path, name) for name in test_names]
    test_set = load_tfrecord(
        test_fullnames,
        inp_shape=[crop_height, crop_width, n_bands],
        img_channel_list=img_channel_list,
    )
    save_info_name = "rfn.csv"

    # ->> Backbone
    model = tf.saved_model.load(model_path)

    # ->> output folders
    if not os.path.exists(os.path.join(output_path, "rfn")):
        os.makedirs(os.path.join(output_path, "rfn", "png"))
        os.makedirs(os.path.join(output_path, "rfn", "npz"))
    if not os.path.exists(os.path.join(output_path, "ref")):
        os.makedirs(os.path.join(output_path, "ref", "png"))
        os.makedirs(os.path.join(output_path, "ref", "npz"))

    # ->> inference
    with open(os.path.join(output_path, save_info_name), "w") as fp:
        fp.writelines(
            "name, theta_alpha, theta_beta, theta_gamma, pred time, full time\n"
        )

        # for each image
        for record, name in zip(test_set, test_names):
            print("Processing {}...".format(name))

            rfn_npz_name = os.path.join(
                output_path, "rfn", "npz", name.replace(".tfrecord", ".npz")
            )
            rfn_png_name = os.path.join(
                output_path, "rfn", "png", name.replace(".tfrecord", ".png")
            )
            ref_npz_name = os.path.join(
                output_path, "ref", "npz", name.replace(".tfrecord", ".npz")
            )
            ref_png_name = os.path.join(
                output_path, "ref", "png", name.replace(".tfrecord", ".png")
            )

            start = time.time()

            x_norm, image, y = record["x_norm"], record["image"], record["y"]

            # ->> UNet predicting
            unary = model(x_norm)

            pred_time = time.time() - start

            unary = unary[0].numpy()
            image = image[0].numpy()
            ref = y[0].numpy()

            # Normalize image
            image = (image.astype(np.float32) - image.min()) / (image.max() - image.min() + 1e-10)

            # ->> Refinement
            # Sizes and dims of spatial and color features
            height, width = image.shape[:2]
            d_spatial, d_image = 2, image.shape[-1]
            n_feats = height * width
            d_bifeats = d_spatial + d_image
            d_spfeats = d_spatial

            # Create DCRF
            dcrf = PyDenseCRF(
                H=height,
                W=width,
                n_classes=n_classes,
                d_bifeats=d_bifeats,
                d_spfeats=d_spfeats,
                theta_alpha=theta_alpha,
                theta_beta=theta_beta,
                theta_gamma=theta_gamma,
                bilateral_compat=bilateral_compat,
                spatial_compat=spatial_compat,
                n_iterations=n_iterations,
            )

            # Compute
            unary_1d = unary.reshape((-1,))
            image_1d = image.reshape((-1,))
            out_1d = np.zeros_like(unary_1d, dtype=np.float32)

            dcrf.inference(unary_1d, image_1d, out_1d)
            refinement = out_1d.reshape((height, width, n_classes))
            refinement = np.argmax(refinement, axis=-1)

            full_time = time.time() - start

            # ->> save as npz
            np.savez(rfn_npz_name, refinement)
            np.savez(ref_npz_name, ref)

            # ->> save as png
            plt.imsave(rfn_png_name, refinement)
            plt.imsave(ref_png_name, ref)

            fp.writelines(
                "{}, {}, {}, {}, {}, {}\n".format(
                    name,
                    theta_alpha,
                    theta_beta,
                    theta_gamma,
                    pred_time,
                    full_time,
                )
            )


if __name__ == "__main__":
    main()
