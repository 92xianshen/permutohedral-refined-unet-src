import os, time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.tfrecord_util import load_tfrecord

output_path = "../../result/rice/prediction/"

n_bands = 3
n_classes = 4
data_path = "../../data/rice/test/"
model_path = "backbone/"

crop_height = 512
crop_width = 512
img_channel_list = [0, 1, 2]


def main():
    # ->> Names of test images
    test_names = os.listdir(data_path)
    test_fullnames = [os.path.join(data_path, name) for name in test_names]
    test_set = load_tfrecord(
        test_fullnames,
        inp_shape=[crop_height, crop_width, n_bands],
        img_channel_list=img_channel_list,
    )
    save_info_name = "pred.csv"

    # ->> Backbone
    model = tf.saved_model.load(model_path)

    # ->> output folders
    if not os.path.exists(os.path.join(output_path, "pred")):
        os.makedirs(os.path.join(output_path, "pred", "png"))
        os.makedirs(os.path.join(output_path, "pred", "npz"))
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

            pred_npz_name = os.path.join(
                output_path, "pred", "npz", name.replace(".tfrecord", ".npz")
            )
            pred_png_name = os.path.join(
                output_path, "pred", "png", name.replace(".tfrecord", ".png")
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
            pred = model(x_norm)

            pred_time = time.time() - start

            pred = pred[0].numpy()
            ref = y[0].numpy()

            # Normalize image

            full_time = time.time() - start

            # ->> save as npz
            np.savez(pred_npz_name, pred)
            np.savez(ref_npz_name, ref)

            # ->> save as png
            plt.imsave(pred_png_name, pred)
            plt.imsave(ref_png_name, ref)

            fp.writelines(
                "{}, {}, {}, {}, {}, {}\n".format(
                    name,
                    None,
                    None,
                    None,
                    pred_time,
                    full_time,
                )
            )


if __name__ == "__main__":
    main()
