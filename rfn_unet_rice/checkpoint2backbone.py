import numpy as np
import tensorflow as tf

from model.unet import UNet

batch_size = 1
crop_height, crop_width = 512, 512
n_classes = 4
gt_prob = .7
inp_channels = 3
pretrained_path = 'checkpoint/'
export_dir = 'backbone/'

unet = UNet()
checkpoint = tf.train.Checkpoint(model=unet)
checkpoint.restore(tf.train.latest_checkpoint(pretrained_path)).expect_partial()

inps = tf.keras.Input(shape=[None, None, inp_channels], name='inputs')

# # False-color image of RGB channels, dynamic range [0, 1]
# images = inps[..., 4:1:-1]

logits = unet(inps)
preds = tf.argmax(logits, axis=-1, name='logits2preds')

model = tf.keras.Model(inputs=inps, outputs=preds)

x = tf.ones([batch_size, crop_height, crop_width, inp_channels], dtype=tf.float32)
y = model(x)

tf.saved_model.save(model, export_dir=export_dir)