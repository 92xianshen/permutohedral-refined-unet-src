"""
Convert checkpoint to tf.SavedModel, for ablation study for perm. rfn. unet by disabling CRF.
Required by r1 of IEEE J-STARS. 
"""

import numpy as np
import tensorflow as tf

from model.unet import UNet

n_classes = 4
gt_prob = .7
inp_channels = 7
pretrained_path = 'pretrained/'
# Attention! Output path
export_dir = 'backbone/'

unet = UNet()
checkpoint = tf.train.Checkpoint(model=unet)
checkpoint.restore(tf.train.latest_checkpoint(pretrained_path)).expect_partial()

inps = tf.keras.Input(shape=[None, None, inp_channels], name='inputs')

logits = unet(inps)
preds = tf.argmax(logits, axis=-1, name='logits2preds')

model = tf.keras.Model(inputs=inps, outputs=preds)

x = tf.ones([1, 512, 512, 7], dtype=tf.float32)
y = model(x)

tf.saved_model.save(model, export_dir=export_dir)