"""
    TFRecord loader, will output normalized x, image, and y
"""

import tensorflow as tf


def load_tfrecord(filenames, inp_shape, img_channel_list):
    """Load a tensorflow TFDataset file as a test set"""
    test_dataset = tf.data.TFRecordDataset(filenames)

    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string), 
        'height': tf.io.FixedLenFeature([], tf.int64), 
        'width': tf.io.FixedLenFeature([], tf.int64), 
        'n_channels': tf.io.FixedLenFeature([], tf.int64), 
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)

        x = tf.io.decode_raw(example['x'], tf.uint8)
        y = tf.io.decode_raw(example['y'], tf.int32)
        height, width, n_channels = example['height'], example['width'], example['n_channels']

        x = tf.reshape(x, [height, width, n_channels])
        y = tf.reshape(y, [height, width])

        x = tf.cast(x, dtype=tf.float32)
        x_norm = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x) + 1e-10)
        img = tf.gather(x, img_channel_list, axis=-1)
        y = tf.cast(y, tf.uint8)

        example["x_norm"] = x_norm
        example["image"] = img
        example["y"] = y

        return example

    test_dataset = test_dataset.map(_parse_function).batch(1, drop_remainder=True)

    return test_dataset
