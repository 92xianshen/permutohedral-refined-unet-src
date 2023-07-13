import tensorflow as tf


def load_tfrecord(filenames, inp_shape, img_channel_list):
    """Load a tensorflow TFDataset file as a test set"""
    test_dataset = tf.data.TFRecordDataset(filenames)

    feature_description = {
        "x_train": tf.io.FixedLenFeature([], tf.string),
        "y_train": tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)

        x = tf.io.decode_raw(example["x_train"], tf.int32)
        x = tf.reshape(x, inp_shape)

        x = tf.cast(x, dtype=tf.float32)
        x_norm = (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x) + 1e-10)
        img = tf.gather(x, img_channel_list, axis=-1)

        example["x_norm"] = x_norm
        example["image"] = img

        return example

    test_dataset = test_dataset.map(_parse_function).batch(1, drop_remainder=True)

    return test_dataset
