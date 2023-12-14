import numpy as np

def reconstruct(pred_patches, crop_height, crop_width, n_channels):
    """ Reconstruct a prediction from the prediction list
    """
    def reconstruct_from_patches(predictions, num_height, num_width, crop_height=crop_height, crop_width=crop_width):
        """ Reconstruct from a prediction list. A reverse function of `extract_pairwise_patches`
        """

        assert num_height * num_width == predictions.shape[0], 'Dim is wrong. {} X {} != {}'.format(
            num_height, num_width, predictions.shape[0])

        if n_channels == None:
            prediction = np.ndarray((num_height * crop_height, num_width * crop_width), dtype=predictions.dtype)
        else:
            prediction = np.ndarray((num_height * crop_height, num_width * crop_width, n_channels), dtype=predictions.dtype)

        for i in range(num_height):
            for j in range(num_width):
                prediction[(crop_height * i):(crop_height * (i + 1)), (crop_width * j):(crop_width * (j + 1))] = predictions[i * num_width + j]

        return prediction

    if pred_patches.shape[0] == 240:
        num_height = 15
        num_width = 16
    elif pred_patches.shape[0] == 256:
        num_height = 16
        num_width = 16
    else:
        print('Prediction shape error!')

    prediction = reconstruct_from_patches(
        pred_patches, num_height=num_height, num_width=num_width, crop_height=crop_height, crop_width=crop_width)

    # prediction.shape is [num_height x 512, num_width x 512]
    return prediction
