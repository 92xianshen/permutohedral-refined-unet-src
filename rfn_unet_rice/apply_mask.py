import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# theta_alpha = '60'
theta_alpha = '80'
# theta_alpha = '100'
# theta_alpha = '120'

# theta_beta = '.25'
# theta_beta = '.125'
# theta_beta = '.0625'
theta_beta = '.03125'

theta_gamma = '3'

# input_path = '../../result/rice/a={}, b={}, r={}/rfn/npz/'.format(theta_alpha, theta_beta, theta_gamma)
# output_path = '../../result/rice/a={}, b={}, r={}/rfn/masked/'.format(theta_alpha, theta_beta, theta_gamma)
# input_path = '../../result/rice/ablation, a={}, b={}, r={}/rfn/npz/'.format(theta_alpha, theta_beta, theta_gamma)
# output_path = '../../result/rice/ablation, a={}, b={}, r={}/rfn/masked/'.format(theta_alpha, theta_beta, theta_gamma)
# input_path = '../../result/rice/prediction/pred/npz/'
# output_path = '../../result/rice/prediction/pred/masked/'
input_path = '../../result/rice/prediction/ref/npz/'
output_path = '../../result/rice/prediction/ref/masked/'
image_path = '../../data/rice/image/'

# ->> Create output dir
if not os.path.exists(output_path):
    os.makedirs(output_path)

""" Apply the given mask to the image. """

def apply_mask(image, mask, mask_value, color, alpha=.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == mask_value, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])

    return image


for f in os.listdir(input_path):
    if os.path.splitext(f)[-1] == '.npz':
        print(f)
        
        mask_name = os.path.join(input_path, f)
        image_name = os.path.join(image_path, f.replace('.npz', '.png'))
        save_name = os.path.join(output_path, f.replace('.npz', '.png'))

        image = np.array(Image.open(image_name), dtype=np.float32)
        mask = np.load(mask_name)['arr_0']

        # print(mask.shape, mask.max(), mask.min())

        cloud, cloud_color = 3, [115, 223, 255]
        shadow, shadow_color = 2, [38, 115, 0]

        mask_image = apply_mask(image, mask, cloud, cloud_color)
        mask_image = apply_mask(mask_image, mask, shadow, shadow_color)

        mask_image = np.uint8(mask_image)

        plt.imsave(save_name, mask_image)