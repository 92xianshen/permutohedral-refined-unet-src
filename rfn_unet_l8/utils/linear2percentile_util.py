import numpy as np

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
