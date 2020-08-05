import numpy as np
from scipy.signal import convolve2d
from math import log10


def get_bayer_masks(n_rows, n_cols):
    R = np.zeros((n_rows, n_cols), dtype=np.uint16)
    R[::2, 1::2] = 1
    G = np.zeros((n_rows, n_cols), dtype=np.uint16)
    G[::2, ::2] = 1
    G[1::2, 1::2] = 1
    B = np.zeros((n_rows, n_cols), dtype=np.uint16)
    B[1::2, ::2] = 1

    return np.dstack((R, G, B))


def get_masks(n_rows, n_cols):
    R = np.zeros((n_rows, n_cols), dtype=np.uint16)
    R[::2, 1::2] = 1
    G1 = np.zeros((n_rows, n_cols), dtype=np.uint16)
    G1[::2, ::2] = 1
    G2 = np.zeros((n_rows, n_cols), dtype=np.uint16)
    G2[1::2, 1::2] = 1
    B = np.zeros((n_rows, n_cols), dtype=np.uint16)
    B[1::2, ::2] = 1

    return R, G1, G2, B


def get_colored_img(raw_img):
    return get_bayer_masks(*raw_img.shape) * raw_img[..., np.newaxis]


def bilinear_interpolation(colored_img):
    kernel = np.ones((3, 3), dtype=np.uint16)
    R = convolve2d(colored_img[..., 0], kernel, mode='same')
    G = convolve2d(colored_img[..., 1], kernel, mode='same')
    B = convolve2d(colored_img[..., 2], kernel, mode='same')

    mask = get_bayer_masks(*colored_img.shape[:2])

    R_mask = convolve2d(mask[..., 0], kernel, mode='same')
    G_mask = convolve2d(mask[..., 1], kernel, mode='same')
    B_mask = convolve2d(mask[..., 2], kernel, mode='same')

    R = R * ~mask[..., 0].astype(bool) * (1 / R_mask) + colored_img[..., 0]
    G = G * ~mask[..., 1].astype(bool) * (1 / G_mask) + colored_img[..., 1]
    B = B * ~mask[..., 2].astype(bool) * (1 / B_mask) + colored_img[..., 2]

    return np.dstack((R, G, B)).astype(np.uint8)


def improved_interpolation(raw_img):
    # ----------------------------------------
    G_at_R_ker = np.array([[0, 0, -1, 0, 0],
                           [0, 0, 0, 0, 0],
                           [-1, 0, 4, 0, -1],
                           [0, 0, 0, 0, 0],
                           [0, 0, -1, 0, 0]])
    G_ker = np.array([[0, 2, 0],
                      [2, 0, 2],
                      [0, 2, 0]])
    scale = np.sum(G_at_R_ker) + np.sum(G_ker)
    G_at_R_ker, G_ker = G_at_R_ker / scale, G_ker / scale
    G_at_B_ker = G_at_R_ker
    # ----------------------------------------

    # ----------------------------------------
    R_at_G_top_ker = np.array([[0, 0, 0.5, 0, 0],
                               [0, -1, 0, -1, 0],
                               [-1, 0, 5, 0, -1],
                               [0, -1, 0, -1, 0],
                               [0, 0, 0.5, 0, 0]])
    R_top_ker = np.array([[0, 0, 0],
                          [4, 0, 4],
                          [0, 0, 0]])
    scale = np.sum(R_at_G_top_ker) + np.sum(R_top_ker)
    R_at_G_top_ker, R_top_ker = R_at_G_top_ker / scale, R_top_ker / scale
    R_at_G_low_ker, R_low_ker = R_at_G_top_ker.T, R_top_ker.T
    B_at_G_top_ker, B_at_G_low_ker, B_top_ker, B_low_ker = R_at_G_low_ker, R_at_G_top_ker, R_low_ker, R_top_ker
    # ----------------------------------------

    # ----------------------------------------
    R_at_B_ker = np.array([[0, 0, -1.5, 0, 0],
                           [0, 0, 0, 0, 0],
                           [-1.5, 0, 6, 0, -1.5],
                           [0, 0, 0, 0, 0],
                           [0, 0, -1.5, 0, 0]])
    RB_ker = np.array([[2, 0, 2],
                       [0, 0, 0],
                       [2, 0, 2]])
    scale = np.sum(R_at_B_ker) + np.sum(RB_ker)
    R_at_B_ker, RB_ker = R_at_B_ker / scale, RB_ker / scale
    B_at_R_ker, BR_ker = R_at_B_ker, RB_ker
    # ----------------------------------------

    r_mask, g_top_mask, g_low_mask, b_mask = get_masks(*raw_img.shape)
    color_img = get_colored_img(raw_img).astype(np.float64)
    R, G, B = color_img[..., 0], color_img[..., 1], color_img[..., 2]

    R += (convolve2d(G, R_at_G_top_ker, mode='same') + convolve2d(R, R_top_ker, mode='same')) * g_top_mask + \
         (convolve2d(G, R_at_G_low_ker, mode='same') + convolve2d(R, R_low_ker, mode='same')) * g_low_mask + \
         (convolve2d(B, R_at_B_ker, mode='same') + convolve2d(R, RB_ker, mode='same')) * b_mask

    G += (convolve2d(R, G_at_R_ker, mode='same') + convolve2d(G, G_ker, mode='same')) * r_mask + \
         (convolve2d(B, G_at_B_ker, mode='same') + convolve2d(G, G_ker, mode='same')) * b_mask

    B += (convolve2d(G, B_at_G_top_ker, mode='same') + convolve2d(B, B_top_ker, mode='same')) * g_top_mask + \
         (convolve2d(G, B_at_G_low_ker, mode='same') + convolve2d(B, B_low_ker, mode='same')) * g_low_mask + \
         (convolve2d(R, B_at_R_ker, mode='same') + convolve2d(B, RB_ker, mode='same')) * r_mask

    return np.dstack((R, G, B)).clip(0, 255).astype(np.uint8)


def MSE(img1, img2):
    return np.mean((img1 - img2) ** 2)


def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype(np.float64, copy=False)
    img_gt = img_gt.astype(np.float64, copy=False)
    mse = MSE(img_pred, img_gt)
    if mse == 0:
        raise ValueError

    return 10 * log10(img_gt.max() ** 2 / mse)
