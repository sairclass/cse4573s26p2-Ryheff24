'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def ready(img):
    if len(img.shape) == 4:
        img = img.squeeze(0)
    if torch.max(img.to(torch.float32)) <= 3:
        img = img * 255
    return img.to(torch.uint8)

def i8(tens):
    return tens.to(torch.uint8)

def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256), dtype=torch.uint8) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    #Global params
    variant = True  # True: t1_1 is src, t1_2 is dst, False: t1_2 is src, t1_1 is dst
    # RANSAC params
    inl_th = 1.5
    max_iter = 100
    confidence = 0.999
    max_lo_iters = 15
    # torch.manual_seed(1234)
    # Mask params
    m = "bilinear" # bilinear bicubic%
    p = "zeros" # "zeros" "border" "reflection" "fill"
    align_corners = True

    imlist = list(imgs.values())
    t1_1 = (imlist[0]/255).unsqueeze(0)
    t1_2 = (imlist[1]/255).unsqueeze(0)

    # From Kornia ImageStitcher class source code:
    if variant:
        inputdict = {
        "image0": K.color.rgb_to_grayscale(t1_1),
        "image1": K.color.rgb_to_grayscale(t1_2)
        }
        shp = t1_1.shape
    else:
        inputdict = {
        "image0": K.color.rgb_to_grayscale(t1_2),
        "image1": K.color.rgb_to_grayscale(t1_1)
        }
        shp = t1_2.shape

    loftr = K.feature.LoFTR(pretrained='outdoor')
    points = loftr(inputdict)

    ransac = K.geometry.RANSAC(inl_th=inl_th, max_iter=max_iter, confidence=confidence, max_lo_iters=max_lo_iters)
    homo = ransac(points["keypoints0"], points["keypoints1"])
    homo = (homo[0],homo[1])

    def transform(points, H):
        # Equation 2.21 Szeliski Computer Vision: Algorithms and Applications
        out = []
        for point in points:
            x = point[0]
            y = point[1]
            h = H
            x_prime = (h[0][0]*x + h[0][1]*y + h[0][2])/(h[2][0]*x + h[2][1]*y + h[2][2])
            y_prime = (h[1][0]*x + h[1][1]*y + h[1][2])/(h[2][0]*x + h[2][1]*y + h[2][2])
            out.append((int(x_prime.round()), int(y_prime.round())))
        return tuple(out)
    # print(shp)

    tl,tr = (0, 0), (shp[3],0)
    bl, br = (0,shp[2]), (shp[3], shp[2])
    points = (tl, tr, bl, br)
    t = transform(points, homo[0])
    tl_prim, tr_prim, bl_prim, br_prim = t[0],t[1],t[2],t[3]
    if variant:
        x = tl_prim[0], tr_prim[0], bl_prim[0], br_prim[0], t1_2.shape[3], 0
        y = tl_prim[1], tr_prim[1], bl_prim[1], br_prim[1], t1_2.shape[2], 0
    else:
        x = tl_prim[0], tr_prim[0], bl_prim[0], br_prim[0], t1_1.shape[3], 0
        y = tl_prim[1], tr_prim[1], bl_prim[1], br_prim[1], t1_1.shape[2], 0


    minx, maxx = int(min(x)), int(max(x))
    miny, maxy = int(min(y)), int(max(y))

    width = maxx-minx
    height = maxy-miny
    outs = (height, width)

    # homo needs transform
    T = torch.tensor([[1, 0, -minx],[0, 1 ,-miny],[0, 0, 1]], dtype=torch.float32)
    H = (T @ homo[0]).unsqueeze(0)

    if variant:
        src_img = K.geometry.warp_perspective(t1_1, H, outs, padding_mode=p, align_corners=align_corners, mode=m) # change mode
        src_mask = K.geometry.warp_perspective(torch.ones_like(t1_1), H, outs, padding_mode=p, align_corners=align_corners, mode=m) # change mode
    else:
        src_img = K.geometry.warp_perspective(t1_2, H, outs, padding_mode=p, align_corners=align_corners, mode=m)
        src_mask = K.geometry.warp_perspective(torch.ones_like(t1_2), H, outs, padding_mode=p, align_corners=align_corners, mode=m)

    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    # then use (padding_left,padding_right, padding_top,padding_bottom)
    # (at least minx,the width + minx, miny, the height + miny)
    # print(minx,width+minx, miny, height+miny)

    minx = abs(minx)
    miny = abs(miny)

    # print(f"minx: {minx}\nwidth+minx: {width+minx}\nminy: {miny}\nheight+miny: {height+miny}\nright.shape: {t1_2.shape}\nsrc_img.shape: {src_img.shape}\nright.shape[2]-height+miny: {t1_2.shape[2]-height+miny} ")
    # print(f"minx: {minx}\nwidth+minx: {width+minx}\nminy: {miny}\nheight+miny: {height+miny}\nleft.shape: {t1_1.shape}\nsrc_img.shape: {src_img.shape}\nleft.shape[2]-height+miny: {t1_1.shape[2]-height+miny} ")
    if variant:
        dst_img = torch.nn.functional.pad(t1_2, (minx,t1_2.shape[3]-width+minx, miny, height-miny-t1_2.shape[2]))
        dst_mask = torch.nn.functional.pad(torch.ones_like(t1_2), (minx,t1_2.shape[3]-width+minx, miny, height-miny-t1_2.shape[2]))
    else:
        dst_img = torch.nn.functional.pad(t1_1, (minx, width-minx-t1_1.shape[3], miny, height-miny-t1_1.shape[2]))
        dst_mask = torch.nn.functional.pad(torch.ones_like(t1_1), (minx, width-minx-t1_1.shape[3], miny, height-miny-t1_1.shape[2]))

    dst_mask = dst_mask.bool()
    src_mask = src_mask.bool()
    # print(src_mask.shape, src_mask.dtype)
    # print(dst_mask.shape, dst_mask.dtype)

    both = dst_mask & src_mask

    both_dst_img = dst_img * both.float()
    both_src_img = src_img * both.float()

    dst_dif = both_dst_img - both_src_img
    src_dif = both_src_img - both_dst_img

    sigma = 7
    filter_size = 6 * sigma + 1
    erosion_kernel = 12
    dilation_kernel = 65

    dst_dif = K.filters.gaussian_blur2d(dst_dif, (filter_size, filter_size), (sigma, sigma))
    src_dif = K.filters.gaussian_blur2d(src_dif, (filter_size, filter_size), (sigma, sigma))

    dst_dif = dst_dif.sum(dim=1, keepdim=True)
    src_dif = src_dif.sum(dim=1, keepdim=True)

    dst_dif = torch.threshold(dst_dif, 0.33, 0)
    src_dif = torch.threshold(src_dif, 0.33, 0)

    dst_dif = torch.any(dst_dif > 0, dim=1, keepdim=True).to(torch.uint16)
    src_dif = torch.any(src_dif > 0, dim=1, keepdim=True).to(torch.uint16)

    dst_dif = K.morphology.erosion(dst_dif, torch.ones(erosion_kernel, erosion_kernel))
    src_dif = K.morphology.erosion(src_dif, torch.ones(erosion_kernel, erosion_kernel))

    dst_dif = K.morphology.dilation(dst_dif, torch.ones(dilation_kernel, dilation_kernel))
    src_dif = K.morphology.dilation(src_dif, torch.ones(dilation_kernel, dilation_kernel))

    # combined_mask = (src_mask & (src_dif.to(torch.uint8) == 1)) | ~dst_mask

    foreground = src_mask & (src_dif.to(torch.uint8) == 1)
    # foreground = K.morphology.erosion(foreground.to(torch.uint16), torch.ones(10, 10))

    blur_mask = foreground.to(torch.float32)

    to_blur = (~foreground * both).to(torch.float32)
    blend_sigma = 3
    # blend_filter_size = 6 * sigma + 1
    blend_filter_size = 7
    blur_mask = ~dst_mask + blur_mask
    mask = blur_mask.clone()
    blur_mask = K.morphology.dilation(blur_mask, torch.ones(7, 7))
    blur_mask = K.filters.gaussian_blur2d(blur_mask, (blend_filter_size, blend_filter_size), (blend_sigma, blend_sigma))

    blend_sigma = 2
    # blend_filter_size = 6 * sigma + 1
    blend_filter_size = 11
    target_region = mask + to_blur
    target_region = K.morphology.erosion(target_region, torch.ones(7,7))
    target_region = K.filters.gaussian_blur2d(target_region, (blend_filter_size, blend_filter_size), (blend_sigma, blend_sigma))
    blur_mask = target_region * blur_mask

    # feathering equation from slides
    # I_blend = alpha * I_left + (1-alpha) * I_right

    out = blur_mask * src_img + (1-blur_mask) * dst_img
    img = ready(out)

    return img


# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256), dtype=torch.uint8) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256), dtype=torch.uint8) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    img_details = torch.zeros((len(imgs.values()), 4), dtype=torch.float32)
    x = 0
    maxwidth = 0
    maxheight = 0
    input = imgs.copy()
    for key, val in input.items():
        i = input[key].float()
        if i.max() > 1:
            i = i / 255
        if i.ndim == 3:
            i = i.unsqueeze(0)
        input[key] = i
        width = i.shape[3]
        height = i.shape[2]
        if width > maxwidth:
            maxwidth = width
        if height > maxheight:
            maxheight = height
        # print(key, i.shape)
    # print(maxheight, maxwidth)
    for key, val in input.items():

        # pad to max
        width = val.shape[3]
        height = val.shape[2]

        rightpad = 0
        bottompad = 0
        # if width < maxwidth:
        #     rightpad = maxwidth - width
        if height < maxheight:
            # bottompad = maxheight - height
        # input[key] = torch.nn.functional.pad(val, (0, 0, 0, bottompad))
            bottompad = maxheight - height
        input[key] = torch.nn.functional.pad(val, (0, rightpad, 0, bottompad))

        # print(input[key].shape, rightpad, bottompad)
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        # then use (padding_left,padding_right, padding_top,padding_bottom)
        # (at least minx,the width + minx, miny, the height + miny)
        # print(minx,width+minx, miny, height+miny)

    matcher = K.feature.LoFTR(pretrained='outdoor')

    IS = K.contrib.ImageStitcher(matcher, estimator='ransac')
    with torch.no_grad():
        out = IS(*input.values())

    return img, overlap