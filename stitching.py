'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''

from matplotlib.pyplot import show
import torch
import kornia as K
from typing import Dict
from utils import show_image, write_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    img = torch.zeros((3, 256, 256), dtype=torch.uint8) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    # 0 t1_1 is src, t1_2 is dst, 
    # 1 t1_2 is src, t1_1 is dst
    variant = 1
    
    torch.manual_seed(1234)
    imlist = list(imgs.values())
    t1_1 = (imlist[0]/255).unsqueeze(0)
    t1_2 = (imlist[1]/255).unsqueeze(0)
    def ready(tens):
        return (tens * 255).to(torch.uint8)
    def i8(tens):
        return tens.to(torch.uint8)
    # padding = abs(t1_1.shape[2] - t1_2.shape[2])
    # pad = (110,)*4
    # left_pad = torch.nn.functional.pad(t1_1, (0,0,0,padding))
    # out_shape = (max(t1_1.shape[-2], t1_2.shape[-2]), t1_1.shape[-1] + t1_2.shape[-1])
    # print(out_shape)
    if variant == 0:
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
    # IS = K.contrib.ImageStitcher()
    # IS.blend_image()
    points = loftr(inputdict)
    ransac = K.geometry.RANSAC(max_iter=20, confidence=0.999, max_lo_iters=20) 
    homo = ransac(points["keypoints0"], points["keypoints1"])
    
    
    # shp = t1_2.shape
    
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
    if variant == 0:
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
    
    # print(homo[0].shape, H.shape, T.shape)
    # print("T:",T)
    # print("homo:", homo[0])
    # print("H:", H)
    
    # good stuff
    m = "bilinear" # bilinear bicubic
    p = "zeros" # "zeros" "border" "reflection" "fill"
    align_corners = True
    
    if variant== 0:
        src_img = K.geometry.warp_perspective(t1_1, H, outs, padding_mode=p, align_corners=align_corners, mode=m) # change mode
        src_mask = K.geometry.warp_perspective(torch.ones_like(t1_1), H, outs, padding_mode=p, align_corners=align_corners, mode=m) # change mode
    else:
        src_img = K.geometry.warp_perspective(t1_2, H, outs, padding_mode=p, align_corners=align_corners, mode=m)
        src_mask = K.geometry.warp_perspective(torch.ones_like(t1_2), H, outs, padding_mode=p, align_corners=align_corners, mode=m)

    # show_image(src_mask.squeeze(0))
    # show_image(src_img.squeeze(0))
    
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html 
    # then use (padding_left,padding_right, padding_top,padding_bottom)
    # (at least minx,the width + minx, miny, the height + miny)
    # print(minx,width+minx, miny, height+miny)
    
    minx = abs(minx)
    miny = abs(miny)
    
    # print(f"minx: {minx}\nwidth+minx: {width+minx}\nminy: {miny}\nheight+miny: {height+miny}\nright.shape: {t1_2.shape}\nsrc_img.shape: {src_img.shape}\nright.shape[2]-height+miny: {t1_2.shape[2]-height+miny} ")
    # print(f"minx: {minx}\nwidth+minx: {width+minx}\nminy: {miny}\nheight+miny: {height+miny}\nleft.shape: {t1_1.shape}\nsrc_img.shape: {src_img.shape}\nleft.shape[2]-height+miny: {t1_1.shape[2]-height+miny} ")
    if variant == 0:
        dst_img = torch.nn.functional.pad(t1_2, (minx,t1_2.shape[3]-width+minx, miny, height-miny-t1_2.shape[2]))     
        dst_mask = torch.nn.functional.pad(torch.ones_like(t1_2), (minx,t1_2.shape[3]-width+minx, miny, height-miny-t1_2.shape[2]))     
    else:
        dst_img = torch.nn.functional.pad(t1_1, (minx, width-minx-t1_1.shape[3], miny, height-miny-t1_1.shape[2]))     
        dst_mask = torch.nn.functional.pad(torch.ones_like(t1_1), (minx, width-minx-t1_1.shape[3], miny, height-miny-t1_1.shape[2]))     
     
    
    
    
    dst_img = dst_img.squeeze(0)
    src_img = src_img.squeeze(0)
    dst_mask = dst_mask.squeeze(0).bool()
    src_mask = src_mask.squeeze(0).bool()
    
    # src_mask = torch.any(src_img > 0, dim=0, keepdim=True)
    # dst_mask = torch.any(dst_img > 0, dim=0, keepdim=True)
    
    print(src_mask.shape, src_mask.dtype)
    print(dst_mask.shape, dst_mask.dtype)
    # src_mask = src_mask.repeat(3,1,1)
    # dst_mask = dst_mask.repeat(3,1,1)
    # src_mask[1] = 0
    # src_mask[2] = 0
    

    
    # combined_mask = torch.where(src_mask, src_img, dst_img)
    # combined_mask = torch.where(dst_mask.bool(), dst_img, src_img)
        
            
    
    both = dst_mask & src_mask
    both_dst_img = dst_img * both.float()
    both_src_img = src_img * both.float()
    dif = abs(both_src_img - both_dst_img)
    
    sigma = 8
    filter_size = 6 * sigma + 1
    erosion_kernel = 10
    dilation_kernel = 75

    dst_dif = both_dst_img - both_src_img 
    src_dif = both_src_img - both_dst_img
    
    dst_dif = K.filters.gaussian_blur2d(dst_dif.unsqueeze(0), (filter_size, filter_size), (sigma, sigma)).squeeze(0)
    src_dif = K.filters.gaussian_blur2d(src_dif.unsqueeze(0), (filter_size, filter_size), (sigma, sigma)).squeeze(0)
    
    dst_dif = dst_dif.sum(axis=0, keepdim=True)
    src_dif = src_dif.sum(axis=0, keepdim=True)
    
    dst_dif = torch.threshold(dst_dif, 0.35, 0)
    src_dif = torch.threshold(src_dif, 0.35, 0)
    
    dst_dif = torch.any(dst_dif > 0, dim=0, keepdim=True).to(torch.uint16)
    src_dif = torch.any(src_dif > 0, dim=0, keepdim=True).to(torch.uint16)
    
    dst_dif = K.morphology.erosion(dst_dif.unsqueeze(0), torch.ones(erosion_kernel, erosion_kernel)).squeeze(0)
    src_dif = K.morphology.erosion(src_dif.unsqueeze(0), torch.ones(erosion_kernel, erosion_kernel)).squeeze(0)
    
    dst_dif = K.morphology.dilation(dst_dif.unsqueeze(0), torch.ones(dilation_kernel, dilation_kernel)).squeeze(0)
    src_dif = K.morphology.dilation(src_dif.unsqueeze(0), torch.ones(dilation_kernel, dilation_kernel)).squeeze(0)
    
    # dst= both_dst_img * (dst_dif + src_dif)
    # src = both_src_img * (dst_dif + src_dif)
    
    dst= both_dst_img * src_dif 
    src = both_src_img * dst_dif

    # t1_1 is src, t1_2 is dst
    # show dst_img if src_dif is 0 else show src_img
    #dst_mask if src_dif is 0 else src_mask
    conditional = (src_mask & (src_dif.to(torch.uint8) == 1)) | ~dst_mask
    combined_mask = torch.where(conditional, src_img, dst_img)
    
    conditional2 = (dst_mask & (dst_dif.to(torch.uint8) == 1)) | ~src_mask
    # combined_mask = torch.where(conditional, dst_img, src_img)
    
        
    
    write_image(ready(conditional),"mask.png")
    write_image(ready(conditional2),"mask_alt.png")
    write_image(ready(combined_mask), "out.png")
    write_image(ready(src_img), "src.png")
    write_image(ready(dst_img), "dst.png")
    
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
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    return img, overlap
