'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import matplotlib.pyplot as plt
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
    #Global params
    # 0 is true
    variant = True  # True: t1_1 is src, t1_2 is dst, False: t1_2 is src, t1_1 is dst
    save = False
    load = True
    save_path = "data/"
    # RANSAC params
    inl_th = 1.5
    max_iter = 100
    confidence = 0.999
    max_lo_iters = 15
    torch.manual_seed(1234)
    # Mask params
    m = "bilinear" # bilinear bicubic
    p = "zeros" # "zeros" "border" "reflection" "fill"
    align_corners = True
    if save:
            
        imlist = list(imgs.values())
        
        t1_1 = (imlist[0]/255).unsqueeze(0)
        t1_2 = (imlist[1]/255).unsqueeze(0)
        
        def ready(tens):
            if len(tens.shape) == 4:
                tens = tens.squeeze(0)
            return (tens * 255).to(torch.uint8)
        
        def i8(tens):
            return tens.to(torch.uint8)

        
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
        
        # save
        # dst_img, dst_mask, dst_dif
        # src_img, src_mask, src_dif
        torch.save(dst_img, f"{save_path}/dst_img.pt")
        torch.save(dst_mask, f"{save_path}/dst_mask.pt")
        torch.save(dst_dif, f"{save_path}/dst_dif.pt")
        torch.save(src_img, f"{save_path}/src_img.pt")
        torch.save(src_mask, f"{save_path}/src_mask.pt")
        torch.save(src_dif, f"{save_path}/src_dif.pt")

    if load and not save:
        dst_img = torch.load(f"{save_path}/dst_img.pt")
        dst_mask = torch.load(f"{save_path}/dst_mask.pt")
        dst_dif = torch.load(f"{save_path}/dst_dif.pt")
        src_img = torch.load(f"{save_path}/src_img.pt")
        src_mask = torch.load(f"{save_path}/src_mask.pt")
        src_dif = torch.load(f"{save_path}/src_dif.pt")
    
    
    alpha = True
    
    if alpha:
        sigma = 6
        filter_size = 6 * sigma + 1
        erosion_kernel = 12
        dilation_kernel = 50
        dst_dif = K.filters.gaussian_blur2d(dst_dif, (filter_size, filter_size), (sigma, sigma))
        src_dif = K.filters.gaussian_blur2d(src_dif, (filter_size, filter_size), (sigma, sigma))
        
        dst_dif = torch.threshold(dst_dif, 0.1, 0)
        src_dif = torch.threshold(src_dif, 0.1, 0)

        
        low_threshold = 0.1
        high_threshold = 0.2
        dst_dif = K.filters.canny(dst_dif, low_threshold, high_threshold, (filter_size, filter_size), (sigma, sigma))
        src_dif = K.filters.canny(src_dif, low_threshold, high_threshold, (filter_size, filter_size), (sigma, sigma))
        


        
    # return img
    else:
        sigma = 6
        filter_size = 6 * sigma + 1
        erosion_kernel = 12
        dilation_kernel = 50
        dst_dif = K.filters.gaussian_blur2d(dst_dif, (filter_size, filter_size), (sigma, sigma))
        src_dif = K.filters.gaussian_blur2d(src_dif, (filter_size, filter_size), (sigma, sigma))
        
        dst_dif = dst_dif.sum(axis=1, keepdim=True)
        src_dif = src_dif.sum(axis=1, keepdim=True)


        dst_dif = torch.threshold(dst_dif, 0.15, 0)
        src_dif = torch.threshold(src_dif, 0.15, 0)


        dst_dif = torch.any(dst_dif > 0, dim=1, keepdim=True).to(torch.uint16)
        src_dif = torch.any(src_dif > 0, dim=1, keepdim=True).to(torch.uint16)


        dst_dif = K.morphology.erosion(dst_dif, torch.ones(erosion_kernel, erosion_kernel))
        src_dif = K.morphology.erosion(src_dif, torch.ones(erosion_kernel, erosion_kernel))

        
        dst_dif = K.morphology.dilation(dst_dif, torch.ones(dilation_kernel, dilation_kernel))
        src_dif = K.morphology.dilation(src_dif, torch.ones(dilation_kernel, dilation_kernel))

        
        # t1_1 is src, t1_2 is dst
        # show dst_img if src_dif is 0 else show src_img
        #dst_mask if src_dif is 0 else src_mask
        
        # combined_mask = (src_mask & (src_dif.to(torch.uint8) == 1)) | ~dst_mask
        
        foreground = src_mask & (src_dif.to(torch.uint8) == 1)
        # foreground = K.morphology.erosion(foreground.to(torch.uint16), torch.ones(10, 10))
        
        foreground2 = dst_mask & (dst_dif.to(torch.uint8) == 1)
        write_image(ready(foreground),"foreground.png")
        # write_image(ready(foreground2),"foreground2.png")
        
        
        blend_sigma = 4
        blend_filter_size = 6 * sigma + 1
        blur_mask = foreground.to(torch.float32)
        
        blur_mask = K.filters.gaussian_blur2d(blur_mask, (blend_filter_size, blend_filter_size), (blend_sigma, blend_sigma))
        blur_mask = K.morphology.erosion(blur_mask, torch.ones(5, 5))
        # blur_mask = blur_mask * both
        blur_mask = (~dst_mask).to(torch.float32) + blur_mask
        
        # feathering equation from slides
        # I_blend = alpha * I_left + (1-alpha) * I_right
        out = blur_mask * src_img + (1-blur_mask) * dst_img
            
        write_image(ready(blur_mask),"mask.png")
        write_image(ready(out), "out.png")
        write_image(ready(both), "both.png")
        
        write_image(ready(dst_mask), "dst_mask.png")
        write_image(ready(src_mask), "src_mask.png")
        
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
