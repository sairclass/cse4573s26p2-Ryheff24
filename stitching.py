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

    torch.manual_seed(1234)
    imlist = list(imgs.values())
    left = (imlist[0]/255).unsqueeze(0)
    right = (imlist[1]/255).unsqueeze(0)
    # padding = abs(left.shape[2] - right.shape[2])
    # pad = (110,)*4
    # left_pad = torch.nn.functional.pad(left, (0,0,0,padding))
    # out_shape = (max(left.shape[-2], right.shape[-2]), left.shape[-1] + right.shape[-1])
    # print(out_shape)

    inputdict = {
      "image0": K.color.rgb_to_grayscale(left),
      "image1": K.color.rgb_to_grayscale(right)
    }

    loftr = K.feature.LoFTR(pretrained='outdoor')
    IS = K.contrib.ImageStitcher(loftr, estimator='ransac')
    points = loftr(inputdict)
    ransac = K.geometry.RANSAC() 
    homo = ransac(points["keypoints0"], points["keypoints1"])
    shp = left.shape
    # print(shp)
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
    
    x = tl_prim[0], tr_prim[0], bl_prim[0], br_prim[0], right.shape[3], 0
    y = tl_prim[1], tr_prim[1], bl_prim[1], br_prim[1], right.shape[2], 0
    minx, maxx = int(min(x)), int(max(x))
    miny, maxy = int(min(y)), int(max(y))
            
        
    width = maxx-minx
    height = maxy-miny
    outs = (height, width)
    
  
    # homo needs transform
    T = torch.tensor([[1, 0, -minx],[0, 1 ,-miny],[0, 0, 1]], dtype=torch.float32)
    # H = T @ homo[0].unsqueeze(0)
    H = (T @ homo[0]).unsqueeze(0)
    # print(homo[0].shape, H.shape, T.shape)
    # print("T:",T)
    # print("homo:", homo[0])
    # print("H:", H)
    
    # good stuff
    src_img = K.geometry.warp_perspective(left, H, outs)
    # show_image(src_img.squeeze())
    padding = (height-right.shape[2])//2
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html 
    # then use (padding_left,padding_right, padding_top,padding_bottom)
    #(at least minx,the width + minx, miny, the height + miny)
    print(minx,width+minx, miny, height+miny)
    
    minx = abs(minx)
    miny = abs(miny)
    print(f"minx: {minx}\nwidth+minx: {width+minx}\nminy: {miny}\nheight+miny: {height+miny}\nright.shape: {right.shape}\nsrc_img.shape: {src_img.shape}\nright.shape[2]-height+miny: {right.shape[2]-height+miny} ")
    right = torch.nn.functional.pad(right, (minx,right.shape[3]-width+minx, miny, height-miny-right.shape[2]))     

    # dst_img = torch.concatenate([right, torch.zeros_like(left_pad)], -1)
    write_image((src_img.squeeze(0) * 255).to(torch.uint8), "src.png")
    write_image((right.squeeze(0) * 255).to(torch.uint8), "dst.png")
    
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
