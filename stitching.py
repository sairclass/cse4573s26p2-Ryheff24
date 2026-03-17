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
    padding = abs(left.shape[2] - right.shape[2])
    pad = (110,)*4
    # left = torch.nn.functional.pad(left, pad)
    # right = torch.nn.functional.pad(right, pad)
    left_pad = torch.nn.functional.pad(left, (0,0,0,padding))
    out_shape = (max(left.shape[-2], right.shape[-2]), left.shape[-1] + right.shape[-1])
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
    print(shp)
    def transform(point, H):
        # Equation 2.21 Szeliski Computer Vision: Algorithms and Applications
        x = point[0]
        y = point[1]
        h = H
        x_prime = (h[0][0]*x + h[0][1]*y + h[0][2])/(h[2][0]*x + h[2][1]*y + h[2][2])
        y_prime = (h[1][0]*x + h[1][1]*y + h[1][2])/(h[2][0]*x + h[2][1]*y + h[2][2])
        return (x_prime, y_prime)
    
    tl = (0, 0)
    tr = (0, shp[3])
    bl = (shp[2], 0)
    br = (shp[2], shp[3])
    
    tl_prim = transform(tl, homo[0])
    tr_prim = transform(tr, homo[0])
    bl_prim = transform(bl, homo[0])
    br_prim = transform(br, homo[0])
    # inv = torch.linalg.inv(homo[0])
    # print(tl, tr, bl, br)
    
    # print(tl_prim, tr_prim, bl_prim, br_prim)
    # print(transform(tl_prim, inv), transform(tr_prim, inv), transform(bl_prim, inv), transform(br_prim, inv))
    minx = min(tl_prim[0], tr_prim[0], bl_prim[0], br_prim[0])
    maxx = max(tl_prim[0], tr_prim[0], bl_prim[0], br_prim[0])
    miny = min(tl_prim[1], tr_prim[1], bl_prim[1], br_prim[1])
    maxy = max(tl_prim[1], tr_prim[1], bl_prim[1], br_prim[1])
    # print((minx, maxx),(miny, maxy))
    # print(maxx-minx,maxy-miny)
    outs = (int((maxx-minx).ceil()),int((maxy-miny).ceil()))
    # print("max bounds", outs[0],outs[1])
    # print("left shapeshp", shp, shp[-2], shp[-1])

    horzpadding = (outs[0]-shp[-2])//2
    vertpadding = (outs[1]-shp[-1])//2
    # print(horzpadding, horzpadding//2, vertpadding, vertpadding//2)
    print(left.shape)
    left = torch.nn.functional.pad(left, (horzpadding, horzpadding, vertpadding, vertpadding)) 
    show_image(left.squeeze())
    
    print(left.shape)
    # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html 
    # then use (padding_left,padding_right, padding_top,padding_bottom)
    
    # good stuff
    src_img = K.geometry.warp_perspective(left, homo[0].unsqueeze(0), outs)
    show_image(src_img.squeeze())
    
    # dst_img = torch.concatenate([right, torch.zeros_like(left_pad)], -1)
    # # dst_img = torch.zeros(out_shape)
    # # print(src_img.dtype, dst_img.dtype)
    # write_image((src_img.squeeze(0) * 255).to(torch.uint8), "src.png")
    # write_image((dst_img.squeeze(0)  * 255).to(torch.uint8), "dst.png")
    
    # bad stuff
    
    # if mask_left is None:
        # mask_left = torch.ones_like(left)
    # if mask_right is None:
        # mask_right = torch.ones_like(right)
    # 'nearest' to ensure no floating points in the mask
    # src_mask = K.geometry.warp_perspective(mask_right, homo, out_shape, mode='nearest')
    # dst_mask = torch.concatenate([mask_left, torch.zeros_like(mask_right)], -1)
    # IS.blend_image(src_img, dst_img, src_mask), (dst_mask + src_mask).bool().to(src_mask.dtype)
    # src_mask = (src_img.squeeze(0).sum(dim=0) > 0).float()
    # dst_mask = (dst_img.squeeze(0).sum(dim=0) > 0).float()
    

    
    
    
    
    # print(dst_img & dst_mask)
    # print(src_img & src_mask)
    # show_image((src_mask and dst_mask).float())
    # show_image((src_mask and not dst_mask).float())
    # show_image((not src_mask and dst_mask).float())
    
    
    # print(src_mask.max(), dst_mask.max())
    # padding = abs(left.shape[1] - right.shape[1])
    # left = torch.nn.functional.pad(left, (0,0,0,padding))
    
    # with torch.no_grad():
    #     out = IS(left.unsqueeze(0), right.unsqueeze(0))

    # show_image(out.squeeze(0))
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
