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


    imlist = list(imgs.values())
    left = (imlist[0]/255).unsqueeze(0)
    right = (imlist[1]/255).unsqueeze(0)
    out_shape = (max(left.shape[-2], right.shape[-2]), left.shape[-1] + right.shape[-1])
    print(out_shape)
    
    padding = abs(left.shape[2] - right.shape[2])
    left_pad = torch.nn.functional.pad(left, (0,0,0,padding))
    
    inputdict = {
      "image0": K.color.rgb_to_grayscale(left),
      "image1": K.color.rgb_to_grayscale(right)
    }
    # show_image(left)
    # show_image(right)
    
    loftr = K.feature.LoFTR(pretrained='outdoor')
    IS = K.contrib.ImageStitcher(loftr, estimator='ransac')
    points = loftr(inputdict)
    ransac = K.geometry.RANSAC() 
    homo = ransac(points["keypoints0"], points["keypoints1"])
    
    src_img = K.geometry.warp_perspective(left, homo[0].unsqueeze(0), out_shape)
    dst_img = torch.concatenate([right, torch.zeros_like(left_pad)], -1)
    # dst_img = torch.zeros(out_shape)
    print(src_img.dtype, dst_img.dtype)
    write_image((src_img.squeeze(0) * 255).to(torch.uint8), "src.png")
    write_image((dst_img.squeeze(0)  * 255).to(torch.uint8), "dst.png")
    
    
    
    
    
    
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
