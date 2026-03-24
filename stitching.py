'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import collections

import torch
import kornia as K
from typing import Dict

from multi_img_output import better_show
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

    # print(H, minx, miny, maxx, maxy, width, height)
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
    overlap = torch.empty((len(imgs.keys()), len(imgs.keys())), dtype=torch.uint8) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    data = {}
    max_h = 0
    max_w = 0
    descriptor = "disk" # dedode
    matchalgo = "lightglue" # lightglue
    loftr = None
    dedode = None
    disk = None
    lg = None
    for k, v in imgs.items():

        v = v / 255
        if v.ndim == 3:
            v = v.unsqueeze(0)
        if v.shape[3] > 900 or v.shape[2] > 900:
            v = torch.nn.functional.interpolate(v, scale_factor=0.5, mode="bilinear", align_corners=False)
        imgs[k] = v
        max_h = max(max_h, v.shape[2])
        max_w = max(max_w, v.shape[3])

    keys = list(imgs.keys())

    # for k, v in imgs.items():
        # imgs[k] = torch.nn.functional.pad(v, (0,  max_w - imgs[k].shape[3], 0, max_h - imgs[k].shape[2]))
        # print(imgs[k].shape)

    if descriptor == "disk":
        disk = K.feature.DISK.from_pretrained("depth").to(device)
        for k, v in imgs.items():
            features = disk(v.to(device), pad_if_not_divisible=True)[0]
            data[k] = {
                "points": (features.keypoints.cpu(), features.descriptors.cpu()),
                "matches": 0
            }
        del disk
    else:
        dedode = K.feature.DeDoDe.from_pretrained(detector_weights="L-C4-v2", descriptor_weights="B-upright").to(device)

        for k, v in imgs.items():

            keypoints, _, descriptors = dedode(v.to(device), apply_imagenet_normalization=True, n=1000) #n=2000
            data[k] = {
                "points": (keypoints.cpu(), descriptors.cpu()),
                "matches": 0
            }
        del dedode



    if device == torch.device("mps"):
        torch.mps.empty_cache()

    print("descriptors done")

    matcher = K.feature.match_smnn

    if matchalgo == "loftr":
        loftr = K.feature.LoFTR(pretrained='indoor_new')
    else:
        if descriptor == "disk":
            lg = K.feature.LightGlue("disk").eval().to(device)
        else :
            lg = K.feature.LightGlue("dedodeb").eval().to(device) # checck if B is better,

    # RANSAC params
    inl_th = 1
    max_iter = 200
    confidence = 0.9999
    max_lo_iters = 20
    ransac = K.geometry.RANSAC("homography", inl_th=inl_th, max_iter=max_iter, confidence=confidence, max_lo_iters=max_lo_iters)#.to(device)

    with torch.no_grad():
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):

                k1 = keys[i]
                k2 = keys[j]
                k1keypoints, k1descriptors = data[k1]["points"]
                k2keypoints, k2descriptors = data[k2]["points"]
                if descriptor == "dedode":
                    k1descriptors = k1descriptors.squeeze(0)
                    k2descriptors = k2descriptors.squeeze(0)

                # k1d = data[k1]["dedode"][2].squeeze(0)
                # k2d = data[k2]["dedode"][2].squeeze(0)

                match = matcher(k1descriptors, k2descriptors)

                match_count = match[0].shape[0]
                data[k1]["matches"] += match_count
                data[k2]["matches"] += match_count

                if match_count > 30:
                    print(i, j, overlap.shape)
                    overlap[i, j] = 1
                    overlap[j, i] = 1
                    if matchalgo == "loftr":
                        print(f"\r{k1}, {k2} LoFTR starting. Prelim matches: {match_count}, img shape: {imgs[k1].shape}", end="", flush=True)
                        print()

                        inputdict = {
                            "image0": K.color.rgb_to_grayscale(imgs[k1]),
                            "image1": K.color.rgb_to_grayscale(imgs[k2])
                        }
                        points = loftr(inputdict)
                        print(f"\r{k1}, {k2} RANSAC starting. LoFTR Matches: {points['keypoints0'].shape}{'':>20}",  end="", flush=True)

                        homo = ransac(points["keypoints0"], points["keypoints1"])

                        # homo = ransac(points["keypoints0"].to(device), points["keypoints1"].to(device))
                        print(f"\r{k1}, {k2} RANSAC Finished. LoFTR matches: {points['keypoints0'].shape}, RANSAC inliers: {homo[1].sum()}",  end="", flush=True)
                        print()

                        data[k1][k2] = torch.linalg.inv(homo[0].cpu())
                        data[k2][k1] = homo[0].cpu()
                    else:
                        print(f"\r{k1}, {k2} LightGlue starting. Prelim matches: {match_count}, img shape: {imgs[k1].shape}", end="", flush=True)

                        if descriptor == "dedode":
                            inputdict = {
                                "image0": {"keypoints": k1keypoints.to(device), "descriptors": k1descriptors.unsqueeze(0).to(device), "image_size": torch.tensor([[imgs[k1].shape[3], imgs[k1].shape[2]]], device=device) },
                                "image1": {"keypoints": k2keypoints.to(device), "descriptors": k2descriptors.unsqueeze(0).to(device), "image_size": torch.tensor([[imgs[k2].shape[3], imgs[k2].shape[2]]], device=device) },
                            }
                        else: # disk
                            inputdict = {
                                "image0": {"keypoints": k1keypoints.unsqueeze(0).to(device), "descriptors": k1descriptors.unsqueeze(0).to(device), "image_size": torch.tensor([[imgs[k1].shape[3], imgs[k1].shape[2]]], device=device)},
                                "image1": {"keypoints": k2keypoints.unsqueeze(0).to(device), "descriptors": k2descriptors.unsqueeze(0).to(device), "image_size": torch.tensor([[imgs[k2].shape[3], imgs[k2].shape[2]]], device=device)}
                            }

                        out = lg(inputdict)
                        match = out["matches"][0].cpu()

                        k1matches = k1keypoints[match[:, 0]]
                        k2matches = k2keypoints[match[:, 1]]

                        print(f"\r{k1}, {k2} LightGlue Finished. LightGlue matches: {k2matches.shape[0]}, img shape: {imgs[k1].shape}{'':>20}",  end="", flush=True)

                        homo = ransac(k1matches, k2matches)

                        # homo = ransac(k1matches.to(device), k2matches.to(device))
                        print(f"\r{k1}, {k2} RANSAC Finished. LightGlue matches: {k2matches.shape[0]}, RANSAC inliers: {homo[1].sum()}, img shape: {imgs[k1].shape}",  end="", flush=True)
                        print()
                        data[k1][k2] = torch.linalg.inv(homo[0].cpu())
                        data[k2][k1] = homo[0].cpu()
    # torch.save(data, 'data/data.pth')
    print("ransac done")

    # if load:
    #     data = torch.load('data/data.pth')

    ref = max(data, key=lambda l: data[l]["matches"])

    #bfs
    Q = collections.deque()
    visited = set()
    qlog = [ref]
    visited.add(ref)
    Q.append(ref)
    Hs = {ref: torch.eye(3)}
    while Q:
        cur = Q.popleft()
        print(cur, data[cur].keys())
        for k, v in data[cur].items(): # neighbors but skip the 2 keys and visited checks
            if k == "matches" or k == "points" or k in visited:
                continue

            Q.append(k)
            visited.add(k)
            qlog.append(k)
            # compute transformations of neighbor to cur
            print(cur, k)
            homo = Hs[cur] @ data[cur][k]
            Hs[k] = homo
    print(qlog)
    print("BFS done")
    x = [0]
    y = [0]
    for k, v in imgs.items():
        shp = v.shape
        H = Hs[k]
        tl, tr = (0, 0), (shp[3], 0)
        bl, br = (0, shp[2]), (shp[3], shp[2])
        points = (tl, tr, bl, br)
        t = transform(points, H)
        tl_prim, tr_prim, bl_prim, br_prim = t[0], t[1], t[2], t[3] #_prim 0 is x, 1 is y
        x.extend([tl_prim[0], tr_prim[0], bl_prim[0], br_prim[0]])
        y.extend([tl_prim[1], tr_prim[1], bl_prim[1], br_prim[1]])
    print("warping...")
    x_max, y_max = max(x), max(y)
    x_min, y_min = min(x), min(y)
    width = x_max-x_min
    height = y_max-y_min
    outs = (height, width)
    final = torch.zeros(outs)
    # homo needs transform
    T = torch.tensor([[1, 0, -x_min],[0, 1 ,-y_min],[0, 0, 1]], dtype=torch.float32)
    out = []
    finalm = None
    for k in keys:
        H = Hs[k]
        H = (T @ H).unsqueeze(0)
        src_img = K.geometry.warp_perspective(imgs[k], H, outs)
        src_mask = K.geometry.warp_perspective(torch.ones_like(imgs[k]), H, outs)
        # data[k] = {
        #     "warped": src_img,
        #     "mask": src_mask
        # }
        out.append(src_img)

        # src_blur_mask
        final = torch.where(src_mask > 0.5, src_img, final)
        if finalm is None:
            finalm = src_mask
        else:
            finalm = finalm + src_mask

        # better_show(src_mask)
    # better_show(finalm)
    img = ready(final)

    # H = (T @ homo[0]).unsqueeze(0)
    # src_img = K.geometry.warp_perspective(t1_1, H, outs)  # change mode
    # src_mask = K.geometry.warp_perspective(torch.ones_like(t1_1))


    return img, overlap