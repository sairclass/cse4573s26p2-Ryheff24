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
    confidence = 0.9999
    max_lo_iters = 20
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

    dst_mask = (dst_mask>.5).bool()
    src_mask = (src_mask>.5).bool()
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
    blur_mask = K.morphology.dilation(blur_mask, torch.ones(9, 9))
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
    matchalgo = "lightglue" # loftr
    # RANSAC params
    inl_th = 1
    max_iter = 200
    confidence = 0.9999
    max_lo_iters = 20

    loftr = None
    lg = None
    for k, v in imgs.items():
        v = v / 255
        if v.ndim == 3:
            v = v.unsqueeze(0)
        # if v.shape[3] > 900 or v.shape[2] > 900:
        #     print("downscaling...")
        #     v = torch.nn.functional.interpolate(v, scale_factor=0.5, mode="bilinear", align_corners=False)
        imgs[k] = v
        max_h = max(max_h, v.shape[2])
        max_w = max(max_w, v.shape[3])

    keys = list(imgs.keys())
    edges = []

    disk = K.feature.DISK.from_pretrained("depth").to(device)
    for k, v in imgs.items():
        features = disk(v.to(device), pad_if_not_divisible=True, n=2000)[0]
        data[k] = {
            "points": (features.keypoints.cpu(), features.descriptors.cpu()),
            "matches": 0
        }
    del disk

    if device == torch.device("mps"):
        torch.mps.empty_cache()

    matcher = K.feature.match_smnn

    if matchalgo == "loftr":
        loftr = K.feature.LoFTR(pretrained='indoor_new')
    else:
        lg = K.feature.LightGlue("disk").eval().to(device)

    ransac = K.geometry.RANSAC("homography", inl_th=inl_th, max_iter=max_iter, confidence=confidence, max_lo_iters=max_lo_iters)#.to(device)
    print("descriptors done")

    with torch.no_grad():
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):

                k1 = keys[i]
                k2 = keys[j]
                k1keypoints, k1descriptors = data[k1]["points"]
                k2keypoints, k2descriptors = data[k2]["points"]

                # k1d = data[k1]["dedode"][2].squeeze(0)
                # k2d = data[k2]["dedode"][2].squeeze(0)

                match = matcher(k1descriptors, k2descriptors)

                match_count = match[0].shape[0]
                data[k1]["matches"] += match_count
                data[k2]["matches"] += match_count
                if match_count > 30:
                    overlap[i, j] = 1
                    overlap[j, i] = 1
                    if matchalgo == "loftr":
                        # print(f"\r{k1}, {k2} LoFTR starting. Prelim matches: {match_count}, img shape: {imgs[k1].shape}", end="", flush=True)

                        inputdict = {
                            "image0": K.color.rgb_to_grayscale(imgs[k1]),
                            "image1": K.color.rgb_to_grayscale(imgs[k2])
                        }
                        points = loftr(inputdict)
                        # print(f"\r{k1}, {k2} RANSAC starting. LoFTR Matches: {points['keypoints0'].shape}{'':>20}",  end="", flush=True)

                        homo = ransac(points["keypoints0"], points["keypoints1"])

                        # homo = ransac(points["keypoints0"].to(device), points["keypoints1"].to(device))
                        # print(f"\r{k1}, {k2} RANSAC Finished. LoFTR matches: {points['keypoints0'].shape}, RANSAC inliers: {homo[1].sum()}",  end="", flush=True)
                        # print()

                        data[k1][k2] = torch.linalg.inv(homo[0].cpu())
                        data[k2][k1] = homo[0].cpu()
                    else:
                        # print(f"\r{k1}, {k2} LightGlue starting. Prelim matches: {match_count}, img shape: {imgs[k1].shape}", end="", flush=True)


                        inputdict = {
                            "image0": {"keypoints": k1keypoints.unsqueeze(0).to(device), "descriptors": k1descriptors.unsqueeze(0).to(device), "image_size": torch.tensor([[imgs[k1].shape[3], imgs[k1].shape[2]]], device=device)},
                            "image1": {"keypoints": k2keypoints.unsqueeze(0).to(device), "descriptors": k2descriptors.unsqueeze(0).to(device), "image_size": torch.tensor([[imgs[k2].shape[3], imgs[k2].shape[2]]], device=device)}
                        }

                        out = lg(inputdict)
                        match = out["matches"][0].cpu()

                        k1matches = k1keypoints[match[:, 0]]
                        k2matches = k2keypoints[match[:, 1]]

                        # print(f"\r{k1}, {k2} LightGlue Finished. LightGlue matches: {k2matches.shape[0]}, img shape: {imgs[k1].shape}{'':>20}",  end="", flush=True)

                        homo = ransac(k1matches, k2matches)

                        # homo = ransac(k1matches.to(device), k2matches.to(device))

                        if homo[1].sum().cpu() > 50:
                            print(f"\r{k1}, {k2} RANSAC Finished. LightGlue matches: {k2matches.shape[0]}, RANSAC inliers: {homo[1].sum()}, Prelim matches: {match_count}, img shape: {imgs[k1].shape}", end="", flush=True)
                            print()
                            data[k1][k2] = torch.linalg.inv(homo[0].cpu())
                            data[k2][k1] = homo[0].cpu()
                            edges.append((k1, k2, homo[1].sum().cpu()))
                        else:
                            print(f"{k1}, {k2} rejected. LightGlue matches: {k2matches.shape[0]}, RANSAC inliers: {homo[1].sum()}, Prelim matches: {match_count}, img shape: {imgs[k1].shape}")
                        if device == torch.device("mps"):
                            torch.mps.empty_cache()

    del matcher
    if matchalgo == "loftr":
        del loftr
    else:
        del lg
    if device == torch.device("mps"):
        torch.mps.empty_cache()
    print("Ransac Done")


    mst = False
    ref = max(data, key=lambda l: data[l]["matches"])

    Hsm = {ref: torch.eye(3)}

    edges = sorted(edges, key=lambda f: f[2])
    #
    # v = set()
    # v.add(ref)
    # while len(keys) > len(v):
    #     for e in edges:
    #         k1, k2, _ = e
    #         if k1 in v and k2 not in v:
    #             # add it
    #             Hsm[k2] = Hsm[k1] @ data[k1][k2]
    #             v.add(k2)
    #             edges.remove(e)
    #             break
    #         if k2 in v and k1 not in v:
    #             Hsm[k1] = Hsm[k2] @ data[k2][k1]
    #             v.add(k1)
    #             edges.remove(e)
    #             break

    Hsb = {ref: torch.eye(3)}
    Q = []
    visited = set()
    visited.add(ref)
    Q.append(ref)
    while Q:
        cur = Q.pop(0)
        for k, v in data[cur].items(): # neighbors but skip the 2 keys and visited checks
            if k == "matches" or k == "points" or k in visited:
                continue

            Q.append(k)
            visited.add(k)
            # compute transformations of neighbor to cur

            homo = Hsb[cur] @ data[cur][k]
            Hsb[k] = homo
    # if mst:
    #     Hs = Hsm
    # else:
    Hs = Hsb

    # for i in keys:
    #     if not torch.equal(Hsm[i], Hsb[i]):
    #         print(f"{i} FAILED: {Hsm[i]} != {Hsb[i]} ")


    x = [0]
    y = [0]
    print("bfs done")

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
    x_max, y_max = max(x), max(y)
    x_min, y_min = min(x), min(y)
    width = x_max-x_min
    height = y_max-y_min
    outs = (height, width)
    final = torch.zeros(outs)
    # homo needs transform
    T = torch.tensor([[1, 0, -x_min],[0, 1 ,-y_min],[0, 0, 1]], dtype=torch.float32)
    out_imgs = []
    out_masks = []
    finalm = torch.zeros(outs)
    acc = torch.zeros(outs)
    dervs = []
    for k in keys:
        H = Hs[k]
        H = (T @ H).unsqueeze(0)
        src_img = K.geometry.warp_perspective(imgs[k], H, outs)
        src_mask = K.geometry.warp_perspective(torch.ones_like(imgs[k]), H, outs)
        delimg = K.filters.spatial_gradient(src_img)
        dervs.append(delimg)
        src_mask = (src_mask > 0.99)
        out_imgs.append(src_img)
        out_masks.append(src_mask)
        if acc is None:
            acc = src_mask
            overlaps = src_mask
        else:
            overlaps = (acc > .5) & src_mask
            acc = torch.clamp(acc + src_mask, 0, 1)

        blend_sigma = 4
        blend_filter_size = 6 * blend_sigma + 1

        # src_mask = K.morphology.erosion(src_mask, torch.ones(blend_filter_size, blend_filter_size)).to(torch.float32)
        target_region = -torch.nn.functional.max_pool2d(-src_mask.to(torch.float32).to(device), kernel_size=blend_filter_size, stride=1, padding=blend_filter_size//2).cpu()
        target_region = K.filters.gaussian_blur2d(target_region.to(torch.float32), (blend_filter_size, blend_filter_size),(blend_sigma, blend_sigma))

        # final = torch.where(src_mask > 0.5, src_img, final)
        # src_mask = (src_mask > 0.5)
        #

        # I_blend = alpha * I_left + (1-alpha) * I_right
        mask = torch.where(overlaps, target_region, src_mask)

        final = mask * src_img + (1 - mask) * final
        # print(f"total time: {blendtime - start}, warptime: {warptime- start}, erodetime: {erodetime- warptime}, blurtime: {blurtime-erodetime} blendtime: {blendtime-blurtime}")


    nowimg = out_imgs[0]
    nowmask = out_masks[0]
    def findbb(curmask, a, b):
        """"
        Returns: curmask, a, b cropped
        """
        zeros = torch.nonzero(curmask[0,0])
        minh, minw = zeros.min(dim=0).values
        maxh, maxw = zeros.max(dim=0).values
        return curmask[:, :, minh:maxh+1, minw:maxw+1], a[:, :, minh:maxh+1, minw:maxw+1], b[:, :, minh:maxh+1, minw:maxw+1],(minw, minh, maxw, maxh)



    def minerrcut(m, b_1, b_2):
        # https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf
        # 2.1 Minimum Error Boundary Cut
        maxerr = 1e10

        m = m.squeeze()[0]
        e = (b_1.squeeze(0) - b_2.squeeze(0))**2
        e = e.sum(dim=0)
        e = e + (~m).to(torch.float32) * maxerr
        E = e.clone()
        N1 = E.shape[0]
        # N2 = E.shape[0]
        path = torch.zeros_like(E)
        for i in range(1, N1):
            currow = E[i-1]
            left = torch.cat([torch.tensor([maxerr]), currow[:-1]])
            right = torch.cat([currow[1:], torch.tensor([maxerr])])
            mid = currow

            stack = torch.stack(([left, mid, right]))
            bestidx = torch.argmin(stack, dim=0)
            best = torch.gather(stack, dim=0, index=bestidx.unsqueeze(0))
            E[i] = e[i] + best
            path[i] = bestidx - 1


        curidx = E[-1].argmin().item()
        ret = torch.zeros_like(m)
        for w in range(N1-1, -1, -1):
            ret[w, :curidx] = 1
            if w > 0:
                curidx = curidx + int(path[w, curidx])


        return ret

    for i in range (1, len(out_imgs)):
        if i == 0: continue
        cur_img = out_imgs[i]
        cur_mask = out_masks[i].bool()
        overlaps = nowmask & cur_mask
        union = nowmask | cur_mask
        notov = union & ~overlaps

        # cur_ov = cur_img & overlaps
        # now_ov = nowimg & overlaps
        a, cur_ov_crop, now_ov_crop, loc = findbb(overlaps, cur_img, nowimg)
        minw, minh, maxw, maxh = loc
        cut = minerrcut(a,cur_ov_crop,now_ov_crop)
        cut_mask = torch.nn.functional.pad(cut, (minw, cur_img.shape[3]-maxw-1, minh, cur_img.shape[2]-maxh-1)).bool()
        newmask = torch.clamp(nowmask + cur_mask, 0,1).bool()
        nowimg = torch.where(cur_mask & ~overlaps, cur_img, nowimg)
        nowimg = torch.where(overlaps & cut_mask, cur_img, nowimg)
        nowmask = newmask



        # break


    img = ready(final)



    return img, overlap