import cv2
import numpy as np
import scipy
import torch


def edge_pad(img, mask, mode=1):
    mask = 255 - mask
    if mode == 0:
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res0 = 1 - nmask
        res1 = nmask
        p0 = np.stack(res0.nonzero(), axis=0).transpose()
        p1 = np.stack(res1.nonzero(), axis=0).transpose()
        min_dists, min_dist_idx = scipy.cKDTree(p1).query(p0, 1)
        loc = p1[min_dist_idx]
        for (a, b), (c, d) in zip(p0, loc):
            img[a, b] = img[c, d]
    elif mode == 1:
        record = {}
        kernel = [[1] * 3 for _ in range(3)]
        nmask = mask.copy()
        nmask[nmask > 0] = 1
        res = scipy.signal.convolve2d(
            nmask, kernel, mode="same", boundary="fill", fillvalue=1
        )
        res[nmask < 1] = 0
        res[res == 9] = 0
        res[res > 0] = 1
        ylst, xlst = res.nonzero()
        queue = [(y, x) for y, x in zip(ylst, xlst)]
        # bfs here
        cnt = res.astype(np.float32)
        acc = img.astype(np.float32)
        step = 1
        h = acc.shape[0]
        w = acc.shape[1]
        offset = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while queue:
            target = []
            for y, x in queue:
                val = acc[y][x]
                for yo, xo in offset:
                    yn = y + yo
                    xn = x + xo
                    if 0 <= yn < h and 0 <= xn < w and nmask[yn][xn] < 1:
                        if record.get((yn, xn), step) == step:
                            acc[yn][xn] = acc[yn][xn] * cnt[yn][xn] + val
                            cnt[yn][xn] += 1
                            acc[yn][xn] /= cnt[yn][xn]
                            if (yn, xn) not in record:
                                record[(yn, xn)] = step
                                target.append((yn, xn))
            step += 1
            queue = target
        img = acc.astype(np.uint8)
    else:
        nmask = mask.copy()
        ylst, xlst = nmask.nonzero()
        yt, xt = ylst.min(), xlst.min()
        yb, xb = ylst.max(), xlst.max()
        content = img[yt : yb + 1, xt : xb + 1]
        img = np.pad(
            content,
            ((yt, mask.shape[0] - yb - 1), (xt, mask.shape[1] - xb - 1), (0, 0)),
            mode="edge",
        )
    return img, 255 - mask


def cv2_telea(img, mask):
    ret = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    return ret, mask


def cv2_ns(img, mask):
    ret = cv2.inpaint(img, mask, 5, cv2.INPAINT_NS)
    return ret, mask


class FillImageForOutpainting:
    fill_methods = ["cv2_ns", "cv2_telea", "edge_pad"]
    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "fill_method": (s.fill_methods,),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")

    FUNCTION = "fill_image"
    CATEGORY = "image"

    def fill_image(self, image, mask, fill_method):
        image = (image.numpy() * 255).astype(np.uint8)[0]
        mask = (mask.numpy() * 255).astype(np.uint8)
        if fill_method == "cv2_ns":
            image, mask = cv2_ns(image, mask)
        elif fill_method == "cv2_telea":
            image, mask = cv2_telea(image, mask)
        elif fill_method == "edge_pad":
            image, mask = edge_pad(image, mask)

        image = torch.from_numpy(image) / 255.0
        image = torch.unsqueeze(image, 0)
        mask = torch.from_numpy(mask) / 255.0
        return image, mask


NODE_CLASS_MAPPINGS = {
    "FillImageForOutpainting": FillImageForOutpainting
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FillImageForOutpainting": "Fill Image For Outpainting"
}
