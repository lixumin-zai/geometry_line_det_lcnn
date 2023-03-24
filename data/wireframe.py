import os
import sys
import json
from itertools import combinations

import cv2
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt

from scipy.ndimage import zoom



def inrange(v, shape):
    return 0 <= v[0] < shape[0] and 0 <= v[1] < shape[1]


def to_int(x):
    return tuple(map(int, x))


def save_heatmap(prefix, image, lines):
    im_rescale = (512, 512)

    # 最终输出为 128
    heatmap_scale = (128, 128)

    fy, fx = heatmap_scale[1] / image.shape[0], heatmap_scale[0] / image.shape[1]
    # (1, 128, 128)
    jmap = np.zeros((1,) + heatmap_scale, dtype=np.float32)
    # (1, 2, 128, 128)
    joff = np.zeros((1, 2) + heatmap_scale, dtype=np.float32)
    # (128, 128)
    lmap = np.zeros(heatmap_scale, dtype=np.float32)

    # print(lines[:, :, 0])  # 所有 x 值
    # print(lines[:, :, 1])  # 所有 y 值
    # np.clip是一个截取函数,用于截取数组中小于或者大于某值的部分,并使得被截取部分等于固定值。
    lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4)  # 所有 x 值 映射到 w 为 128 上
    lines[:, :, 1] = np.clip(lines[:, :, 1] * fy, 0, heatmap_scale[1] - 1e-4)  # 所有 y 值 映射到 h 为 128 上

    """ 
    [[[ 63.47664835  25.44342508]
    [ 23.56456044 101.57798165]]

    [[ 23.56456044 101.57798165]
    [103.38873626 101.18654434]]

    [[103.38873626 101.18654434]
    [ 63.47664835  25.44342508]]]

        y               x
    [[[ 25.44342508  63.47664835]
    [101.57798165  23.56456044]]

    [[101.57798165  23.56456044]
    [101.18654434 103.38873626]]

    [[101.18654434 103.38873626]
    [ 25.44342508  63.47664835]]]
    """
    lines = lines[:, :, ::-1]  # x 和 y 交换


    junc = []
    jids = {}

    def jid(jun):
        jun = tuple(jun[:2])
        if jun in jids:
            return jids[jun]
        jids[jun] = len(junc)
        junc.append(np.array(jun + (0,)))
        return len(junc) - 1

    lnid = []   # 线段连接的点的下标  [(1,2), ]   第一个点链接到第二个点
    lpos, lneg = [], []
    for v0, v1 in lines:
        """根据直线，得到端点坐标
        """
        lnid.append((jid(v0), jid(v1)))
        lpos.append([junc[jid(v0)], junc[jid(v1)]])

        vint0, vint1 = to_int(v0), to_int(v1)
        jmap[0][vint0] = 1
        jmap[0][vint1] = 1
        rr, cc, value = skimage.draw.line_aa(*to_int(v0), *to_int(v1))
        lmap[rr, cc] = np.maximum(lmap[rr, cc], value)

    print(jmap.shape[1])
    # print(lmap)
    # 偏移
    for v in junc:  
        vint = to_int(v[:2])
        print(v[:2], vint, 0.5)
        print(v[:2] - vint - 0.5)
        joff[0, :, vint[0], vint[1]] = v[:2] - vint - 0.5

    llmap = zoom(lmap, [0.5, 0.5])
    lineset = set([frozenset(l) for l in lnid])
    for i0, i1 in combinations(range(len(junc)), 2):
        if frozenset([i0, i1]) not in lineset:
            v0, v1 = junc[i0], junc[i1]
            vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)
            rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
            lneg.append([v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))])

    if len(lneg) == 0:
        for i0, i1 in combinations(range(len(junc)), 2):
            v0, v1 = junc[i0], junc[i1]
            vint0, vint1 = to_int(v0[:2] / 2), to_int(v1[:2] / 2)
            rr, cc, value = skimage.draw.line_aa(*vint0, *vint1)
            lneg.append([v0, v1, i0, i1, np.average(np.minimum(value, llmap[rr, cc]))])
            break

    assert len(lneg) != 0
    lneg.sort(key=lambda l: -l[-1])

    junc = np.array(junc, dtype=np.float32)
    Lpos = np.array(lnid, dtype=np.int)
    Lneg = np.array([l[2:4] for l in lneg][:4000], dtype=np.int)
    lpos = np.array(lpos, dtype=np.float32)
    lneg = np.array([l[:2] for l in lneg[:2000]], dtype=np.float32)

    # 直接resize为（512，512）
    image = cv2.resize(image, im_rescale)

    # plt.subplot(131), plt.imshow(lmap)
    # plt.subplot(132), plt.imshow(image)
    # for i0, i1 in Lpos:
    #     plt.scatter(junc[i0][1] * 4, junc[i0][0] * 4)
    #     plt.scatter(junc[i1][1] * 4, junc[i1][0] * 4)
    #     plt.plot([junc[i0][1] * 4, junc[i1][1] * 4], [junc[i0][0] * 4, junc[i1][0] * 4])
    # plt.subplot(133), plt.imshow(lmap)
    # for i0, i1 in Lneg[:150]:
    #     plt.plot([junc[i0][1], junc[i1][1]], [junc[i0][0], junc[i1][0]])
    # plt.show()

    # For junc, lpos, and lneg that stores the junction coordinates, the last
    # dimension is (y, x, t), where t represents the type of that junction.  In
    # the wireframe dataset, t is always zero.
    np.savez_compressed(
        f"{prefix}_label.npz",
        aspect_ratio=image.shape[1] / image.shape[0],
        jmap=jmap,  # [J, H, W]    Junction heat map     点的热图
        joff=joff,  # [J, 2, H, W] Junction offset within each pixel  偏移 off
        lmap=lmap,  # [H, W]       Line heat map with anti-aliasing    线的热图
        junc=junc,  # [Na, 3]      Junction coordinate   连接点坐标
        Lpos=Lpos,  # [M, 2]       Positive lines represented with junction indices   正样本热点链接成线的点组下标[] [(1,2), ]   第一个点链接到第二个点 
        Lneg=Lneg,  # [M, 2]       Negative lines represented with junction indices   负样本热点之间连成线点组下标[]
        lpos=lpos,  # [Np, 2, 3]   Positive lines represented with junction coordinates   对应的值 [array([25.44342508, 63.47664835,  0. ]), array([101.57798165,  23.56456044,   0. ])]， 。。。】
        lneg=lneg,  # [Nn, 2, 3]   Negative lines represented with junction coordinates
    )
    cv2.imwrite(f"{prefix}.png", image)

    # plt.imshow(jmap[0])
    # plt.savefig("/tmp/1jmap0.jpg")
    # plt.imshow(jmap[1])
    # plt.savefig("/tmp/2jmap1.jpg")
    # plt.imshow(lmap)
    # plt.savefig("/tmp/3lmap.jpg")
    # plt.imshow(Lmap[2])
    # plt.savefig("/tmp/4ymap.jpg")
    # plt.imshow(jwgt[0])
    # plt.savefig("/tmp/5jwgt.jpg")
    # plt.cla()
    # plt.imshow(jmap[0])
    # for i in range(8):
    #     plt.quiver(
    #         8 * jmap[0] * cdir[i] * np.cos(2 * math.pi / 16 * i),
    #         8 * jmap[0] * cdir[i] * np.sin(2 * math.pi / 16 * i),
    #         units="xy",
    #         angles="xy",
    #         scale_units="xy",
    #         scale=1,
    #         minlength=0.01,
    #         width=0.1,
    #         zorder=10,
    #         color="w",
    #     )
    # plt.savefig("/tmp/6cdir.jpg")
    # plt.cla()
    # plt.imshow(lmap)
    # plt.quiver(
    #     2 * lmap * np.cos(ldir),
    #     2 * lmap * np.sin(ldir),
    #     units="xy",
    #     angles="xy",
    #     scale_units="xy",
    #     scale=1,
    #     minlength=0.01,
    #     width=0.1,
    #     zorder=10,
    #     color="w",
    # )
    # plt.savefig("/tmp/7ldir.jpg")
    # plt.cla()
    # plt.imshow(jmap[1])
    # plt.quiver(
    #     8 * jmap[1] * np.cos(tdir),
    #     8 * jmap[1] * np.sin(tdir),
    #     units="xy",
    #     angles="xy",
    #     scale_units="xy",
    #     scale=1,
    #     minlength=0.01,
    #     width=0.1,
    #     zorder=10,
    #     color="w",
    # )
    # plt.savefig("/tmp/8tdir.jpg")


def main():
    """
    data
        - images
        - train.json
        - valid.json
    """
    data_root = "../../geometry_rec/lcnn/data/whole_datasets/"
    data_output = "./data/123"

    os.makedirs(data_output, exist_ok=True)
    for batch in ["train", "valid"]:
        anno_file = os.path.join(data_root, f"{batch}.json")

        with open(anno_file, "r") as f:
            dataset = json.load(f)

        def handle(data):
            """解析json数据

            Args:
                data (_type_): json数据
                dict ： "filename":[[x1,y1,x2,y2],……]
            """
            im = cv2.imread(os.path.join(data_root, "images", data["filename"]))
            prefix = data["filename"].split(".")[0]  # 图片名字
            print(data["lines"])

            # 将
            """
            [[59.19248466257669, 53.82194244604316, 59.59739263803681, 403.46223021582733], [59.59739263803681, 403.46223021582733, 209.4133435582822, 403.8669064748201], [209.4133435582822, 403.8669064748201, 59.19248466257669, 53.82194244604316]]
            transform
            [[[ 59.19248466  53.82194245]
            [ 59.59739264 403.46223022]]

            [[ 59.59739264 403.46223022]
            [209.41334356 403.86690647]]

            [[209.41334356 403.86690647]
            [ 59.19248466  53.82194245]]]
            """
            lines = np.array(data["lines"]).reshape(-1, 2, 2)
            print(lines)

            os.makedirs(os.path.join(data_output, batch), exist_ok=True)

            lines0 = lines.copy()
            lines1 = lines.copy()

            # 
            """
            [[[175.5526056   63.21100917]
            [ 65.17073747 252.35779817]]

            [[ 65.17073747 252.35779817]
            [285.93447373 251.3853211 ]]

            [[285.93447373 251.3853211 ]
            [175.5526056   63.21100917]]]


            变 h - 原来的值
            [[[178.4473944   63.21100917]
            [288.82926253 252.35779817]]

            [[288.82926253 252.35779817]
            [ 68.06552627 251.3853211 ]]

            [[ 68.06552627 251.3853211 ]
            [178.4473944   63.21100917]]]

                            变 w - 原来的值
            [[[175.5526056  254.78899083]
            [ 65.17073747  65.64220183]]

            [[ 65.17073747  65.64220183]
            [285.93447373  66.6146789 ]]

            [[285.93447373  66.6146789 ]
            [175.5526056  254.78899083]]]

                变              变
            [[[178.4473944  254.78899083]
            [288.82926253  65.64220183]]

            [[288.82926253  65.64220183]
            [ 68.06552627  66.6146789 ]]

            [[ 68.06552627  66.6146789 ]
            [178.4473944  254.78899083]]]
            """
            lines1[:, :, 0] = im.shape[1] - lines1[:, :, 0]
            # print(lines1)
            lines2 = lines.copy()
            lines2[:, :, 1] = im.shape[0] - lines2[:, :, 1]
            # print(lines2)
            lines3 = lines.copy()
            lines3[:, :, 0] = im.shape[1] - lines3[:, :, 0]
            lines3[:, :, 1] = im.shape[0] - lines3[:, :, 1]
            # print(lines3)
            path = os.path.join(data_output, batch, prefix)
            save_heatmap(f"{path}_0", im[::, ::], lines0)
            if batch != "valid":
                save_heatmap(f"{path}_1", im[::, ::-1], lines1)
                save_heatmap(f"{path}_2", im[::-1, ::], lines2)
                save_heatmap(f"{path}_3", im[::-1, ::-1], lines3)
            print("Finishing", os.path.join(data_output, batch, prefix))

        a = 0
        for d in dataset[:1]:
            try:
                handle(d)
            except:
                a += 1   
                continue
        print(a)
        # parmap(handle, dataset, 16)


if __name__ == "__main__":
    main()

