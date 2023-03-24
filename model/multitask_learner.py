from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Head_size = [[2], [1], [2]]
from config import config


class MultitaskHead(nn.Module):
    def __init__(self, input_channels, num_class):
        super(MultitaskHead, self).__init__()

        m = int(input_channels / 4)
        heads = []
        for output_channels in sum(config["head_size"], []):
            heads.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, output_channels, kernel_size=1),
                )
            )
        self.heads = nn.ModuleList(heads)
        assert num_class == sum(sum(config["head_size"], []))

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=1)

class MultitaskLearner(nn.Module):
    def __init__(self, backbone):
        super(MultitaskLearner, self).__init__()
        self.backbone = backbone
        head_size = config["head_size"]
        self.num_class = sum(sum(head_size, []))   # 5
        self.head_off = np.cumsum([sum(h) for h in head_size])  # [2, 3, 5] 

    def forward(self, input_dict):
        image = input_dict["image"]

        # [n, 5, 128, 128]
        # [n, 5, 128, 128]
        # feature [n, 256, 128, 128]
        outputs, feature = self.backbone(image)  # stack2 两个outputs 分数 直线 ， 一个feature 
        result = {"feature": feature}
        batch, channel, row, col = outputs[0].shape  # 第一个output

        # for i in outputs:
        #     print(i.shape)
        #     output = i.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()  # [5, n, 128, 128]
        #     print(output.shape)  
        # print(feature.shape)

        # 真实数据
        T = input_dict["target"].copy() #
        """
        {
            "jmap": , n, 2, 128, 128
            "joff": , n, 2, 128, 128
            "lmap": , n, 1, 128, 128
        }
        
        """
        n_jtyp = T["jmap"].shape[1]   # batchsize

        # switch to CNHW
        for task in ["jmap"]:
            T[task] = T[task].permute(1, 0, 2, 3)  # 换位
        for task in ["joff"]:
            T[task] = T[task].permute(1, 2, 0, 3, 4)  # 

        offset = self.head_off   #[2, 3, 5]

        """
        loss_weight:
            jmap: 8.0
            lmap: 0.5
            joff: 0.25
            lpos: 1
            lneg: 1
        """
        loss_weight = config["loss_weight"]
        # 输入数据上却少loss
        losses = []
        for stack, output in enumerate(outputs):
            # output [n, 5, 128, 128]
            output = output.transpose(0, 1).reshape([-1, batch, row, col]).contiguous()
            # [5, n, 128, 128]

            # 得到
            jmap = output[0 : offset[0]].reshape(n_jtyp, 2, batch, row, col)   # 点预测 mat 图  output 0 - 2 热图   2维  [1, 2, n, 128,128]
            
            lmap = output[offset[0] : offset[1]].squeeze(0)                    # 线预测热图， output中 2 - 3 为 线的特征图  1维 [n,1,128,128]

            joff = output[offset[1] : offset[2]].reshape(n_jtyp, 2, batch, row, col)  # 偏移   3-5  2维 [1, 2, n, 128,128]
            if stack == 0:
                result["preds"] = {
                    "jmap": jmap.permute(2, 0, 1, 3, 4).softmax(2)[:, :, 1],    # [n, 1,2,128,128] -> [n, 1, 128, 128]
                    "lmap": lmap.sigmoid(),                                     # [n,1,128,128]
                    "joff": joff.permute(2, 0, 1, 3, 4).sigmoid() - 0.5,  # 偏移量 [1, 2, n, 128,128] - > [n, 1, 2, 128, 128]
                }
                if input_dict["mode"] == "testing":
                    return result

            L = OrderedDict()
            L["jmap"] = sum(
                cross_entropy_loss(jmap[i], T["jmap"][i]) for i in range(n_jtyp)
            )
            L["lmap"] = (
                F.binary_cross_entropy_with_logits(lmap, T["lmap"], reduction="none")
                .mean(2)
                .mean(1)
            )
            L["joff"] = sum(
                sigmoid_l1_loss(joff[i, j], T["joff"][i, j], -0.5, T["jmap"][i])
                for i in range(n_jtyp)
                for j in range(2)
            )
            for loss_name in L:
                L[loss_name].mul_(loss_weight[loss_name])
            losses.append(L)
        result["losses"] = losses
        return result


def l2loss(input, target):
    return ((target - input) ** 2).mean(2).mean(1)


def cross_entropy_loss(logits, positive):
    nlogp = -F.log_softmax(logits, dim=0)
    return (positive * nlogp[1] + (1 - positive) * nlogp[0]).mean(2).mean(1)


def sigmoid_l1_loss(logits, target, offset=0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp - target)
    if mask is not None:
        w = mask.mean(2, True).mean(1, True)
        w[w == 0] = 1
        loss = loss * (mask / w)

    return loss.mean(2).mean(1)


if __name__ == "__main__":
    import numpy as np
    from hourglass_pose import hg
    data = np.random.randint(0, 256, (2, 3, 512, 512)).astype(np.float32)
    data = torch.tensor(data)

    config = {
        "head": lambda c_in, c_out: MultitaskHead(c_in, c_out),
        "depth": 4,
        "num_stacks": 2,
        "num_blocks": 1,
        "num_classes": 5
    }
    backbone = hg(**config)
    model = MultitaskLearner(backbone)
    inputs = {
        "image": data
    }
    model(inputs)