import torchvision
import torch
import time
import logging
import numpy as np

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)

def xywh2xyxy(x):
    """将 [x_center, y_center, width, height] 转换为 [x1, y1, x2, y2]"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        max_det=300,
        nm=0,  # 掩码数量
):
    """
    非极大值抑制 (NMS)
    输入 prediction 形状: [batch, num_anchors, 4 + num_classes + nm]
    """
    # 针对 YOLOv9 ONNX 导出的维度修正
    # 如果形状是 [1, 84, 8400]，需要转置为 [1, 8400, 84]
    # if prediction.shape[2] > prediction.shape[1]:
    #     prediction = prediction.transpose(1, 2)

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 4  # 类别数量
    mi = 4 + nc  # 掩码起始索引
    
    # 筛选候选框：最大类别置信度 > 阈值
    xc = prediction[..., 4:mi].amax(2) > conf_thres

    # 设置
    max_wh = 7680
    max_nms = 30000
    time_limit = 2.0 + 0.05 * bs
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    
    t = time.time()
    for xi, x in enumerate(prediction):  # 遍历 batch
        x = x[xc[xi]]  # 过滤低置信度框

        if not x.shape[0]:
            continue

        # 拆分 坐标, 类别分数, 掩码
        box, cls, mask = x.split((4, nc, nm), 1)
        box = xywh2xyxy(box)  # 格式转换

        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:
            # 仅保留得分最高的类别
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # 按类别过滤
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        
        # 排序并限制进入 NMS 的数量
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # 批量 NMS (Batched NMS)
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # 类别偏移
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i]

        if (time.time() - t) > time_limit:
            break

    return output

def scale_boxes(img1_shape, boxes, img0_shape):
    """
    专门适配 cv.resize (暴力拉伸) 的坐标还原函数
    img1_shape: 推理尺寸 (例如 640, 640)
    img0_shape: 原图尺寸 (例如 1080, 1920) (h, w)
    """
    # 1. 计算简单的拉伸比例
    # img1_shape 通常是 (h, w), 这里假设是 (640, 640)
    # 注意：opencv resize 是 (width, height)，但 shape 是 (height, width)
    
    gain_y = img0_shape[0] / img1_shape[0]  # 高度缩放比例
    gain_x = img0_shape[1] / img1_shape[1]  # 宽度缩放比例

    # 2. 独立缩放 x 和 y
    boxes[:, [0, 2]] *= gain_x  # x1, x2 乘以 宽度比例
    boxes[:, [1, 3]] *= gain_y  # y1, y2 乘以 高度比例

    # 3. 裁剪越界部分
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, img0_shape[0])
    
    return boxes

def check_img_size(imgsz, s=32):
    """确保图片尺寸是步长 s 的倍数"""
    return int(np.ceil(imgsz / s) * s)