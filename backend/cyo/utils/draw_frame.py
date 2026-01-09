import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def draw_frame(img_path, detections, category):
    # 1. 加载图像 (PIL 默认就是 RGB)
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    def get_color(class_id):
        np.random.seed(class_id)
        # PIL 接收 tuple 格式的 RGB 颜色
        return tuple(np.random.randint(0, 255, 3).tolist())

    # 设置基础尺寸参考
    thickness = max(2, int(width / 500))
    # PIL 的字体需要指定大小，这里根据图片宽度动态计算
    font_size = max(12, int(width / 50))
    try:
        # 尝试加载系统字体，如果失败则使用默认字体
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        bbox = det['bbox']
        class_name = det['class_name']
        conf = det['confidence']
        class_id = det['class_id']

        x1, y1, x2, y2 = bbox
        color = get_color(class_id=class_id)

        # --- 1. 绘制矩形框 ---
        # PIL 的 line width 可以直接指定
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        # --- 2. 准备文本 ---
        label = f"{class_name} {conf:.2f}"
        
        # 获取文本占用的像素范围 [left, top, right, bottom]
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # 计算标签背景的位置（确保不越界）
        label_y1 = max(y1 - text_h - 10, 0)
        label_y2 = label_y1 + text_h + 10
        
        # --- 3. 绘制标签背景 ---
        draw.rectangle([x1, label_y1, x1 + text_w + 10, label_y2], fill=color)

        # --- 4. 绘制文本 ---
        draw.text((x1 + 5, label_y1 + 2), label, fill=(255, 255, 255), font=font)

    # 2. 修改保存逻辑
    save_dir = f"static/{category}"
    path = Path(save_dir)
    path.mkdir(exist_ok=True, parents=True)
    
    # 获取原始文件名，并将后缀统一修改为 .png，确保无损
    file_stem = Path(img_path).stem  # 获取不带后缀的文件名
    save_path = os.path.join(save_dir, f"{file_stem}.png")

    # 3. 执行无损保存
    # PNG 格式默认就是无损的。optimize=True 会在不损失画质的前提下进一步压缩体积
    img.save(save_path, format="PNG", optimize=True)
    
    print(f"✅ 已完成无损保存至: {save_path}")
    return save_path