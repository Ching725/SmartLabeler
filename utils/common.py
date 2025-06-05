import numpy as np
import torch
import cv2
import supervision as sv
from torchvision import transforms

def preprocess_image(image):
    # 將影像縮小
    # image = cv2.resize(image, (256, 256))
    # 將影像模糊化
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    # kernel = np.ones((3, 3), np.uint8)
    # # 開運算：去小雜點
    # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # # 閉運算：補小黑洞
    # closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # sv.plot_image(image, (6, 6))
    # 對比度增強
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # cl = clahe.apply(l)
    # limg = cv2.merge((cl, a, b))
    # image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # sv.plot_image(image, (6, 6))#, title="Contrast Enhanced Image")

    return image


def convert_image_to_tensor(image_source, device="cpu"):
    """
    模仿 GroundingDINO 的 load_image() 邏輯：
    - BGR ➜ RGB
    - Normalize to [0, 1]
    - To tensor [C, H, W]
    - Add batch dim [1, C, H, W]
    """
    # Step 1: BGR ➜ RGB
    image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

    # Step 2: To tensor (normalized to [0, 1])
    transform = transforms.Compose([
        transforms.ToTensor(),  # (H, W, C) ➜ (C, H, W), [0~255] ➜ [0~1]
    ])
    image_tensor = transform(image_rgb).to(device)  # ➜ [1, 3, H, W]

    return image_tensor


def filter_large_boxes(boxes, img_w, img_h, threshold=0.9):
    """
    過濾掉寬或高超過圖片大小指定比例的框（預設為 90%）
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    filtered_boxes = []
    for box in boxes:
        cx, cy, bw, bh = box
        if bw > threshold or bh > threshold:
            continue
        filtered_boxes.append([cx, cy, bw, bh])
    return np.array(filtered_boxes)


def filter_large_boxes(boxes, img_w, img_h, threshold=0.9):
    """
    過濾掉寬或高超過圖片大小指定比例的框（預設為 90%）
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    filtered_boxes = []
    for box in boxes:
        cx, cy, bw, bh = box
        if bw > threshold or bh > threshold:
            continue
        filtered_boxes.append([cx, cy, bw, bh])
        
    return np.array(filtered_boxes)


# def filter_large_boxes_and_label_by_area(boxes, logits, img_w, img_h, max_area=0.8, threshold_area=0.25):
#     """
#     過濾掉寬或高超過圖片大小指定比例的框（預設為 90%），
#     並可選擇性地標註面積大於 min_area 的框（面積比例，預設為 10%）
    
#     Args:
#         boxes: [N, 4] 格式為 [cx, cy, w, h] 的框
#         img_w: 圖片寬度
#         img_h: 圖片高度
#         threshold: 寬或高的比例閾值（預設為 0.9）
#         min_area: 框佔整張圖片的面積比例閾值（預設為 0.1）
#         label: 若為 True，則回傳符合面積條件的框
    
#     Returns:
#         np.array(filtered_boxes), np.array(labeled_boxes)（如果 label=True）
#     """
#     check_label=False
#     if isinstance(boxes, torch.Tensor):
#         boxes = boxes.cpu().numpy()
    
#     filtered_boxes = []

#     image_area = img_w * img_h

#     for box in boxes:
#         cx, cy, bw, bh = box
#         box_area = bw * bh # 注意 bw, bh 是比例
#         if box_area > max_area:
#             # 如果面積大於 min_area，則不過濾
#             continue

#         filtered_boxes.append([cx, cy, bw, bh])
#         if threshold_area != None:
#             if box_area > threshold_area and check_label == False:
#                 check_label = True

#     return np.array(filtered_boxes), check_label

def filter_boxes(boxes, logits, conference, threshold_area, max_area=0.8):
    """
    過濾掉寬或高超過圖片大小指定比例的框（預設為 80%），
    並可選擇性地標註面積大於 threshold_area 的框（面積比例閾值）
    
    Args:
        boxes: [N, 4] 格式為 [cx, cy, w, h] 的框
        logits: 模型輸出的框的信心分數
        conference: 信心分數閾值，低於此值的框將被忽略
        threshold_area: 面積閾值，框的面積大於此值會被標註
        max_area: 框的最大面積比例閾值，預設為 0.8
    
    Returns:
        filtered_boxes: 過濾後的框
        check_label: 如果有框符合面積或信心閾值，則返回 True，否則為 False
    """
    check_label = False
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    
    filtered_boxes = []

    for i in range(len(logits)):
        cx, cy, bw, bh = boxes[i]
        box_area = bw * bh  # 注意 bw, bh 是比例，面積計算

        # 過濾掉面積超過 max_area 的框
        if box_area > max_area:
            continue

        filtered_boxes.append([cx, cy, bw, bh])

        # 如果框的面積大於 threshold_area，標註並返回
        if threshold_area is not None and box_area > threshold_area and not check_label:
            check_label = True
            return np.array(filtered_boxes), check_label

        # 如果框的信心分數低於 conference，標註並返回
        if conference is not None and logits[i] < conference:
            check_label = True
            return np.array(filtered_boxes), check_label

    return np.array(filtered_boxes), check_label