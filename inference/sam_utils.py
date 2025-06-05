import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
# from unils.common import preprocess_image
from utils.visualize import show_masks

import os; print("Working dir:", os.getcwd())

def init_sam(device, model_type="vit_h", checkpoint="models/sam_vit_h_4b8939.pth"):
    import os
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"❌ SAM checkpoint not found: {checkpoint}\n請確認模型檔案是否存在於正確的 models 資料夾中。")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    return predictor

def load_and_set_image(image_path, predictor):
    image = cv2.imread(image_path)
    # image = preprocess_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    return image

def convert_and_mask_with_sam(boxes, image, predictor, device="cpu"):
    if len(boxes) == 0:
        raise ValueError("No bounding boxes provided")

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
    else:
        boxes = boxes.float()

    boxes = boxes.cpu().numpy()
    h, w = image.shape[:2]

    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # 將相對座標轉換為絕對座標 (最後＋-5 是為了避免邊界問題)
    x1 = (cx - bw / 2) * w #+ 5
    y1 = (cy - bh / 2) * h #+ 5
    x2 = (cx + bw / 2) * w #- 5
    y2 = (cy + bh / 2) * h #- 5

    boxes_abs = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    boxes_abs = torch.tensor(boxes_abs, dtype=torch.float32).to(device)

    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_abs, image.shape[:2]
    ).to(device)

    masks, scores, logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False
    )
    return masks, scores, logits, transformed_boxes


def load_image_and_generate_masks(img_path, predictor, boxes, device):
    """
    Loads the image and generates the corresponding masks using SAM.
    """
    image = load_and_set_image(str(img_path), predictor)
    masks, scores, logits, transformed_boxes = convert_and_mask_with_sam(
        boxes=boxes,
        image=image,
        predictor=predictor,
        device=device
    )
    # show_masks(image, masks)  # Show the image with masks overlaid
    return masks, image  # Return the generated masks and the image for further use


# def show_masks(image, masks):
#     import matplotlib.pyplot as plt
#     for i, mask in enumerate(masks):
#         mask_np = mask[0].cpu().numpy()
#         red_mask = np.zeros_like(image, dtype=np.uint8)
#         red_mask[:, :, 0] = 255
#         red_mask[~mask_np.astype(bool)] = 0
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         plt.imshow(red_mask, alpha=0.4)
#         plt.title(f"Mask {i}")
#         plt.axis('off')
#         plt.show()