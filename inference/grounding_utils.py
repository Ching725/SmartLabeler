import numpy as np
import cv2
import torch
from groundingdino.util.inference import load_image, predict, annotate
import supervision as sv
# from unils.common import preprocess_image, convert_image_to_tensor


def crop_largest_box(image_np, boxes):
    if not isinstance(image_np, np.ndarray):
        raise TypeError(f"image_np must be numpy.ndarray, but got {type(image_np)}")

    if len(boxes) == 0:
        raise ValueError("No bounding boxes found to crop")

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    h, w = image_np.shape[:2]

    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h

    boxes_abs = np.stack([x1, y1, x2, y2], axis=1).astype(int)
    areas = abs(boxes_abs[:, 2] - boxes_abs[:, 0]) * (boxes_abs[:, 3] - boxes_abs[:, 1])
    largest_idx = np.argmax(areas)
    x1, y1, x2, y2 = boxes_abs[largest_idx].astype(int)

    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))

    cropped = image_np[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

def object_detection_with_GroundingDINO(img_path, prompt, device, grounding_model, crop=False):
    box_threshold = 0.3
    text_threshold = 0.25

    image_source, image = load_image(img_path)
    # image_source = preprocess_image(image_source)
    # image_Tensor = convert_image_to_tensor(image_source, device=device)

    boxes, logits, phrases = predict(
        model=grounding_model,
        image=image,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # sv.plot_image(annotated_frame, (8, 8))

    if crop:
        image_cropped = crop_largest_box(image_source, boxes)
    else:
        image_cropped = None
    return image_cropped, boxes, logits