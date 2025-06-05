import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import torch

def mask_to_polygons(mask):
    if mask.ndim == 3:
        mask = mask[0]
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [contour.squeeze(1) for contour in contours if contour.shape[0] >= 3]
    return polygons

def normalize_polygon(polygon, img_width, img_height):
    normalized = []
    for (x, y) in polygon:
        normalized.append(x / img_width)
        normalized.append(y / img_height)
    return normalized

def write_yolo_seg_txt(filepath, class_name, polygons):
    with open(filepath, 'w') as f:
        for poly in polygons:
            line = [class_name] + [f'{p:.6f}' for p in poly]
            f.write(' '.join(line) + '\n')

def process_and_convert_masks(masks, img_name):
    """
    Process the masks and convert them to numpy format if necessary,
    then wrap them in a dictionary.
    """
    if isinstance(masks, torch.Tensor):
        masks_np_list = masks.cpu().squeeze(0).numpy()
    else:
        masks_np_list = masks

    # Create the dictionary containing the mask
    masks_dict = {
        img_name: list(masks_np_list)
    }
    return masks_dict

def convert_masks_to_yolo_seg(masks_dict, output_dir, img_shape, class_name="defect"):
    output_dir = Path(output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir / "yolo"
    output_dir.mkdir(parents=True, exist_ok=True)
    

    img_h, img_w = img_shape[:2]
    for img_name, masks in tqdm(masks_dict.items()):
        all_polys = []
        for mask in masks:
            polygons = mask_to_polygons(mask)
            for poly in polygons:
                norm_poly = normalize_polygon(poly, img_w, img_h)
                all_polys.append(norm_poly)

        if all_polys:
            output_txt = output_dir / f"{img_name}.txt"
            write_yolo_seg_txt(output_txt, class_name, all_polys)