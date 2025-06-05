import os
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
import torch

from inference.grounding_utils import object_detection_with_GroundingDINO
from inference.sam_utils import init_sam, load_image_and_generate_masks
from inference.yolo_utils import process_and_convert_masks, convert_masks_to_yolo_seg
from inference.coco_utils import convert_masks_to_coco_format
from utils.common import filter_boxes
from groundingdino.util.inference import load_model
from validate.validate_sam_masks import validate_masks

import cv2
import numpy as np
from pathlib import Path


# 設置日誌配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HOME = os.getcwd()
GroundingDINO_CONFIG_PATH = os.path.join(HOME, "models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GroundingDINO_WEIGHT_PATH = os.path.join(HOME, "models/groundingdino_swint_ogc.pth")

def AutoLabel_image(img_path, prompt, device, grounding_model, predictor, output_dir, confidence, threshold_area, max_area=0.8, class_name="defect", output_formats="both", vaildate=False, gt_mask_dir=None, image_dir=None, save_dir=None):
    img_name = img_path.stem
    logger.info(f'Processing: {img_name}')
    
    cropped_img, boxes, logits = object_detection_with_GroundingDINO(str(img_path), prompt, device, grounding_model, crop=False)
    img_shape = cv2.imread(str(img_path)).shape[:2]

    boxes, check_label = filter_boxes(boxes, logits, confidence, threshold_area, max_area=max_area)
    
    if len(boxes) == 0:
        logger.warning(f"⚠️ {img_name} 沒有偵測到任何物件，已跳過")
        return None

    manual_check_list = []
    if check_label:
        manual_check_list.append(img_path.name)
        logger.warning(f"⚠️ {img_name} 需要人工檢查")

    # Call the function to load the image and generate the masks
    masks, image = load_image_and_generate_masks(img_path, predictor, boxes, device)

    if vaildate:
        # 驗證 + 疊圖
        validate_masks(
            img_name=img_path.stem,
            masks=masks,
            gt_mask_dir=gt_mask_dir,
            image_dir=image_dir,
            save_dir=save_dir,
            image_shape=(512, 512)
        )

    # Process and convert the masks, now wrapped in a dictionary
    masks_dict = process_and_convert_masks(masks, img_name)


    if output_formats in ("yolo", "both"):
        # Convert masks to YOLO format
        convert_masks_to_yolo_seg(masks_dict, output_dir, img_shape, class_name=class_name)

    if output_formats in ("coco", "both"):
        # Convert masks to COCO format
        convert_masks_to_coco_format(masks_dict, output_dir, img_name, img_shape, class_name=class_name)
    

    return manual_check_list

def main(input_dir, output_dir, prompt, output_formats, confidence, threshold_area, max_area=0.8, class_name="defect", vaildate=False, gt_mask_dir=None):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    grounding_model = load_model(GroundingDINO_CONFIG_PATH, GroundingDINO_WEIGHT_PATH).to(device)
    predictor = init_sam(device)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if vaildate:
        # gt_mask_dir = input_dir.parent / "Label_org"   # 假設跟 images 同一層
        validation_vis_dir = input_dir / "validation_vis_sam"  # 同一層新建資料夾
        validation_vis_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning directory: {input_dir.resolve()}")
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    logger.info(f"Found {len(image_paths)} images")

    if len(image_paths) == 0:
        logger.error(f"No images found in folder {input_dir}")
        raise FileNotFoundError(f"No images found in folder {input_dir}")

    # Process each image in sequence
    manual_check_list = []
    for img_path in tqdm(image_paths):
        result = AutoLabel_image(
            img_path, prompt, device, grounding_model, predictor,
            output_dir,
            confidence, threshold_area, max_area, class_name,
            output_formats,      # "yolo", "coco", or "both"
            vaildate=vaildate,   # 驗證打開
            gt_mask_dir=gt_mask_dir,   
            image_dir=input_dir,
            save_dir=validation_vis_dir,
        )
        if result:
            manual_check_list.extend(result)

    # Write the manual check list once at the end
    if manual_check_list:
        manual_check_list_path = output_dir / "manual_check_list.txt"
        with manual_check_list_path.open("w") as f:
            f.write("\n".join(manual_check_list))
        logger.info(f"Manual check list written to {manual_check_list_path}")

if __name__ == "__main__":

    input_dir = "test_data/class3"
    output_dir = "test_data/class3/outputs"
    gt_mask_dir = "test_data/class3/true_masks" 
    main(
        input_dir, 
        output_dir, 
        prompt="crack",  #"pollute", "crack"
        output_formats="yolo", 
        confidence=0.35, 
        threshold_area=0.1, 
        max_area=0.8, 
        class_name="class3", 
        vaildate=True,
        gt_mask_dir=gt_mask_dir
        )