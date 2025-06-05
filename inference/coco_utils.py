import json
from pycocotools import mask as coco_mask
import numpy as np
from pathlib import Path

def initialize_coco_data(class_name="defect"):
    """
    Initialize the COCO data structure.
    :param class_name: the class name to be used for annotations
    :return: Initialized COCO data structure
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": class_name, "supercategory": class_name}],
    }
    return coco_data



def convert_mask_to_coco_annotation(mask, image_id, annotation_id_start):
    import numpy as np
    from pycocotools import mask as mask_utils

    # encode the mask
    rle_mask = mask_utils.encode(np.asfortranarray(mask))

    annotations = []

    # 🔥 如果是 list，loop 一個一個處理
    if isinstance(rle_mask, list):
        for single_rle in rle_mask:
            if isinstance(single_rle, dict):
                single_rle["counts"] = single_rle["counts"].decode("utf-8")
                area = mask_utils.area(single_rle).item()
                bbox = mask_utils.toBbox(single_rle).tolist()

                annotation = {
                    "id": annotation_id_start,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": single_rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
                annotations.append(annotation)
                annotation_id_start += 1
            else:
                raise TypeError(f"Expected single_rle to be dict, but got {type(single_rle)}")
    elif isinstance(rle_mask, dict):
        rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
        area = mask_utils.area(rle_mask).item()
        bbox = mask_utils.toBbox(rle_mask).tolist()

        annotation = {
            "id": annotation_id_start,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": rle_mask,
            "area": area,
            "bbox": bbox,
            "iscrowd": 0,
        }
        annotations.append(annotation)
    else:
        raise TypeError(f"Expected rle_mask to be dict or list, but got {type(rle_mask)}")

    return annotations

def save_coco_annotations(coco_data, output_dir, img_name):
    """
    Save the COCO data structure to a JSON file.
    :param coco_data: The COCO data structure
    :param output_dir: Directory to save the COCO annotations file
    """
    output_file = Path(output_dir) / f"{img_name}.json"
    with open(output_file, "w") as json_file:
        json.dump(coco_data, json_file)
    print(f"COCO annotations saved to {output_file}")

def convert_masks_to_coco_format(masks_dict, output_dir, img_name, img_shape, class_name="defect"):
    """
    Convert masks to COCO format and save the annotations in a JSON file.
    :param masks_dict: dictionary of masks with image name as key
    :param img_name: image name
    :param img_shape: image height and width (height, width)
    :param class_name: the class name to be used for annotations
    :param coco_output_dir: directory to save the coco annotations file
    """
    output_dir = Path(output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir / "coco"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize COCO data structure
    coco_data = initialize_coco_data(class_name)
    

    # Add image information
    img_id = 1  # COCO image id
    image_info = {
        "id": img_id,
        "file_name": f"{img_name}.PNG",
        "width": img_shape[1],
        "height": img_shape[0],
    }
    coco_data["images"].append(image_info)

    # Process each mask and add to coco annotations
    annotation_id = 1  # COCO annotation id
    for mask in masks_dict[img_name]:
        # annotation = convert_mask_to_coco_annotation(mask, img_id, annotation_id)
        # coco_data["annotations"].append(annotation)
        # annotation_id += 1
        # 接收是 list of annotations
        annotations = convert_mask_to_coco_annotation(mask, img_id, annotation_id)

        for ann in annotations:
            coco_data["annotations"].append(ann)
        annotation_id += len(annotations)

        

    # Save the COCO annotations to a JSON file
    save_coco_annotations(coco_data, output_dir, img_name)