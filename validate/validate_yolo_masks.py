import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def get_yolo_polygons(yolo_file, img_w, img_h):
    """
    從 YOLO 檔案中提取多邊形座標。
    :param yolo_file: Path to YOLO txt file
    :param img_w: image width
    :param img_h: image height
    :return: list of polygons (np.array of shape (N,2))
    """
    yolo_polys = []
    with open(yolo_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = list(map(float, parts[1:]))
            poly = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * img_w)
                y = int(coords[i+1] * img_h)
                poly.append([x, y])
            yolo_polys.append(np.array(poly, dtype=np.int32))
    return yolo_polys

def compute_iou(mask1, mask2):
    """
    計算兩個二值化遮罩的 IOU 值。
    :param mask1: binary mask
    :param mask2: binary mask
    :return: iou float
    """
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    intersection = np.logical_and(mask1_bool, mask2_bool).sum()
    union = np.logical_or(mask1_bool, mask2_bool).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou

def draw_visualization(yolo_polys, gt_binary, img_h, img_w):
    """
    畫出 YOLO polygons 和 GT Mask 的可視化。
    :param yolo_polys: list of polygons (np.array)
    :param gt_binary: binary GT mask
    :param img_h: image height
    :param img_w: image width
    :return: visualization image (H,W,3) uint8
    """
    yolo_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for poly in yolo_polys:
        cv2.fillPoly(yolo_mask, [poly], 255)

    img_vis = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img_vis[..., 0] = yolo_mask  # 紅色通道為 YOLO polygons
    img_vis[..., 1] = gt_binary  # 綠色通道為 GT mask

    cv2.putText(img_vis, "YOLO: yellow", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    cv2.putText(img_vis, "GT Mask: green", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

    return img_vis


def save_overlay_visualization(img_name, image_dir, yolo_mask, gt_binary, save_dir, image_shape):
    img_h, img_w = image_shape
    print('image_dir:', image_dir)
    print('img_name:', img_name)
    img_path = image_dir / f"{img_name}.PNG"
    print('img_path:', img_path)

    if img_path.exists():
        raw_image = cv2.imread(str(img_path))
        raw_image = cv2.resize(raw_image, (img_w, img_h))

        # 建立彩色遮罩圖層
        overlay_mask = np.zeros_like(raw_image)
        overlay_mask[gt_binary > 0] = [0, 255, 0]     # 綠色：GT mask
        overlay_mask[yolo_mask > 0] = [0, 255, 255]   # 黃色：YOLO 預測
        # 分別標記區域
        # overlap = np.logical_and(gt_binary > 0, yolo_mask > 0)
        # gt_only = np.logical_and(gt_binary > 0, yolo_mask == 0)
        # yolo_only = np.logical_and(yolo_mask > 0, gt_binary == 0)

        # 標記不同區域為不同顏色
        # overlay_mask[gt_only] = [0, 255, 0]        # 綠色：GT Only
        # overlay_mask[yolo_only] = [0, 255, 255]    # 黃色：YOLO Only
        # overlay_mask[overlap] = [255, 255, 255]      # 白色：GT + YOLO Overlap


        # 疊加原圖與遮罩圖層（半透明效果）
        overlay = cv2.addWeighted(raw_image, 0.8, overlay_mask, 0.3, 0)

        # 圖例文字
        cv2.putText(overlay, "DAGM Label: green", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        cv2.putText(overlay, "Predictive Label: yellow", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
        # cv2.putText(overlay, "Overlap: transparent", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)    
        # 儲存疊圖結果
        # overlay_save_dir = save_dir / "overlay"
        # overlay_save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{img_name}_vis.png"
        # overlay_path = overlay_save_dir / f"{img_name}_overlay.jpg"
        cv2.imwrite(str(save_path), overlay)
    else:
        print(f"⚠️ 找不到原圖：{img_path}")


# def process_yolo_file(yolo_file, gt_mask_dir, img_h, img_w, save_dir):
def process_yolo_file(yolo_file, gt_mask_dir, image_dir, img_h, img_w, save_dir):
    img_name = yolo_file.stem
    gt_mask_path = gt_mask_dir / f"{img_name}_label.PNG"
    if not gt_mask_path.exists():
        yolo_polys = get_yolo_polygons(yolo_file, img_w, img_h)
        img_vis = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        for poly in yolo_polys:
            cv2.fillPoly(img_vis, [poly], (255, 255, 0))  # 黃色通道為 YOLO polygons
        cv2.putText(img_vis, "Misjudgment, this image has no defect.", (10, img_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 155, 255), 2)
        save_path = save_dir / f"{img_name}_vis.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
        return

    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        print(f"⚠️ 無法讀取 GT mask: {gt_mask_path}, 跳過 {img_name}")
        return
    _, gt_binary = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

    yolo_polys = get_yolo_polygons(yolo_file, img_w, img_h)

    yolo_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for poly in yolo_polys:
        cv2.fillPoly(yolo_mask, [poly], 255)

    iou = compute_iou(gt_binary, yolo_mask)
    print('iou:', iou)

    # 疊圖視覺化
    save_overlay_visualization(img_name, image_dir, yolo_mask, gt_binary, save_dir, (img_h, img_w))

# def validate_yolo_masks(yolo_dir, gt_mask_dir, image_shape, save_dir):
def validate_yolo_masks(yolo_dir, gt_mask_dir, image_dir, image_shape, save_dir):
    """
    驗證 YOLO segmentation 標註與 ground truth mask 的重疊情況，並可視化結果。
    :param yolo_dir: YOLO segmentation txt 檔案資料夾路徑
    :param gt_mask_dir: ground truth mask png 檔案資料夾路徑
    :param image_shape: tuple (height, width)
    :param save_dir: 輸出圖片資料夾路徑
    """
    yolo_dir = Path(yolo_dir)
    gt_mask_dir = Path(gt_mask_dir)
    save_dir = Path(save_dir)
    image_dir = Path(image_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    img_h, img_w = image_shape

    yolo_files = list(yolo_dir.glob("*.txt"))
    if not yolo_files:
        print(f"❌ 找不到任何 YOLO 標註檔案於 {yolo_dir}")
        return

    for yolo_file in yolo_files:
        # process_yolo_file(yolo_file, gt_mask_dir, img_h, img_w, save_dir)
        process_yolo_file(yolo_file, gt_mask_dir, image_dir, img_h, img_w, save_dir)



if __name__ == "__main__":
    # validate_yolo_masks(yolo_dir, gt_mask_dir, image_shape, save_dir)
    # validate_yolo_masks(
    #     "DAGM_2007_Dataset_Class2/Train/test/labels",
    #     "DAGM_2007_Dataset_Class2/Train/test/Label_org",
    #     (512, 512),
    #     "DAGM_2007_Dataset_Class2/Train/test/validation_vis"
    # )
    validate_yolo_masks(
        "DAGM_2007_Dataset_Class2/Train/test/labels",
        "DAGM_2007_Dataset_Class2/Train/test/Label_org",
        "DAGM_2007_Dataset_Class2/Train/test/images",
        (512, 512),
        "DAGM_2007_Dataset_Class2/Train/test/validation_vis"
    )