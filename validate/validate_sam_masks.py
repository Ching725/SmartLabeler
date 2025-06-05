
from pathlib import Path
import cv2
import numpy as np

def validate_masks(img_name, masks, gt_mask_dir, image_dir, save_dir, image_shape):
    """
    使用 SAM 預測的 masks 與 ground truth mask 做驗證並可視化。
    :param img_name: 圖片名稱 (不含副檔名)
    :param masks: SAM 預測的 mask (numpy array or tensor)
    :param gt_mask_dir: ground truth mask 資料夾路徑
    :param image_dir: 原始圖片資料夾路徑
    :param save_dir: 輸出資料夾路徑
    :param image_shape: (height, width)
    """
    img_h, img_w = image_shape
    gt_mask_path = Path(gt_mask_dir) / f"{img_name}_label.PNG"
    save_dir = Path(save_dir)
    image_dir = Path(image_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not gt_mask_path.exists():
        print(f"⚠️ 找不到 GT mask：{gt_mask_path}，只顯示 SAM 預測結果")
        gt_binary = np.zeros((img_h, img_w), dtype=np.uint8)
    else:
        # 讀取 Ground Truth mask
        gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"⚠️ 無法讀取 GT mask：{gt_mask_path}，只顯示 SAM 預測結果")
            gt_binary = np.zeros((img_h, img_w), dtype=np.uint8)
        else:
            _, gt_binary = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

    # 處理 SAM 預測的 masks
    if isinstance(masks, np.ndarray):
        sam_mask = masks.squeeze()
    else:
        sam_mask = masks.cpu().squeeze().numpy()

    # 如果有多個 mask，可以選最大一個或者全部合併
    if sam_mask.ndim == 3:  # 多個 mask
        combined_mask = np.any(sam_mask, axis=0).astype(np.uint8) * 255
    else:
        combined_mask = (sam_mask > 0).astype(np.uint8) * 255

    # 計算 IoU
    intersection = np.logical_and(gt_binary > 0, combined_mask > 0).sum()
    union = np.logical_or(gt_binary > 0, combined_mask > 0).sum()
    iou = intersection / union if union > 0 else 0.0
    if union > 0:
        print(f"{img_name} IoU: {iou:.4f}")
    else:
        print(f"{img_name} 無 GT mask，無法計算 IoU")

    # 讀取原始圖
    img_path = image_dir / f"{img_name}.PNG"
    if not img_path.exists():
        print(f"⚠️ 找不到原始圖片：{img_path}")
        return

    raw_image = cv2.imread(str(img_path))
    raw_image = cv2.resize(raw_image, (img_w, img_h))

    # 建立彩色遮罩圖層
    overlay_mask = np.zeros_like(raw_image)
    overlay_mask[gt_binary > 0] = [0, 255, 0]          # 綠色：GT mask
    overlay_mask[combined_mask > 0] = [0, 255, 255]    # 黃色：SAM 預測

    # 疊加原圖與遮罩圖層（半透明效果）
    overlay = cv2.addWeighted(raw_image, 0.8, overlay_mask, 0.3, 0)

    # 畫上文字
    if gt_binary.sum() > 0:
        cv2.putText(overlay, "GT Label: green", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        cv2.putText(overlay, "SAM Predict: yellow", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
    else:
        cv2.putText(overlay, "SAM Predict Only (yellow)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)

    # 儲存結果
    save_path = save_dir / f"{img_name}_sam_validation.png"
    cv2.imwrite(str(save_path), overlay)