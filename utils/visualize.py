import matplotlib.pyplot as plt
import numpy as np
import cv2

# === 可視化單張 mask 疊圖 ===
# def show_masks(image, masks):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     for i, mask in enumerate(masks):
#         mask_np = mask[0].cpu().numpy()

#         red_mask = np.zeros_like(image, dtype=np.uint8)
#         red_mask[:, :, 0] = 255  # 紅色遮罩
#         red_mask[~mask_np.astype(bool)] = 0  # 非 mask 區設為透明

#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         plt.imshow(red_mask, alpha=0.4)
#         plt.title(f"Mask {i}")
#         plt.axis('off')
#         plt.show()


def show_masks(image, masks):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color=(255, 0, 0)
    alpha=0.4

    for i, mask in enumerate(masks):
        mask_np = mask[0].cpu().numpy()

        if mask_np.dtype != bool:
            mask_np = mask_np.astype(bool)

        overlay = image.copy()
        color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)

        # 遮罩區域進行加權融合
        overlay[mask_np] = ((1 - alpha) * image[mask_np] + alpha * color_arr).astype(np.uint8)

        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(f"Mask {i}")
        plt.axis('off')
        plt.show()