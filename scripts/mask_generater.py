import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_bottom_middle_mask(image_path, save_mask_path=None, save_overlay_path=None):
    try:
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size
        img_np = np.array(img)
    except Exception as e:
        print(f"??????: {e}")
        return None, None

    # ???mask
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    center_x = img_width // 2
    x_start = max(0, center_x - 100)
    x_end = min(img_width, center_x + 130)
    y_start = max(0, img_height - 140)
    y_end = img_height
    region = {"x_start": x_start, "x_end": x_end, "y_start": y_start, "y_end": y_end}


    mask[y_start:y_end, x_start:x_end] = 1

    if save_mask_path:
        mask_visual = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_visual).save(save_mask_path)
        print(f"?mask?????: {save_mask_path}")


    if save_overlay_path:

        mask_color = np.zeros_like(img_np)
        mask_color[mask == 1] = [255, 0, 0]  # ?? (R, G, B)

        overlay = cv2.addWeighted(img_np, 0.7, mask_color, 0.3, 0)


        overlay = cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_end, y_end),
            (0, 255, 0),  # ?? (B, G, R) - OpenCV?BGR??
            2
        )

        overlay_pil = Image.fromarray(overlay)
        draw = ImageDraw.Draw(overlay_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        text = f"??: ({x_start},{y_start}) - ({x_end},{y_end})"
        draw.text((10, 10), text, fill=(0, 255, 0), font=font)

        overlay_pil.save(save_overlay_path)
        print(f"overlay path :{save_overlay_path}")

    return mask, region


# ------------------- ???? -------------------
if __name__ == "__main__":
    input_image = "/data/dataset/gemini2/mm3dgs/gemini_02/rgb/rgb_000011.png"

    mask, region = generate_bottom_middle_mask(
        image_path=input_image,
        save_mask_path="mask.png",
        save_overlay_path="mask_overlay.png"
    )

    if mask is not None:
        print(f"\n???????")
        print(f"  ????: {region['x_start']} ~ {region['x_end']}")
        print(f"  ????: {region['y_start']} ~ {region['y_end']}")
        print(f"  ????: {region['x_end'] - region['x_start']} x {region['y_end'] - region['y_start']} ??")
