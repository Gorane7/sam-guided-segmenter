import cv2
import glob
import torch
import numpy as np
import os

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from hydra import initialize, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

click = None

def mouse_callback(event, x, y, flags, param):
    global click
    if event == cv2.EVENT_LBUTTONDOWN:
        click = (x, y, True)
        print(f"Clicked positive at: ({x}, {y})")
    if event == cv2.EVENT_RBUTTONDOWN:
        click = (x, y, False)
        print(f"Clicked negative at: ({x}, {y})")


def main():
    global click

    folder = "../Vesneri_2025_03_09/Kristjan"
    img_type = "jpg"
    downscale = 4
    sam2_path = "/home/gorane/code/masters/semester4/machine-vision/sam2"
    i = 0  # Increase this to skip first i images in the folder

    name = folder.split('/')[-1]
    os.system(f"mkdir segmentations/{name}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=f"{sam2_path}/sam2/configs/sam2.1/")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2)

    cv2.namedWindow("display")
    cv2.setMouseCallback("display", mouse_callback)

    filenames = sorted(glob.glob(f"{folder}/*.{img_type}"))
    exiting = False
    while True:
        filename = filenames[i]
        imgname_raw = filename.split("/")[-1].split(".")[0]
        os.system(f"mkdir segmentations/{name}/{imgname_raw}")
        img = cv2.imread(filename)
        predictor.set_image(img)
        disp_img = cv2.resize(img, None, fx=1/downscale, fy=1/downscale)

        old_mask = None
        save_index = 0
        for imgname in glob.glob(f"segmentations/{name}/{imgname_raw}/*"):
            this_mask = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
            if old_mask is None:
                old_mask = this_mask
            else:
                old_mask[this_mask > 0] = 255
            save_index = max(save_index, int(imgname.split("/")[-1].split(".")[0]) + 1)
        if old_mask is not None:
            old_mask = cv2.resize(old_mask, None, fx=1/downscale, fy=1/downscale)
        else:
            old_mask = np.zeros((disp_img.shape[0], disp_img.shape[1]), dtype=np.uint8)

        masks = None
        mask = None
        small_mask = None
        sam_index = 0
        sam_mask_amount = 1
        pos_points = []
        neg_points = []
        
        while True:
            true_disp = disp_img.copy()

            if old_mask is not None:
                green_overlay = np.zeros_like(true_disp)
                green_overlay[:, :] = [0, 255, 0]
                if np.any(old_mask > 0):
                    true_disp[old_mask > 0] = cv2.addWeighted(true_disp[old_mask > 0], 0.5, green_overlay[old_mask > 0], 0.5, 0)

            if small_mask is not None:
                red_overlay = np.zeros_like(true_disp)
                red_overlay[:, :] = [0, 0, 255]
                if np.any(small_mask > 0):
                    true_disp[small_mask > 0] = cv2.addWeighted(true_disp[small_mask > 0], 0.5, red_overlay[small_mask > 0], 0.5, 0)

            cv2.imshow("display", true_disp)
            key = cv2.waitKey(20)
            #print(f"Clicked key: {key}")

            if key == ord("c"):
                break
            if key == ord("q"):
                exiting = True
                break
            if key == ord("r"):
                pos_points = []
                neg_points = []
                masks = None
                mask = None
                small_mask = None
                sam_index = 0
                sam_mask_amount = 1
            if key == ord(" "):
                # TODO: Save current mask
                cv2.imwrite(f"segmentations/{name}/{imgname_raw}/{save_index:05d}.png", mask)
                old_mask[small_mask > 0] = 255
                save_index += 1
                pos_points = []
                neg_points = []
                masks = None
                mask = None
                small_mask = None
                sam_index = 0
                sam_mask_amount = 1
            if key == 81: # Left arrow key
                sam_index = (sam_index - 1) % sam_mask_amount
                print(f"Showing SAM mask {sam_index}")
                mask = masks[sam_index] * 255
                small_mask = cv2.resize(mask, None, fx=1/downscale, fy=1/downscale)
            if key == 83: # Right arrow key
                sam_index = (sam_index + 1) % sam_mask_amount
                print(f"Showing SAM mask {sam_index}")
                mask = masks[sam_index] * 255
                small_mask = cv2.resize(mask, None, fx=1/downscale, fy=1/downscale)
            if click is not None:
                x, y, is_pos = click
                click = None
                x *= downscale
                y *= downscale
                if is_pos:
                    pos_points.append([x, y])
                else:
                    neg_points.append([x, y])
                points = torch.tensor(pos_points + neg_points, dtype=torch.float32, device=device)
                point_labels = torch.tensor([1] * len(pos_points) + [0] * len(neg_points), dtype=torch.int32, device=device)

                masks, scores, _ = predictor.predict(
                    point_coords=points,
                    point_labels=point_labels,
                    box=None,
                    multimask_output=True,
                )
                print(f"Gave points: {points.shape}")
                print(f"Got mask: {masks.shape}")
                sam_mask_amount = masks.shape[0]
                mask = masks[sam_index] * 255
                small_mask = cv2.resize(mask, None, fx=1/downscale, fy=1/downscale)
                print(f"Did SAM, showing sam mask {sam_index}")
        
        i += 1

        if exiting:
            break


if __name__ == '__main__':
    main()
