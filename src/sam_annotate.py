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
import json
import argparse

click = None
# Add variables for box drawing
drawing_box = False
box_mode = False
box_start = None
box_end = None
delete_mode = False

def mouse_callback(event, x, y, flags, param):
    global click, box_start, box_end, drawing_box
    
    if box_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing box
            box_start = (x, y)
            drawing_box = True
            print(f"Started drawing box at: ({x}, {y})")
        elif event == cv2.EVENT_MOUSEMOVE and drawing_box:
            # Update box end point while dragging
            box_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing_box:
            # Finish drawing box
            box_end = (x, y)
            drawing_box = False
            print(f"Finished box: from ({box_start[0]}, {box_start[1]}) to ({box_end[0]}, {box_end[1]})")
    else:
        # Original point selection
        if event == cv2.EVENT_LBUTTONDOWN:
            click = (x, y, True)
            print(f"Clicked positive at: ({x}, {y})")
        if event == cv2.EVENT_RBUTTONDOWN:
            click = (x, y, False)
            print(f"Clicked negative at: ({x}, {y})")

def main(i=0, downscale=4):
    global click, box_mode, box_start, box_end, drawing_box, delete_mode

    confs = json.load(open("../conf.json"))

    input_image_directory = confs["input_image_directory"]
    img_type = confs["img_type"]
    sam_path = confs["sam_path"]

    name = input_image_directory.split('/')[-1]
    if not os.path.exists(f"../segmentations/{name}"):
        os.system(f"mkdir ../segmentations/{name}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=f"{sam_path}/sam2/configs/sam2.1/")
    
    # Set up CUDA device if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load SAM2 model configuration and checkpoint
    sam2_checkpoint = f"{sam_path}/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    predictor = SAM2ImagePredictor(sam2)

    # Create display window and attach mouse callback for interaction
    cv2.namedWindow("display")
    cv2.setMouseCallback("display", mouse_callback)

    # Get sorted list of all images in the directory
    filenames = sorted(glob.glob(f"{input_image_directory}/*.{img_type}"))
    exiting = False
    while True:
        # Get current image filename and create its base name
        filename = filenames[i]
        imgname_raw = filename.split("/")[-1].split(".")[0]
        
        # Create directory for segmentations if it doesn't exist
        if not os.path.exists(f"../segmentations/{name}/{imgname_raw}"):
            os.system(f"mkdir ../segmentations/{name}/{imgname_raw}")
            
        # Load image and set it for the SAM predictor
        img = cv2.imread(filename)
        predictor.set_image(img)
        
        # Create downsized version for display purposes
        disp_img = cv2.resize(img, None, fx=1/downscale, fy=1/downscale)

        mask_mapper = {}

        # Retrieve and combine any existing masks for this image
        old_mask = None
        save_index = 0
        for imgname in glob.glob(f"../segmentations/{name}/{imgname_raw}/*"):
            this_mask = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
            mask_mapper[imgname] = this_mask.copy()
            if old_mask is None:
                old_mask = this_mask
            else:
                # Merge multiple masks by taking the union
                old_mask[this_mask > 0] = 255
            # Track the highest mask index for saving new masks
            save_index = max(save_index, int(imgname.split("/")[-1].split(".")[0]) + 1)
            
        # Resize existing mask for display or create empty mask if none exists
        if old_mask is not None:
            old_mask = cv2.resize(old_mask, None, fx=1/downscale, fy=1/downscale)
        else:
            old_mask = np.zeros((disp_img.shape[0], disp_img.shape[1]), dtype=np.uint8)

        # Initialize segmentation state variables
        masks = None        # Will hold SAM predicted masks
        mask = None         # Current active mask
        small_mask = None   # Downsized version of active mask for display
        sam_index = 0       # Index of currently displayed SAM mask
        sam_mask_amount = 1 # Total number of SAM masks available
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

            # Draw the box if in box mode and have at least a start point
            if box_mode and box_start is not None:
                end_point = box_end if box_end is not None else (box_start[0], box_start[1])
                cv2.rectangle(true_disp, box_start, end_point, (255, 255, 0), 2)

            # Display info on the image
            mode_text = "BOX MODE" if box_mode else "POINT MODE" 
            if delete_mode:
                mode_text = "DELETE MODE"
            file_idx = f"Image {i + 1} of {len(filenames)}"
            display_text = f"{file_idx} | {imgname_raw} | {sam_index + 1}/{sam_mask_amount} | {mode_text}"
            cv2.putText(true_disp, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            
            cv2.imshow("display", true_disp)
            key = cv2.waitKey(20)
            #print(f"Clicked key: {key}")

            if key == ord("b"):
                # Toggle box mode
                box_mode = not box_mode
                box_start = None
                box_end = None
                drawing_box = False
                print(f"Box mode: {'ON' if box_mode else 'OFF'}")

            if key == ord("d"):
                # Toggle delete mode
                if box_mode:
                    # If in box mode, delete the box
                    box_start = None
                    box_end = None
                    drawing_box = False
                delete_mode = not delete_mode
                print(f"Delete mode: {'ON' if delete_mode else 'OFF'}")
                if delete_mode:
                    # Clear points if delete mode is activated
                    pos_points = []
                    neg_points = []
                    masks = None
                    mask = None
                    small_mask = None
                    sam_index = 0
                    sam_mask_amount = 1
                

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
                cv2.imwrite(f"../segmentations/{name}/{imgname_raw}/{save_index:05d}.png", mask)
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


                        # Process box for prediction
            if box_mode and box_start is not None and box_end is not None and not drawing_box:
                # Convert box coordinates to full resolution
                x1, y1 = box_start
                x2, y2 = box_end
                x1, y1 = x1 * downscale, y1 * downscale
                x2, y2 = x2 * downscale, y2 * downscale
                
                # Ensure x1 < x2 and y1 < y2
                box_coords = torch.tensor([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], 
                                          dtype=torch.float32, device=device)
                
                # Predict with box
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_coords[None, :],
                    multimask_output=True,
                )
                
                print(f"Predicted with box: {box_coords.tolist()}")
                print(f"Got masks: {masks.shape}")
                sam_mask_amount = masks.shape[0]
                sam_index = 0
                mask = masks[sam_index] * 255
                small_mask = cv2.resize(mask, None, fx=1/downscale, fy=1/downscale)
                
                # Reset box to allow drawing a new one
                box_start = None
                box_end = None
            if click is not None and not box_mode and not delete_mode:
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
            if delete_mode and click is not None:
                x, y, is_pos = click
                click = None
                x *= downscale
                y *= downscale

                if is_pos:
                    deletion_file = None


                    for k, v in list(mask_mapper.items()):
                        
                        if v[y,x] > 0:
                            # Delete the mask
                            deletion_file = k
                            break
                    if deletion_file is not None:
                        
                        mask_to_delete = mask_mapper[deletion_file]
                        os.remove(deletion_file)
                        mask_mapper.pop(deletion_file)
                        
                        # Regenerate old_mask from scratch with remaining masksq

                        resized_mask_to_delete = cv2.resize(mask_to_delete,  None, fx=1/downscale, fy=1/downscale)
                        old_mask[resized_mask_to_delete > 0] = 0

                        small_mask = None
                        mask = None
                        sam_index = 0
                        sam_mask_amount = 1 
                        pos_points = []
                        neg_points = []



                        print(f"Deleted mask: {k}")

                    # old_mask = np.zeros((disp_img.shape[0], disp_img.shape[1]), dtype=np.uint8)
                    # for _, remaining_mask in mask_mapper.items():
                    #     resized_mask = cv2.resize(remaining_mask, (old_mask.shape[1], old_mask.shape[0]))
                    #     old_mask[resized_mask > 0] = 255
                        
    
        i += 1
        if i >= len(filenames):
            i = 0
        print(f"Showing image {i} of {len(filenames)}")

        if exiting:
            break


if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    parser = argparse.ArgumentParser(description='SAM Annotator')
    parser.add_argument('start_index',type=int, default=0, help='Start index for images')
    parser.add_argument('downscale', type=int, default=4, help='Downscale factor for display')	
    args = parser.parse_args()    
    main(args.start_index, args.downscale)

