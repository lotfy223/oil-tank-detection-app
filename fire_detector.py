import os
import numpy as np
from PIL import Image
import torch
import streamlit as st
from transformers import AutoImageProcessor, SiglipForImageClassification

# Reuse the existing non-maximum suppression function
from detector import non_max_suppression

@st.cache_resource
def load_fire_model():
    """
    Loads the Hugging Face fire/smoke classification model and image processor.
    Cached so it only runs once per app session.
    """
    model_name = "prithivMLmods/Forest-Fire-Detection"
    
    # Show loading message during weight download
    model = SiglipForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


def sliding_window_fire_detection(image, model, processor, confidence_threshold=0.5, win_size=128, stride=64, iou_threshold=0.3):
    """
    Performs sliding-window fire/smoke detection over a full satellite/aerial image.
    
    Args:
        image: PIL Image object (any resolution).
        model: Loaded Hugging Face model.
        processor: Loaded Hugging Face image processor.
        confidence_threshold: Float, minimum probability to classify a window as Fire/Smoke.
        win_size: Int, window dimension in pixels (default 128).
        stride: Int, step size for sliding the window (default 64).
        iou_threshold: Float, intersection-over-union threshold for NMS (default 0.3).
        
    Returns:
        List of bounding boxes: (x1, y1, x2, y2, label, confidence).
    """
    # Convert image to RGB numpy array
    img_array = np.array(image.convert("RGB"))
    H, W, C = img_array.shape

    # Handle images smaller than the win_size by padding
    if H < win_size or W < win_size:
        pad_y = max(0, win_size - H)
        pad_x = max(0, win_size - W)
        img_array = np.pad(img_array, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
        H, W, C = img_array.shape

    patches = []
    coords = []

    # Generate sliding window coordinates, ensuring we cover the boundaries
    y_indices = list(range(0, H - win_size + 1, stride))
    if not y_indices or y_indices[-1] != H - win_size:
        y_indices.append(H - win_size)

    x_indices = list(range(0, W - win_size + 1, stride))
    if not x_indices or x_indices[-1] != W - win_size:
        x_indices.append(W - win_size)

    # Extract all patches
    for y in y_indices:
        for x in x_indices:
            patch_np = img_array[y:y+win_size, x:x+win_size]
            if patch_np.shape == (win_size, win_size, 3):
                # Save as PIL Image since processor expects it
                patch_img = Image.fromarray(patch_np)
                patches.append(patch_img)
                coords.append((x, y))

    if not patches:
        return []

    # Run predictions in batches of 32 to prevent VRAM/RAM exhaustion
    batch_size = 32
    all_probs = []
    
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i+batch_size]
        # Hugging Face processor normalizes and prepares images automatically
        inputs = processor(images=batch_patches, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # SiglipForImageClassification outputs logits for 3 classes
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        all_probs.append(probs.cpu().numpy())

    probs_np = np.concatenate(all_probs, axis=0) # shape (N, 3)

    # Label indices mapping: 0 -> Fire, 1 -> Normal, 2 -> Smoke
    fire_candidates = []
    smoke_candidates = []

    for idx, prob in enumerate(probs_np):
        fire_conf = float(prob[0])
        smoke_conf = float(prob[2])
        x, y = coords[idx]
        
        if fire_conf > confidence_threshold:
            fire_candidates.append([
                float(x), 
                float(y), 
                float(x + win_size), 
                float(y + win_size), 
                float(fire_conf)
            ])
            
        if smoke_conf > confidence_threshold:
            smoke_candidates.append([
                float(x), 
                float(y), 
                float(x + win_size), 
                float(y + win_size), 
                float(smoke_conf)
            ])

    # Run NMS independently for each class
    final_fire = non_max_suppression(fire_candidates, iou_threshold=iou_threshold)
    final_smoke = non_max_suppression(smoke_candidates, iou_threshold=iou_threshold)

    # Format output boxes: (x1, y1, x2, y2, label, confidence)
    results = []
    for box in final_fire:
        results.append((box[0], box[1], box[2], box[3], "Fire", box[4]))
    for box in final_smoke:
        results.append((box[0], box[1], box[2], box[3], "Smoke", box[4]))

    return results
