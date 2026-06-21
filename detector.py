import numpy as np
from PIL import Image

def non_max_suppression(boxes, iou_threshold=0.3):
    """
    Applies Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes.
    
    Args:
        boxes: List of bounding boxes, where each box is [x1, y1, x2, y2, confidence].
        iou_threshold: Float, intersection-over-union threshold for overlap suppression.
        
    Returns:
        List of filtered bounding boxes [x1, y1, x2, y2, confidence].
    """
    if not boxes:
        return []

    # Convert boxes to a numpy array
    boxes_arr = np.array(boxes, dtype=np.float32)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]
    scores = boxes_arr[:, 4]

    # Compute area of the boxes
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort the boxes by confidence score in descending order
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # Index of the box with the highest confidence
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Find the coordinates of the intersection rectangle
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # Calculate width and height of the intersection area
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        # Intersection area
        intersection = w * h

        # Union area
        union = areas[i] + areas[order[1:]] - intersection

        # IoU calculation
        iou = intersection / union

        # Find indices where IoU is less than or equal to threshold
        # (suppressing boxes with high overlap)
        inds = np.where(iou <= iou_threshold)[0]
        
        # Update order (adding 1 because iou calculation was done on order[1:])
        order = order[inds + 1]

    return [boxes[k] for k in keep]


def sliding_window_detection(image, model, confidence_threshold=0.5, stride=32, iou_threshold=0.3):
    """
    Performs sliding-window object detection over a full satellite image.
    
    Args:
        image: PIL Image object (any resolution).
        model: Loaded Keras model.
        confidence_threshold: Float, minimum confidence to keep a window (default 0.5).
        stride: Int, step size in pixels for sliding the window (default 32).
        iou_threshold: Float, threshold for NMS overlap filtering (default 0.3).
        
    Returns:
        List of final bounding boxes [x1, y1, x2, y2, confidence] in original coordinates.
    """
    # Convert image to RGB numpy array
    img_array = np.array(image.convert("RGB"))
    H, W, C = img_array.shape

    # Handle images smaller than the 64x64 window size by padding or skipping
    if H < 64 or W < 64:
        # Pad the image to at least 64x64 using constant zero padding
        pad_y = max(0, 64 - H)
        pad_x = max(0, 64 - W)
        img_array = np.pad(img_array, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')
        H, W, C = img_array.shape

    patches = []
    coords = []

    # Generate sliding window coordinates, ensuring we cover the boundaries
    y_indices = list(range(0, H - 64 + 1, stride))
    if not y_indices or y_indices[-1] != H - 64:
        y_indices.append(H - 64)

    x_indices = list(range(0, W - 64 + 1, stride))
    if not x_indices or x_indices[-1] != W - 64:
        x_indices.append(W - 64)

    # Extract all patches
    for y in y_indices:
        for x in x_indices:
            patch = img_array[y:y+64, x:x+64]
            if patch.shape == (64, 64, 3):
                patches.append(patch)
                coords.append((x, y))

    if not patches:
        return []

    # Prepare patches for prediction: convert to float and normalize (divide by 255.0)
    patches_arr = np.array(patches, dtype=np.float32) / 255.0

    # Run batch prediction to maximize throughput and avoid CPU/GPU overhead
    predictions = model.predict(patches_arr, batch_size=64, verbose=0)

    # Convert predictions to NumPy array if they are tensors (depends on active backend)
    if hasattr(predictions, "numpy"):
        predictions = predictions.numpy()
    elif hasattr(predictions, "cpu"):
        predictions = predictions.cpu().numpy()
    predictions = np.array(predictions)

    # Extract confidence scores
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Categorical output: index 1 represents 'contains tank'
        confidences = predictions[:, 1]
    else:
        # Binary output: scalar probability represents 'contains tank'
        confidences = predictions.flatten()

    # Filter by confidence threshold
    raw_boxes = []
    for i, conf in enumerate(confidences):
        if conf > confidence_threshold:
            x, y = coords[i]
            # [x1, y1, x2, y2, confidence]
            raw_boxes.append([
                float(x), 
                float(y), 
                float(x + 64), 
                float(y + 64), 
                float(conf)
            ])

    # Apply Non-Maximum Suppression to remove redundant overlaps
    final_boxes = non_max_suppression(raw_boxes, iou_threshold=iou_threshold)
    
    return final_boxes
