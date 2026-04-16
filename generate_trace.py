import cv2
import os
import numpy as np
import random

# Function to compute centroids of each labeled cell in the segmented image
def get_center(serial, label, directory):
    track_picture = sorted([file for file in os.listdir(directory) if ".tif" in file or ".tif" in file])
    result_picture = cv2.imread(os.path.join(directory, track_picture[serial]), -1)
    if result_picture is None:
        print(f"Error reading {track_picture[serial]} from {directory}")
        return None

    label_picture = ((result_picture == label) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(label_picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        M = cv2.moments(contours[0])
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    return None

# Function to detect blue markers in an image
def detect_blue_markers(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask

# Function to get a colored mask for each contour
def get_coloured_mask(mask):
    colours = [
        [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255],
        [255, 128, 0], [128, 255, 0], [0, 255, 128], [0, 128, 255], [128, 0, 255], [255, 0, 128],
        [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190], [0, 128, 0], [255, 165, 0]
    ]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    idx = random.randrange(0, len(colours))
    r[mask == 255], g[mask == 255], b[mask == 255] = colours[idx]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

# Function to generate traces and division detection
def get_trace(image_path, track_path, trace_path):
    track_picture = sorted([file for file in os.listdir(track_path) if ".tif" in file or ".tif" in file])
    test_image = sorted([file for file in os.listdir(image_path) if ".tif" in file or ".tif" in file])
    trace_image = []

    # Load parent-child relationship data
    file = os.path.join(track_path, "res_track.txt")
    with open(file, "r") as f:
        tracking_data = f.readlines()

    # Enhanced data structure for tracking
    tracking_info = {}
    for line in tracking_data:
        parts = line.strip().split()
        cell_id = int(parts[0])
        start_frame = int(parts[1])
        end_frame = int(parts[2])
        parent_id = int(parts[3])
        
        tracking_info[cell_id] = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'parent_id': parent_id
        }

    for i, img_name in enumerate(test_image):
        img_path = os.path.join(image_path, img_name)
        original = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if original is None:
            print(f"Error reading image {img_path}. Skipping.")
            continue
            
        if original.shape[0] >= 741 and original.shape[1] >= 769:
            original = original[5:741, 1:769]

        track_img_path = os.path.join(track_path, track_picture[i]) if i < len(track_picture) else None
        if track_img_path is None:
            print(f"No tracking data for frame {i}. Skipping.")
            continue
            
        result_picture = cv2.imread(track_img_path, -1)
        if result_picture is None:
            print(f"Error reading track image {track_img_path}. Skipping.")
            continue
            
        label_picture = ((result_picture >= 1) * 255).astype(np.uint8)

        image_to_draw = original.copy()
        
        # Detect blue markers (iPS cells)
        blue_mask = detect_blue_markers(original)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours, _ = cv2.findContours(label_picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_to_draw, (x, y), (x + w, y + h), (255, 255, 255), 1)

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            if cy < result_picture.shape[0] and cx < result_picture.shape[1]:
                cell_id = result_picture[cy, cx]
            else:
                continue

            is_tracked = cell_id in tracking_info and tracking_info[cell_id]['start_frame'] <= i <= tracking_info[cell_id]['end_frame']
            is_blue = any(cv2.pointPolygonTest(blue_contour, (cx, cy), False) >= 0 for blue_contour in blue_contours)
            label_prefix = "iPS_" if is_blue else ""
            
            if is_tracked:
                parent_id = tracking_info[cell_id]['parent_id']
                if parent_id != 0:
                    label = f"{label_prefix}{cell_id}({parent_id})"  # Simplified division display
                else:
                    label = f"{label_prefix}{cell_id}"
            else:
                label = f"{label_prefix}{cell_id}"

            cv2.putText(image_to_draw, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            mask = np.zeros_like(image_to_draw[:, :, 0])
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            colored_mask = get_coloured_mask(mask)
            image_to_draw = cv2.addWeighted(image_to_draw, 1, colored_mask, 0.5, 0)

        cv2.putText(original, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        combined_image = np.hstack((original, image_to_draw))
        trace_image.append(combined_image)

    # Add trajectory lines
    for cell_id, info in tracking_info.items():
        start_frame = info['start_frame']
        end_frame = info['end_frame']
        
        if start_frame < end_frame:
            cell_centers = []
            cell_combined_centers = []
            width_offset = trace_image[start_frame].shape[1] // 2
            
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx >= len(trace_image):
                    break
                
                center = get_center(frame_idx, cell_id, track_path)
                if center is None:
                    if cell_centers:
                        center = cell_centers[-1]
                    else:
                        continue
                
                cell_centers.append(center)
                combined_center = (center[0] + width_offset, center[1])
                cell_combined_centers.append(combined_center)
                cv2.circle(trace_image[frame_idx], combined_center, 3, (0, 0, 255), -1)
            
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx >= len(trace_image):
                    break
                
                center_idx = frame_idx - start_frame
                if center_idx < 0 or center_idx >= len(cell_combined_centers):
                    continue
                
                for j in range(1, center_idx + 1):
                    cv2.line(
                        trace_image[frame_idx],
                        cell_combined_centers[j-1],
                        cell_combined_centers[j],
                        (0, 0, 255),
                        1
                    )

    os.makedirs(trace_path, exist_ok=True)
    for i, img in enumerate(trace_image):
        cv2.imwrite(os.path.join(trace_path, f"{i:06d}.tif"), img)

    print(f"Generated {len(trace_image)} trace images with trajectory lines. Saved to {trace_path}")

# Function to create video from traced images
def get_video(trace_path):
    pictures = sorted([name for name in os.listdir(trace_path) if name.endswith(".tif") or name.endswith(".tif")])
    if not pictures:
        print("No images found to generate video.")
        return
    
    first_img_path = os.path.join(trace_path, pictures[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"Error reading first image {first_img_path}. Cannot create video.")
        return
    
    size = first_img.shape[1::-1]
    
    video_path = os.path.join(trace_path, "trace.mp4")
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 1, size)

    for picture in pictures:
        frame_path = os.path.join(trace_path, picture)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Error reading frame {frame_path}. Skipping.")
            continue
        video.write(frame)
    
    video.release()
    print(f"Video generation completed. Saved to {video_path}")

# Main execution
if __name__ == "__main__":
    test_folders = [os.path.join("nuclear_dataset", folder) for folder in sorted(os.listdir("nuclear_dataset"))]
    for folder in test_folders:
        test_path = os.path.join(folder, "test")
        track_result_path = os.path.join(folder, "track_result")
        trace_path = os.path.join(folder, "trace")
        
        os.makedirs(trace_path, exist_ok=True)
        
        print(f"Processing {folder}...")
        get_trace(test_path, track_result_path, trace_path)
        get_video(trace_path)

    print("Processing completed.")
