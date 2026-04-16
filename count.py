import cv2
import os
import numpy as np
import csv
from generate_trace import detect_blue_markers
from collections import defaultdict


def enhance_image_intensity(image):
    """Enhance the intensity of the image for better visualization."""
    return np.clip(image * 1.5, 0, 255).astype(np.uint8)


def calculate_shape_features(contours):
    """Calculate shape-related features (area, circularity, aspect ratio)"""
    if not contours or len(contours) == 0:
        return 0, 0, 0
        
    area = cv2.contourArea(contours[0])
    perimeter = cv2.arcLength(contours[0], True)
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 0
    x, y, w, h = cv2.boundingRect(contours[0])
    aspect_ratio = float(w) / h if h != 0 else 0
    return area, circularity, aspect_ratio


def count_cells_in_dataset(test_path, track_result_path, trace_path=None):
    """
    Count iPS and normal cells based on tracking data from res_track.txt.
    Handles cell divisions and ensures consistent labeling of iPS cells.
    
    Args:
        test_path: Path to original test images
        track_result_path: Path to tracking results including res_track.txt
        trace_path: Optional path to trace results
        
    Returns:
        tuple: (frame_results, missing_cells, overall_counts) with detailed cell counts
    """
    try:
        test_files = sorted([f for f in os.listdir(test_path) if f.endswith('.tif') or f.endswith('.tif')])
        track_files = sorted([f for f in os.listdir(track_result_path) if f.endswith('.tif') or f.endswith('.tif')])
        
        if not test_files or not track_files:
            print("No image files found. Please check the paths.")
            return [], [], []
    except FileNotFoundError:
        print(f"Path not found: {test_path} or {track_result_path}")
        return [], [], []
    except Exception as e:
        print(f"Error accessing directories: {e}")
        return [], [], []

    # Load tracking data from res_track.txt - This is our primary source of truth
    tracking_data = {}
    parent_to_children = {}  # Dictionary to map parent cells to their children
    division_frames = {}     # Dictionary to record when divisions occur
    
    try:
        tracking_file = os.path.join(track_result_path, 'res_track.txt')
        if not os.path.exists(tracking_file):
            print(f"Tracking data file {tracking_file} not found. Skipping cell counting.")
            return [], [], []  # Return empty lists if tracking data is missing

        with open(tracking_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    print(f"Warning: Malformed line in tracking file: {line}")
                    continue  # Skip malformed lines
                    
                try:
                    cell_id = int(parts[0])
                    start_frame = int(parts[1])
                    end_frame = int(parts[2])
                    parent_id = int(parts[3])
                    
                    tracking_data[cell_id] = {
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'parent_id': parent_id,
                        'is_ips': False,  # Will be determined later
                        'is_missing': False,  # Will be determined later
                        'children': []  # Will be populated later
                    }
                    
                    # Build parent-child relationships
                    if parent_id != 0:
                        if parent_id not in parent_to_children:
                            parent_to_children[parent_id] = []
                        parent_to_children[parent_id].append(cell_id)
                        
                        # Record division frame (when this child appears)
                        if parent_id not in division_frames:
                            division_frames[parent_id] = {}
                        division_frames[parent_id][cell_id] = start_frame
                except (ValueError, IndexError) as e:
                    print(f"Error parsing tracking line: {line}. Error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Tracking data file {tracking_file} not found. Skipping cell counting.")
        return [], [], []  # Return empty lists if tracking data is missing
    except Exception as e:
        print(f"Error reading tracking data: {e}. Skipping cell counting.")
        return [], [], []

    # Update tracking data with children information
    for parent_id, children in parent_to_children.items():
        if parent_id in tracking_data:
            tracking_data[parent_id]['children'] = children

    # Check for missing cells by identifying frames where cells should be present but aren't
    missing_cells = set()
    frame_cell_presence = {}  # Track which cells are present in each frame
    
    # First, determine which cells are present in each frame
    for frame_idx in range(len(track_files)):
        track_img_path = os.path.join(track_result_path, track_files[frame_idx])
        track_image = cv2.imread(track_img_path, -1)
        
        if track_image is not None:
            present_cells = set(np.unique(track_image))
            if 0 in present_cells:
                present_cells.remove(0)  # Remove background
            frame_cell_presence[frame_idx] = present_cells
        else:
            frame_cell_presence[frame_idx] = set()
    
    # Now identify missing cells
    for cell_id, data in tracking_data.items():
        start_frame = data['start_frame']
        end_frame = data['end_frame']
        
        for frame_idx in range(start_frame, end_frame + 1):
            # If frame is within our dataset and cell should be present but isn't
            if frame_idx in frame_cell_presence and cell_id not in frame_cell_presence[frame_idx]:
                tracking_data[cell_id]['is_missing'] = True
                missing_cells.add(cell_id)
                break

    # Detect iPS cells (blue markers)
    for frame_idx in range(len(test_files)):
        if frame_idx >= len(track_files):
            continue
            
        test_img_path = os.path.join(test_path, test_files[frame_idx])
        track_img_path = os.path.join(track_result_path, track_files[frame_idx])
        
        test_image = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
        track_image = cv2.imread(track_img_path, -1)
        
        if test_image is None or track_image is None:
            continue
            
        # Detect blue markers
        blue_mask = detect_blue_markers(test_image)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each cell in this frame
        present_cells = frame_cell_presence.get(frame_idx, set())
        for cell_id in present_cells:
            if cell_id not in tracking_data:
                continue
                
            # Get cell mask and centroid
            cell_mask = (track_image == cell_id).astype(np.uint8)
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            M = cv2.moments(contours[0])
            if M['m00'] == 0:
                continue
                
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Check if cell is near a blue marker (iPS cell)
            is_blue = False
            for contour in blue_contours:
                for point in contour:
                    distance = np.sqrt((point[0][0] - cx) ** 2 + (point[0][1] - cy) ** 2)
                    if distance < 20:  # Proximity threshold
                        is_blue = True
                        break
                if is_blue:
                    break
            
            # If this cell is an iPS cell, mark it
            if is_blue:
                tracking_data[cell_id]['is_ips'] = True

    # Prepare frame-by-frame results
    frame_results = []
    missing_cells_tracker = []  # To track missing cells by frame
    
    for frame_idx in range(len(test_files)):
        frame_counts = {
            "Frame": test_files[frame_idx] if frame_idx < len(test_files) else f"Frame_{frame_idx}",
            "iPS Count": 0,
            "Normal Count": 0,
            "Divided iPS": 0,
            "Divided Normal": 0,
            "Missing iPS": 0,
            "Missing Normal": 0,
            "Total Cells": 0
        }
        
        missing_cells_in_frame = {
            "Frame": test_files[frame_idx] if frame_idx < len(test_files) else f"Frame_{frame_idx}",
            "Missing Cell IDs": []
        }
        
        # Process cells that should be in this frame according to tracking data
        for cell_id, data in tracking_data.items():
            if data['start_frame'] <= frame_idx <= data['end_frame']:
                # Check if cell is actually present in this frame
                is_present = frame_idx in frame_cell_presence and cell_id in frame_cell_presence[frame_idx]
                
                # If not present, count as missing
                if not is_present:
                    if data['is_ips']:
                        frame_counts["Missing iPS"] += 1
                    else:
                        frame_counts["Missing Normal"] += 1
                    
                    missing_cells_in_frame["Missing Cell IDs"].append(cell_id)
                    continue
                
                # Count present cells
                is_divided = len(data['children']) > 0
                is_child = data['parent_id'] != 0
                
                if is_child:  # This is a daughter cell
                    if data['is_ips']:
                        frame_counts["Divided iPS"] += 1
                    else:
                        frame_counts["Divided Normal"] += 1
                else:  # This is a root/parent cell
                    if data['is_ips']:
                        frame_counts["iPS Count"] += 1
                    else:
                        frame_counts["Normal Count"] += 1
        
        # Calculate total cells for this frame
        frame_counts["Total Cells"] = (
            frame_counts["iPS Count"] + frame_counts["Normal Count"] +
            frame_counts["Divided iPS"] + frame_counts["Divided Normal"] +
            frame_counts["Missing iPS"] + frame_counts["Missing Normal"]
        )
        
        frame_results.append(frame_counts)
        missing_cells_in_frame["Missing Cell IDs"] = ",".join(map(str, missing_cells_in_frame["Missing Cell IDs"]))
        missing_cells_tracker.append(missing_cells_in_frame)

    # Prepare overall dataset counts based on unique cell IDs
    # Count each cell only once
    cell_counts = {
        'iPS': 0, 
        'normal': 0, 
        'divided_iPS': 0, 
        'divided_normal': 0,
        'parent_iPS': 0,
        'parent_normal': 0,
        'missing_iPS': 0,
        'missing_normal': 0
    }
    
    for cell_id, data in tracking_data.items():
        is_parent = cell_id in parent_to_children
        is_child = data['parent_id'] != 0
        is_missing = data['is_missing']
        
        if is_child:  # Daughter cell
            if data['is_ips']:
                cell_counts['divided_iPS'] += 1
                if is_missing:
                    cell_counts['missing_iPS'] += 1
            else:
                cell_counts['divided_normal'] += 1
                if is_missing:
                    cell_counts['missing_normal'] += 1
        else:  # Parent/root cell
            if data['is_ips']:
                cell_counts['iPS'] += 1
                if is_parent:
                    cell_counts['parent_iPS'] += 1
                if is_missing:
                    cell_counts['missing_iPS'] += 1
            else:
                cell_counts['normal'] += 1
                if is_parent:
                    cell_counts['parent_normal'] += 1
                if is_missing:
                    cell_counts['missing_normal'] += 1
    
    overall_dataset_counts = {
        "iPS Count": cell_counts['iPS'],
        "Normal Count": cell_counts['normal'],
        "Divided iPS": cell_counts['divided_iPS'],
        "Divided Normal": cell_counts['divided_normal'],
        "Parent iPS": cell_counts['parent_iPS'],
        "Parent Normal": cell_counts['parent_normal'],
        "Missing iPS": cell_counts['missing_iPS'],
        "Missing Normal": cell_counts['missing_normal'],
        "Total Cells": (
            cell_counts['iPS'] + cell_counts['normal'] +
            cell_counts['divided_iPS'] + cell_counts['divided_normal']
        )
    }

    # Save detailed cell lineage information
    try:
        lineage_path = os.path.join(track_result_path, 'detailed_lineage.csv')
        with open(lineage_path, 'w', newline='') as f:
            fieldnames = [
                "Cell ID", "Start Frame", "End Frame", "Parent ID", 
                "Is iPS", "Has Children", "Is Missing", "Children IDs"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for cell_id, data in tracking_data.items():
                entry = {
                    "Cell ID": cell_id,
                    "Start Frame": data['start_frame'],
                    "End Frame": data['end_frame'],
                    "Parent ID": data['parent_id'],
                    "Is iPS": "Yes" if data['is_ips'] else "No",
                    "Has Children": "Yes" if data['children'] else "No",
                    "Is Missing": "Yes" if data['is_missing'] else "No",
                    "Children IDs": ",".join(map(str, data['children'])) if data['children'] else "None"
                }
                writer.writerow(entry)
                
        print(f"Detailed lineage information saved to {lineage_path}")
    except Exception as e:
        print(f"Error saving detailed lineage information: {e}")
    
    # Save missing cells information to be used by features.py
    try:
        missing_cells_path = os.path.join(track_result_path, 'missing_cells.csv')
        with open(missing_cells_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Cell ID", "Is Missing"])
            for cell_id, data in tracking_data.items():
                writer.writerow([cell_id, "Yes" if data['is_missing'] else "No"])
        print(f"Missing cells information saved to {missing_cells_path}")
    except Exception as e:
        print(f"Error saving missing cells information: {e}")

    return frame_results, missing_cells_tracker, [overall_dataset_counts]


def save_results_to_csv(results, missing_cells_tracker, overall_counts, output_csv):
    """
    Save the counting results to a CSV file.
    Also saves a separate missing_cells.csv file with frame-by-frame missing cell IDs.
    """
    try:
        # Save main results
        with open(output_csv, mode='w', newline='') as file:
            fieldnames = [
                "Frame", "iPS Count", "Normal Count", "Divided iPS", "Divided Normal", 
                "Missing iPS", "Missing Normal", "Total Cells"
            ]
            
            # Add Parent iPS/Normal if in overall_counts
            if overall_counts and len(overall_counts) > 0 and "Parent iPS" in overall_counts[0]:
                fieldnames.extend(["Parent iPS", "Parent Normal"])
                
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write frame-by-frame results if available
            if results:
                writer.writerows(results)
            
            # Only write overall counts if they exist and are not empty
            if overall_counts and len(overall_counts) > 0:
                # Create a new row for overall counts
                overall_row = {
                    "Frame": "Overall",
                    "iPS Count": overall_counts[0]['iPS Count'],
                    "Normal Count": overall_counts[0]['Normal Count'],
                    "Divided iPS": overall_counts[0]['Divided iPS'],
                    "Divided Normal": overall_counts[0]['Divided Normal'],
                    "Missing iPS": overall_counts[0].get('Missing iPS', 0),
                    "Missing Normal": overall_counts[0].get('Missing Normal', 0),
                    "Total Cells": overall_counts[0]['Total Cells']
                }
                
                # Add Parent iPS/Normal if available
                if "Parent iPS" in overall_counts[0]:
                    overall_row["Parent iPS"] = overall_counts[0]['Parent iPS']
                    overall_row["Parent Normal"] = overall_counts[0]['Parent Normal']
                    
                writer.writerow(overall_row)
            else:
                print("Warning: No overall counts to save. Skipping overall summary.")

        print(f"Counting results saved to {output_csv}")
        
        # Save missing cells tracker to separate file
        if missing_cells_tracker:
            missing_cells_csv = os.path.splitext(output_csv)[0] + "_missing_cells.csv"
            with open(missing_cells_csv, mode='w', newline='') as file:
                fieldnames = ["Frame", "Missing Cell IDs"]
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(missing_cells_tracker)
            print(f"Missing cells tracker saved to {missing_cells_csv}")
            
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        # Create an empty file to prevent further errors
        with open(output_csv, 'w') as f:
            f.write("Error occurred, no valid data\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Count cells and track divisions in an image sequence")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test images")
    parser.add_argument("--track_result_path", type=str, required=True, help="Path to tracking results")
    parser.add_argument("--trace_path", type=str, help="Path to trace results (optional)")
    parser.add_argument("--output_csv", type=str, required=True, help="Path for output CSV file")
    
    args = parser.parse_args()
    
    trace_path = args.trace_path if hasattr(args, 'trace_path') and args.trace_path else None
    results, missing_cells_tracker, overall_counts = count_cells_in_dataset(
        args.test_path, args.track_result_path, trace_path
    )
    save_results_to_csv(results, missing_cells_tracker, overall_counts, args.output_csv)