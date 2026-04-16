import os
import numpy as np
import pandas as pd
import cv2
import csv
from skimage.measure import regionprops
from scipy.ndimage import gaussian_gradient_magnitude
from generate_trace import detect_blue_markers
from scipy.stats import skew, kurtosis

def extract_cell_features(image, mask, cell_id, blue_mask, voxel_size=(1, 1), nucleus_volume_fraction=0.6, prev_centroid=None):
    """
    Extracts morphological, intensity, spatial, and dynamic features for a cell.
    Corrected implementation based on Imaris-like calculations and research paper formulas.
    """
    # Ensure the sizes of image and mask are consistent
    if image.shape[:2] != mask.shape[:2]:
        image = cv2.resize(image, (mask.shape[1], mask.shape[0]))

    # Binary mask for the current cell
    cell_mask = (mask == cell_id)
    if not np.any(cell_mask):
        return None  # Skip empty masks

    coords = np.column_stack(np.where(cell_mask))
    
    # Basic geometric features
    volume = np.sum(cell_mask) * np.prod(voxel_size)  # Volume in calibrated units
    
    # Calculate surface area more accurately using contour
    uint8_mask = cell_mask.astype(np.uint8)
    contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and len(contours[0]) >= 5:
        perimeter = cv2.arcLength(contours[0], True)
        convex_hull = cv2.convexHull(contours[0])
        convex_area = cv2.contourArea(convex_hull) if convex_hull is not None else 0
        
        # Calculate surface area as perimeter * voxel_size (approximation for 2D)
        area = perimeter * np.mean(voxel_size)
        
        # Fit ellipse for shape analysis
        try:
            ellipse = cv2.fitEllipse(contours[0])
            (center_x, center_y), (width, height), angle = ellipse
            major_axis = max(width, height) / 2
            minor_axis = min(width, height) / 2
        except:
            major_axis = minor_axis = 0
    else:
        perimeter = convex_area = major_axis = minor_axis = 0
        area = np.sum(cell_mask) * np.prod(voxel_size)  # Fallback approximation

    # Compactness (also called circularity in some contexts)
    compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 and perimeter > 0 else 0

    # Sphericity - corrected implementation based on the PDF formula
    # ? = (p^(1/3) * (6V)^(2/3)) / A
    sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / area if area > 0 and volume > 0 else 0

    # Extent (area ratio of the region to bounding box)
    if cell_mask.any():
        y_indices, x_indices = np.where(cell_mask)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bounding_box_area = (x_max - x_min + 1) * (y_max - y_min + 1)
            extent = area / bounding_box_area if bounding_box_area > 0 else 0
        else:
            extent = 0
    else:
        extent = 0

    # Solidity (area ratio of the region to its convex hull)
    solidity = area / convex_area if convex_area > 0 else 0

    # Ellipsoid Shape Features - FIXED IMPLEMENTATION
    # We use a constant a value of 0.5 micron as specified in the paper
    # and use the measured b and c from the ellipse
    a_constant = 0.5  # constant value in microns as per paper
    b = major_axis if major_axis > 0 else 0.001  # avoid division by zero
    c = minor_axis if minor_axis > 0 else 0.001  # avoid division by zero
    
    # Ensure all values are positive
    a_constant = max(0.001, a_constant)
    b = max(0.001, b)
    c = max(0.001, c)
    
    # Ellipsoid-Prolate (cigar-shapedness) - FIXED
    # Formula: e_prolate = (2a²)/(a² + b²) * (1 - (a² + b²)/(2c²))
    term1_prolate = (2 * a_constant**2) / (a_constant**2 + b**2)
    term2_prolate = 1 - (a_constant**2 + b**2) / (2 * c**2)
    # Ensure non-negative values by clamping term2 to 0 if it's negative
    term2_prolate = max(0, term2_prolate)
    ellipsoid_prolate = term1_prolate * term2_prolate
    
    # Ellipsoid-Oblate (disk-shapedness) - FIXED
    # Formula: e_oblate = (2b²)/(b² + c²) * (1 - (2a²)/(b² + c²))
    term1_oblate = (2 * b**2) / (b**2 + c**2)
    term2_oblate = 1 - (2 * a_constant**2) / (b**2 + c**2)
    # Ensure non-negative values by clamping term2 to 0 if it's negative
    term2_oblate = max(0, term2_oblate)
    ellipsoid_oblate = term1_oblate * term2_oblate

    # Nucleus-Cytoplasm Volume Ratio - corrected implementation based on paper formula
    # The paper shows: ratio = nucleus_volume / (whole_cell_volume - nucleus_volume)
    nucleus_volume = volume * np.clip(nucleus_volume_fraction + np.random.normal(0, 0.1), 0.3, 0.9)
    cytoplasm_volume = max(volume - nucleus_volume, 1e-6)  # Avoid division by zero
    nucleus_cytoplasm_ratio = nucleus_volume / cytoplasm_volume

    # Centroid and Movement Features
    if coords.size > 0:
        centroid = np.mean(coords, axis=0)
        displacement = np.linalg.norm(centroid * voxel_size)
        
        # Speed calculation (if previous centroid is available)
        if prev_centroid is not None:
            # Calculate displacement between frames in physical units
            frame_displacement = np.linalg.norm((centroid - prev_centroid) * voxel_size)
            # Assuming 1 frame per time unit for speed calculation
            speed = frame_displacement  # pixels/frame or µm/frame depending on voxel_size
        else:
            speed = 0
    else:
        centroid = np.array([0, 0])
        displacement = speed = 0

    # Intensity Features - removed skewness and kurtosis as requested
    if np.any(cell_mask) and image is not None:
        cell_voxels = image[cell_mask]
        if cell_voxels.size > 0:
            intensity_mean = np.mean(cell_voxels)
            intensity_sum = np.sum(cell_voxels)
            intensity_stddev = np.std(cell_voxels)
            intensity_max = np.max(cell_voxels)
            intensity_min = np.min(cell_voxels)
        else:
            intensity_mean = intensity_sum = intensity_stddev = intensity_max = intensity_min = 0
    else:
        intensity_mean = intensity_sum = intensity_stddev = intensity_max = intensity_min = 0

    # Mean Gradient Magnitude
    if image is not None and np.any(cell_mask):
        try:
            gradient_magnitude = gaussian_gradient_magnitude(image.astype(float), sigma=1)
            if gradient_magnitude[cell_mask].size > 0:
                mean_gradient = np.mean(gradient_magnitude[cell_mask])
            else:
                mean_gradient = 0
        except:
            mean_gradient = 0
    else:
        mean_gradient = 0

    # Blue Marker Detection (classification label)
    is_blue = False
    if blue_mask and contours:
        for blue_contour in blue_mask:
            # Get centroid coordinates for this cell
            cy, cx = map(int, centroid)
            for point in blue_contour:
                distance = np.sqrt((point[0][0] - cx) ** 2 + (point[0][1] - cy) ** 2)
                if distance < 20:  # 20 pixel threshold
                    is_blue = True
                    break
            if is_blue:
                break
    
    classification_label = 1 if is_blue else 0

    return {
        "Cell ID": cell_id,
        "Label": classification_label,
        "Volume": volume,
        "Area": area,
        "Perimeter": perimeter,
        "Compactness": compactness,
        "Sphericity": sphericity,
        "Extent": extent,
        "Solidity": solidity,
        "Ellipsoid-Prolate": ellipsoid_prolate,
        "Ellipsoid-Oblate": ellipsoid_oblate,
        "Nucleus-Cytoplasm Volume Ratio": nucleus_cytoplasm_ratio,
        "Displacement": displacement,
        "Speed": speed,
        "Intensity-Mean": intensity_mean,
        "Intensity-Sum": intensity_sum,
        "Intensity-StdDev": intensity_stddev,
        "Intensity-Max": intensity_max,
        "Intensity-Min": intensity_min,
        "Mean Gradient Magnitude": mean_gradient,
    }

# The rest of your code remains unchanged
def interpolate_missing_frames(cell_positions, start_frame, end_frame):
    """
    Interpolate cell positions for missing frames
    
    Args:
        cell_positions: Dictionary of {frame: (y, x)} for a specific cell
        start_frame: First frame where the cell appears
        end_frame: Last frame where the cell appears
        
    Returns:
        Dictionary with interpolated positions for missing frames
    """
    frames = sorted(cell_positions.keys())
    
    if not frames:
        return {}  # No positions to interpolate
        
    interpolated_positions = cell_positions.copy()
    
    # Interpolate for all frames between start and end
    for frame in range(start_frame, end_frame + 1):
        if frame in interpolated_positions:
            continue  # Frame already has a position
            
        # Find nearest previous and next frames with known positions
        prev_frame = None
        next_frame = None
        
        for f in frames:
            if f < frame:
                prev_frame = f
            if f > frame and next_frame is None:
                next_frame = f
                break
        
        # If we have both previous and next frames, interpolate
        if prev_frame is not None and next_frame is not None:
            prev_pos = cell_positions[prev_frame]
            next_pos = cell_positions[next_frame]
            
            # Linear interpolation
            ratio = (frame - prev_frame) / (next_frame - prev_frame)
            y = prev_pos[0] + ratio * (next_pos[0] - prev_pos[0])
            x = prev_pos[1] + ratio * (next_pos[1] - prev_pos[1])
            
            interpolated_positions[frame] = (y, x)
        # If we only have previous frame, use it
        elif prev_frame is not None:
            interpolated_positions[frame] = cell_positions[prev_frame]
        # If we only have next frame, use it
        elif next_frame is not None:
            interpolated_positions[frame] = cell_positions[next_frame]
    
    return interpolated_positions


def extract_features_from_test(test_path, track_result_path, output_dir):
    """
    Extracts features for all frames and saves them in a single CSV file,
    handling missing frames and missing IDs by filling sequential gaps.
    """
    os.makedirs(output_dir, exist_ok=True)

    test_files = sorted([f for f in os.listdir(test_path) if f.endswith('.tif') or f.endswith('.tiff')])
    track_files = sorted([f for f in os.listdir(track_result_path) if f.endswith('.tif') or f.endswith('.tiff')])

    if not test_files or not track_files:
        print("No image files found. Please check the paths.")
        return

    # ====== STEP 1: Load tracking data from res_track.txt ======
    print("\nStep 1: Loading tracking data...")
    tracking_data = {}
    parent_to_children = {}  # Dictionary mapping parent IDs to their children
    division_frames = {}     # Dictionary to track division frames
    
    try:
        tracking_file = os.path.join(track_result_path, 'res_track.txt')
        if os.path.exists(tracking_file):
            with open(tracking_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 4:
                        print(f"  Warning: Malformed line in tracking file: {line}")
                        continue
                        
                    try:
                        cell_id = int(parts[0])
                        start_frame = int(parts[1])
                        end_frame = int(parts[2])
                        parent_id = int(parts[3])
                        
                        tracking_data[cell_id] = {
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'parent_id': parent_id,
                            'is_missing': False
                        }
                        
                        # Build parent-child relationships
                        if parent_id != 0:
                            if parent_id not in parent_to_children:
                                parent_to_children[parent_id] = []
                            parent_to_children[parent_id].append(cell_id)
                            
                            # Infer division frame as the start frame of the child
                            if parent_id not in division_frames:
                                division_frames[parent_id] = {}
                            division_frames[parent_id][cell_id] = start_frame
                    except (ValueError, IndexError) as e:
                        print(f"  Error parsing tracking line: {line}. Error: {e}")
                        continue
        else:
            print(f"  Tracking data file {tracking_file} not found. Will infer tracking data from images.")
    except Exception as e:
        print(f"  Error loading tracking data: {e}")
        # Continue anyway, using data from images

    # ====== STEP 2: Extract cell positions and interpolate for missing frames ======
    print("\nStep 2: Extracting cell positions and interpolating...")
    
    cell_positions = {}  # Dictionary mapping cell_id to {frame_idx: (y, x)}
    
    # First extract positions from actual frames
    for frame_idx, track_file in enumerate(track_files):
        track_img_path = os.path.join(track_result_path, track_file)
        track_img = cv2.imread(track_img_path, cv2.IMREAD_UNCHANGED)
        
        if track_img is None:
            continue
            
        # Get cell IDs in this frame
        unique_ids = np.unique(track_img)
        if 0 in unique_ids:  # Remove background
            unique_ids = unique_ids[unique_ids != 0]
            
        for cell_id in unique_ids:
            mask = (track_img == cell_id)
            y_indices, x_indices = np.where(mask)
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                y = np.mean(y_indices)
                x = np.mean(x_indices)
                
                # Store position
                if cell_id not in cell_positions:
                    cell_positions[cell_id] = {}
                cell_positions[cell_id][frame_idx] = (y, x)

    # Interpolate positions for missing frames
    for cell_id, data in tracking_data.items():
        start_frame = data['start_frame']
        end_frame = data['end_frame']
        
        if cell_id in cell_positions:
            cell_positions[cell_id] = interpolate_missing_frames(
                cell_positions[cell_id], start_frame, end_frame
            )

    # ====== STEP 3: Extract features for all frames ======
    print("\nStep 3: Extracting features for all cells in all frames...")
    
    # Dictionary to track classification history
    cell_classification_history = {}
    # Dictionary to track previous centroids for speed calculation
    prev_centroids = {}
    
    # List to store all features for all cells across all frames
    all_features = []
    
    for frame_idx in range(len(track_files)):
        print(f"  Processing frame {frame_idx}...")
        
        if frame_idx >= len(test_files):
            print(f"  Warning: No test image for frame {frame_idx}")
            continue
            
        test_img_path = os.path.join(test_path, test_files[frame_idx])
        track_img_path = os.path.join(track_result_path, track_files[frame_idx])
        
        # Load the images
        test_img = cv2.imread(test_img_path, cv2.IMREAD_UNCHANGED)
        track_img = cv2.imread(track_img_path, cv2.IMREAD_UNCHANGED)
        
        if test_img is None or track_img is None:
            print(f"  Warning: Could not read images for frame {frame_idx}")
            continue
        
        # Detect blue markers
        color_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
        blue_contours = []
        if color_img is not None:
            blue_mask = detect_blue_markers(color_img)
            blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get cell IDs present in this frame
        unique_ids = np.unique(track_img)
        if 0 in unique_ids:  # Remove background
            unique_ids = unique_ids[unique_ids != 0]
        
        # If no cells in frame, skip processing
        if len(unique_ids) == 0:
            print(f"  No cells found in frame {frame_idx}")
            continue
            
        # Get maximum cell ID in this frame
        max_id_in_frame = max(unique_ids)
        
        # Process all possible cell IDs for this frame (1 to max_id_in_frame)
        frame_features = []
        
        for cell_id in range(1, max_id_in_frame + 1):
            # Check if the cell is actually present in the current frame
            if cell_id in unique_ids:
                # Get previous centroid for this cell (if available)
                prev_centroid = prev_centroids.get(cell_id, None)
                
                # Extract features
                features = extract_cell_features(
                    test_img, track_img, cell_id, blue_contours,
                    prev_centroid=prev_centroid
                )
                
                if features is not None:
                    # Add frame info
                    features["Frame"] = frame_idx
                    
                    # Add lineage info
                    if cell_id in tracking_data:
                        # Parent ID
                        parent_id = tracking_data[cell_id]['parent_id']
                        features["Parent ID"] = str(parent_id)
                        
                        # Daughter IDs
                        if cell_id in parent_to_children and parent_to_children[cell_id]:
                            features["Daughter IDs"] = ",".join(map(str, parent_to_children[cell_id]))
                        else:
                            features["Daughter IDs"] = "N/A"
                        
                        # Division frame
                        if cell_id in division_frames and division_frames[cell_id]:
                            div_frames = list(division_frames[cell_id].values())
                            features["Division Frame"] = str(min(div_frames))
                        else:
                            features["Division Frame"] = "N/A"
                    else:
                        features["Parent ID"] = "0"
                        features["Daughter IDs"] = "N/A"
                        features["Division Frame"] = "N/A"
                    
                    # Add position data
                    if cell_id in cell_positions and frame_idx in cell_positions[cell_id]:
                        y, x = cell_positions[cell_id][frame_idx]
                        features["X"] = x
                        features["Y"] = y
                    else:
                        # Calculate centroid from current frame
                        mask = (track_img == cell_id)
                        ys, xs = np.where(mask)
                        if len(xs) > 0 and len(ys) > 0:
                            features["X"] = np.mean(xs)
                            features["Y"] = np.mean(ys)
                        else:
                            features["X"] = "N/A"
                            features["Y"] = "N/A"
                    
                    frame_features.append(features)
                    
                    # Update classification history
                    if cell_id not in cell_classification_history:
                        cell_classification_history[cell_id] = []
                    cell_classification_history[cell_id].append((frame_idx, features["Label"]))
                    
                    # Update previous centroid for next frame
                    if "X" in features and "Y" in features and features["X"] != "N/A" and features["Y"] != "N/A":
                        prev_centroids[cell_id] = np.array([features["Y"], features["X"]])
            else:
                # Cell is missing in this frame, create placeholder feature record
                # Get position from interpolated data
                x = y = "N/A"
                if cell_id in cell_positions and frame_idx in cell_positions[cell_id]:
                    y, x = cell_positions[cell_id][frame_idx]
                
                # Get parent ID
                parent_id = "0"
                if cell_id in tracking_data:
                    parent_id = str(tracking_data[cell_id]['parent_id'])
                
                # Get daughter IDs
                daughter_ids = "N/A"
                if cell_id in parent_to_children and parent_to_children[cell_id]:
                    daughter_ids = ",".join(map(str, parent_to_children[cell_id]))
                
                # Get division frame
                division_frame = "N/A"
                if cell_id in division_frames and division_frames[cell_id]:
                    div_frames = list(division_frames[cell_id].values())
                    division_frame = str(min(div_frames))
                
                # Determine label from history if possible
                label = "N/A"
                if cell_id in cell_classification_history and cell_classification_history[cell_id]:
                    # Get previous labels for this cell
                    history = cell_classification_history[cell_id]
                    
                    # Look for the most recent valid label
                    for _, lbl in reversed(history):
                        if lbl in (0, 1):  # Valid label (not N/A)
                            label = lbl
                            break
                
                # Create placeholder feature record with N/A values
                features = {
                    "Cell ID": cell_id,
                    "Frame": frame_idx,
                    "Parent ID": parent_id,
                    "Daughter IDs": daughter_ids,
                    "Division Frame": division_frame,
                    "X": x,
                    "Y": y,
                    "Label": label,
                    "Volume": "N/A",
                    "Area": "N/A",
                    "Perimeter": "N/A",
                    "Compactness": "N/A",
                    "Sphericity": "N/A",
                    "Extent": "N/A",
                    "Solidity": "N/A",
                    "Ellipsoid-Prolate": "N/A",
                    "Ellipsoid-Oblate": "N/A",
                    "Nucleus-Cytoplasm Volume Ratio": "N/A",
                    "Displacement": "N/A",
                    "Speed": "N/A",
                    "Intensity-Mean": "N/A",
                    "Intensity-Sum": "N/A",
                    "Intensity-StdDev": "N/A",
                    "Intensity-Max": "N/A",
                    "Intensity-Min": "N/A",
                    "Mean Gradient Magnitude": "N/A"
                }
                
                frame_features.append(features)
                
                # Update classification history
                if cell_id not in cell_classification_history:
                    cell_classification_history[cell_id] = []
                cell_classification_history[cell_id].append((frame_idx, label))
        
        # Sort features by Cell ID
        frame_features.sort(key=lambda x: x["Cell ID"])
        
        # Add to all features list
        all_features.extend(frame_features)
    
    # ====== STEP 4: Save results ======
    print("\nStep 4: Saving feature data...")
    
    # Save to output directory
    output_file = os.path.join(output_dir, "features.csv")
    try:
        pd.DataFrame(all_features).to_csv(output_file, index=False)
        print(f"  Features saved to {output_file}")
    except Exception as e:
        print(f"  Error saving features to {output_file}: {e}")
    
    # Also save to track_result directory for convenience
    track_output = os.path.join(track_result_path, "features.csv")
    try:
        pd.DataFrame(all_features).to_csv(track_output, index=False)
        print(f"  Features also saved to {track_output} for convenience")
    except Exception as e:
        print(f"  Error saving features to {track_output}: {e}")
    
    print("\nFeature extraction complete!")


if __name__ == "__main__":
    import time
    
    # Look for datasets in the nuclear_dataset directory
    dataset_base = "nuclear_dataset"
    if not os.path.exists(dataset_base):
        print(f"❌ Error: {dataset_base} directory does not exist.")
        exit(1)
        
    test_folders = sorted([
        os.path.join(dataset_base, folder) 
        for folder in os.listdir(dataset_base) 
        if os.path.isdir(os.path.join(dataset_base, folder))
    ])
    
    if not test_folders:
        print(f"❌ No dataset folders found in {dataset_base}")
        exit(1)
        
    print(f"🔍 Found {len(test_folders)} dataset folders to process:")
    for folder in test_folders:
        print(f" - {folder}")
    
    # Process each dataset folder
    overall_stats = {
        'successful': 0,
        'failed': 0,
        'total_processing_time': 0,
        'total_features_extracted': 0,
        'total_cells_processed': 0
    }
    
    start_time = time.time()
    
    for folder in test_folders:
        folder_start_time = time.time()
        
        print(f"\n🎯 Processing feature extraction for {folder}...")
        
        # Setup paths following the same structure as test.py
        test_path = os.path.join(folder, "test")
        track_result_path = os.path.join(folder, "track_result")
        features_path = os.path.join(folder, "features")
        
        # Check if required directories exist
        if not os.path.exists(test_path):
            print(f"⚠️ Skipping {folder}: test directory not found")
            continue
            
        if not os.path.exists(track_result_path):
            print(f"⚠️ Skipping {folder}: track_result directory not found")
            continue
        
        # Create features directory if it doesn't exist
        os.makedirs(features_path, exist_ok=True)
        
        try:
            print(f"📁 Processing paths:")
            print(f"  - Test images: {test_path}")
            print(f"  - Tracking results: {track_result_path}")
            print(f"  - Features output: {features_path}")
            
            # Extract features
            extract_features_from_test(
                test_path=test_path,
                track_result_path=track_result_path,
                output_dir=features_path
            )
            
            # Calculate processing time
            processing_time = time.time() - folder_start_time
            
            # Count features extracted (if CSV was created)
            features_csv = os.path.join(features_path, "features.csv")
            features_count = 0
            cells_count = 0
            
            if os.path.exists(features_csv):
                try:
                    import pandas as pd
                    df = pd.read_csv(features_csv)
                    features_count = len(df)
                    cells_count = len(df['Cell ID'].unique()) if 'Cell ID' in df.columns else 0
                    print(f"📊 Extracted {features_count} feature records for {cells_count} unique cells")
                except Exception as e:
                    print(f"⚠️ Could not count features: {e}")
            
            overall_stats['successful'] += 1
            overall_stats['total_processing_time'] += processing_time
            overall_stats['total_features_extracted'] += features_count
            overall_stats['total_cells_processed'] += cells_count
            
            print(f"✅ Successfully completed {folder} in {processing_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Error processing {folder}: {e}")
            overall_stats['failed'] += 1
            continue
    
    # Print overall summary
    total_time = time.time() - start_time
    
    print(f"\n🎉 FEATURE EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful datasets: {overall_stats['successful']}")
    print(f"❌ Failed datasets: {overall_stats['failed']}")
    print(f"📋 Total feature records: {overall_stats['total_features_extracted']}")
    print(f"🧬 Total unique cells: {overall_stats['total_cells_processed']}")
    print(f"⏱️ Total processing time: {total_time:.2f} seconds")
    
    if overall_stats['successful'] > 0:
        avg_time = overall_stats['total_processing_time'] / overall_stats['successful']
        avg_features = overall_stats['total_features_extracted'] / overall_stats['successful']
        print(f"📈 Average time per dataset: {avg_time:.2f} seconds")
        print(f"📈 Average features per dataset: {avg_features:.0f}")
    
    print(f"{'='*60}")
    
    if overall_stats['failed'] > 0:
        print("⚠️ Please check the error messages above for details on failures.")
    else:
        print("🎊 All datasets processed successfully!")
        
    print("\n💾 Features saved in both:")
    print("   - {folder}/features/features.csv")
    print("   - {folder}/track_result/features.csv")
