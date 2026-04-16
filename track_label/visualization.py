#!/usr/bin/env python3
"""
Consensus Visualization System

WORKFLOW:
1. Take segmentation images from KIT-GE/nuclear_dataset/FOV/track_result  
2. Take test images from KIT-GE/nuclear_dataset/FOV/test (crucial for visualization)
3. Superimpose preprocessed green on segmentation where possible
4. Check IDs, start frame, end frame, parent, daughters in res_track (CONSENSUS results)
5. Look at positions in consensus_retrospective_labels.csv (the labelled result)
6. Use those IDs and positions for visualization

Direct use of consensus_retrospective_labels.csv instead of expensive position mapping
"""

import os
import pandas as pd
import numpy as np
import cv2
from typing import List, Tuple, Set
from collections import defaultdict, deque

class ConsensusVisualizerOptimized:
    """
    Consensus Visualization System 
    """
    
    def __init__(self, base_path: str, fov: str, reference_model: str = "KIT-GE"):
        self.base_path = base_path
        self.fov = fov
        self.reference_model = reference_model
        
        # CONSENSUS: All data paths point to consensus folder
        self.consensus_path = os.path.join(base_path, "consensus", fov)
        self.visualization_path = os.path.join(self.consensus_path, "Visualization")
        self.preprocessed_green_path = os.path.join(self.consensus_path, "Preprocessed_green")
        
        # KIT-GE: For segmentation and test images
        self.kit_ge_path = os.path.join(base_path, reference_model, "nuclear_dataset", fov)
        self.kit_ge_track_result_path = os.path.join(self.kit_ge_path, "track_result")
        self.kit_ge_test_path = os.path.join(self.kit_ge_path, "test")
        
        # Create visualization folder
        os.makedirs(self.visualization_path, exist_ok=True)
        
        # Data containers - same structure as visualization__.py
        self.consensus_labels_df = None
        self.tracking_info = {}  # Same as original
        self.ips_cells = set()   # Same as original 
        self.tracking_data_lines = []  # Same as original - needed for trajectory drawing
        self.frame_cells = defaultdict(list)  # frame -> [(cell_id, x, y, is_ips, area, volume)]
        self.cell_trajectories = defaultdict(dict)  # cell_id -> {frame: (x, y)}
        
        print(f"Initializing Optimized Consensus Visualizer for FOV {self.fov}")
        self.load_consensus_data()
    
    def load_image_robust(self, image_path):
        """Robustly load images in various formats"""
        try:
            # Try loading with different OpenCV flags first
            for flag in [-1, cv2.IMREAD_UNCHANGED, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_COLOR]:
                img = cv2.imread(image_path, flag)
                if img is not None:
                    return img
            
            # Try loading with tifffile for problematic TIFF formats
            try:
                import tifffile
                img = tifffile.imread(image_path)
                if img is not None:
                    # Convert to uint8 if needed
                    if img.dtype != np.uint8:
                        if img.max() > 255:
                            img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    return img
            except ImportError:
                print("For better TIFF support, install tifffile: pip install tifffile")
            except Exception as e:
                print(f"tifffile error for {image_path}: {e}")
            
            # Try with PIL as last resort
            try:
                from PIL import Image
                pil_img = Image.open(image_path)
                img = np.array(pil_img)
                if img is not None:
                    return img
            except ImportError:
                print("For additional format support, install pillow: pip install pillow")
            except Exception as e:
                print(f"PIL error for {image_path}: {e}")
            
            print(f"Could not load image with any method: {image_path}")
            return None
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def load_consensus_data(self):
        """Load consensus data - same as visualization__.py but from consensus sources"""
        print("Loading consensus data...")
        
        # Load consensus_retrospective_labels.csv (equivalent to arranged_retrospective_labels.csv)
        consensus_labels_path = os.path.join(self.consensus_path, "Labelled", "consensus_retrospective_labels.csv")
        if not os.path.exists(consensus_labels_path):
            print(f"Consensus labels not found at {consensus_labels_path}")
            return False
        
        self.consensus_labels_df = pd.read_csv(consensus_labels_path)
        print(f"Loaded consensus retrospective labels: {len(self.consensus_labels_df)} records")
        
        # Get iPS cells (Label = 1) - same as visualization__.py
        self.ips_cells = set(self.consensus_labels_df[self.consensus_labels_df['Label'] == 1]['Cell ID'].unique())
        print(f"Found {len(self.ips_cells)} consensus iPS cells")
        
        label_counts = self.consensus_labels_df['Label'].value_counts().to_dict()
        print(f"Consensus label distribution: {label_counts}")
        
        # PERFORMANCE OPTIMIZATION: Pre-organize data by frame for fast lookup
        print("Pre-organizing consensus data by frame...")
        for _, row in self.consensus_labels_df.iterrows():
            frame = row['Frame']
            cell_id = row['Cell ID']
            x = row['X']
            y = row['Y']
            is_ips = (row['Label'] == 1)
            area = row.get('Area', 0)
            volume = row.get('Volume', 0)
            
            # Store for fast frame-based lookup - same structure as visualization__.py
            self.frame_cells[frame].append((cell_id, x, y, is_ips, area, volume))
            
            # Store for trajectory calculations
            self.cell_trajectories[cell_id][frame] = (x, y)
        
        print(f"Organized data for {len(self.frame_cells)} frames")
        print(f"Trajectory data for {len(self.cell_trajectories)} cells")
        
        # Load consensus tracking data from res_track.txt (same format as visualization__.py)
        consensus_track_file = os.path.join(self.consensus_path, "res_track.txt")
        if os.path.exists(consensus_track_file):
            with open(consensus_track_file, "r") as f:
                self.tracking_data_lines = f.readlines()  # Store lines for trajectory drawing
                
            for line in self.tracking_data_lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    cell_id = int(parts[0])
                    start_frame = int(parts[1])
                    end_frame = int(parts[2])
                    parent_id = int(parts[3])
                    
                    self.tracking_info[cell_id] = {
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'parent_id': parent_id
                    }
            
            print(f"Loaded consensus tracking for {len(self.tracking_info)} cells")
        else:
            print("No consensus tracking file found")
            self.tracking_data_lines = []
        
        return True
    
    def get_coloured_mask(self, mask, cell_id):
        """Get a colored mask for each cell based on cell ID"""
        colours = [
            [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255],
            [255, 128, 0], [128, 255, 0], [0, 255, 128], [0, 128, 255], [128, 0, 255], [255, 0, 128],
            [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190], [0, 128, 0], 
            [255, 165, 0], [128, 0, 0], [0, 128, 128], [128, 128, 0], [192, 192, 192], [255, 192, 203],
            [255, 69, 0], [255, 20, 147], [72, 61, 139], [106, 90, 205], [123, 104, 238], [147, 112, 219]
        ]
        
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        
        # Use cell_id to get consistent color for same cell across frames
        color_idx = cell_id % len(colours)
        color = colours[color_idx]
        
        r[mask == 255] = color[2]  # BGR format
        g[mask == 255] = color[1]
        b[mask == 255] = color[0]
        
        coloured_mask = np.stack([b, g, r], axis=2)
        return coloured_mask
    
    def draw_ips_marker(self, image, center, cell_id):
        """Draw a marker for iPS cells - blue bounding box with white text"""
        x, y = center
        
        # Draw "iPS" text with smaller font
        cv2.putText(image, "iPS", (x-12, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw cell ID below with smaller font
        cv2.putText(image, str(cell_id), (x-10, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def draw_normal_marker(self, image, center, cell_id):
        """Draw a simple marker for normal cells"""
        x, y = center
        
        # Only draw cell ID text
        cv2.putText(image, str(cell_id), (x-10, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def get_consensus_center(self, frame_idx, cell_id):
        """Get centroid of a consensus cell in a specific frame"""
        if cell_id in self.cell_trajectories and frame_idx in self.cell_trajectories[cell_id]:
            x, y = self.cell_trajectories[cell_id][frame_idx]
            return (int(x), int(y))
        return None
    
    def draw_trajectory_lines(self, trace_images):
        """Draw trajectory lines using consensus tracking data """
        print("Drawing consensus trajectory lines...")
        
        # Process consensus tracking data to draw trajectories - SAME AS ORIGINAL
        lines = [line.strip('\n') for line in self.tracking_data_lines]
        
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
                
            cell_id = int(parts[0])
            start_frame = int(parts[1])
            end_frame = int(parts[2])
            parent_id = int(parts[3])
            
            # Process for cells that have tracks (multiple frames) - SAME AS ORIGINAL
            if start_frame != end_frame and start_frame < len(trace_images) and end_frame < len(trace_images):
                # Get starting center and place red dot
                center = self.get_consensus_center(start_frame, cell_id)
                if center:
                    cv2.circle(trace_images[start_frame], center, 2, (0, 0, 255), -1)  # Red dot at start
                start_point = center
                
                # Connect tracked centroids across frames with red lines - SAME AS ORIGINAL
                for i in range(start_frame + 1, min(end_frame + 1, len(trace_images))):
                    center = self.get_consensus_center(i, cell_id)
                    if center:
                        cv2.circle(trace_images[i], center, 2, (0, 0, 255), -1)  # Red dot at each point
                        
                        # Draw red line from start_point to current center on all frames from start to current
                        if start_point:
                            for j in range(start_frame, i):
                                if j < len(trace_images):
                                    cv2.line(trace_images[j], start_point, center, (0, 0, 255), 1)  # Red trajectory line
                        start_point = center
    
    def create_frame_visualization(self, frame_idx, show_preprocessed_overlay=True):
        """Create visualization for a single frame using consensus data"""
        try:
            # 1. Load KIT-GE test image (crucial for visualization)
            kit_ge_test_files = sorted([f for f in os.listdir(self.kit_ge_test_path) 
                                      if f.endswith('.tif') or f.endswith('.tiff')])
            
            if frame_idx >= len(kit_ge_test_files):
                print(f"Frame {frame_idx} not available (max: {len(kit_ge_test_files)-1})")
                return None
            
            test_img_path = os.path.join(self.kit_ge_test_path, kit_ge_test_files[frame_idx])
            original_img = self.load_image_robust(test_img_path)
            
            if original_img is None:
                print(f"Could not load test image: {test_img_path}")
                return None
            
            # Ensure it's color image
            if len(original_img.shape) == 2:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
            elif len(original_img.shape) == 3 and original_img.shape[2] == 4:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR)
            
            # Apply cropping (same as original)
            if original_img.shape[0] >= 741 and original_img.shape[1] >= 769:
                original_img = original_img[5:741, 1:769]
            
            # Create visualization image
            vis_img = original_img.copy()
            
            # 3. Superimpose preprocessed green where possible
            if show_preprocessed_overlay and frame_idx >= 275:
                preprocessed_files = sorted([f for f in os.listdir(self.preprocessed_green_path) 
                                           if f.endswith('.tif') or f.endswith('.tiff')])
                
                preprocessed_frame_idx = frame_idx - 275  # Preprocessed starts at frame 275
                
                if 0 <= preprocessed_frame_idx < len(preprocessed_files):
                    preprocessed_img_path = os.path.join(self.preprocessed_green_path, preprocessed_files[preprocessed_frame_idx])
                    preprocessed_img = self.load_image_robust(preprocessed_img_path)
                    
                    if preprocessed_img is not None:
                        # Preprocessed image is already in color (GREEN background + WHITE cells)
                        if len(preprocessed_img.shape) == 2:
                            preprocessed_img = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
                        
                        # Resize preprocessed image to match original
                        if preprocessed_img.shape[:2] != original_img.shape[:2]:
                            preprocessed_img = cv2.resize(preprocessed_img, 
                                                        (original_img.shape[1], original_img.shape[0]),
                                                        interpolation=cv2.INTER_LINEAR)
                        
                        # Overlay preprocessed image (GREEN background + WHITE cells)
                        vis_img = cv2.addWeighted(vis_img, 0.8, preprocessed_img, 0.2, 0)
            
            # 1. Load KIT-GE segmentation images for boundaries
            kit_ge_track_files = sorted([f for f in os.listdir(self.kit_ge_track_result_path) 
                                       if f.endswith('.tif') or f.endswith('.tiff')])
            
            segmentation_mask = None
            if frame_idx < len(kit_ge_track_files):
                track_img_path = os.path.join(self.kit_ge_track_result_path, kit_ge_track_files[frame_idx])
                track_img = self.load_image_robust(track_img_path)
                
                if track_img is not None:
                    # Ensure single channel
                    if len(track_img.shape) > 2:
                        track_img = cv2.cvtColor(track_img, cv2.COLOR_BGR2GRAY)
                    
                    # Apply same cropping
                    if track_img.shape[0] >= 741 and track_img.shape[1] >= 769:
                        track_img = track_img[5:741, 1:769]
                    
                    segmentation_mask = track_img
            
            # 5-6. Use consensus IDs and positions for visualization - SAME LOGIC as visualization__.py
            cell_count = 0
            ips_count = 0
            
            if frame_idx in self.frame_cells:
                for cell_id, x, y, is_ips, area, volume in self.frame_cells[frame_idx]:
                    # Validate cell should be present in this frame - SAME AS ORIGINAL
                    if cell_id in self.tracking_info:
                        start_frame = self.tracking_info[cell_id]['start_frame']
                        end_frame = self.tracking_info[cell_id]['end_frame']
                        
                        if not (start_frame <= frame_idx <= end_frame):
                            continue
                    
                    # Convert to int coordinates
                    x, y = int(x), int(y)
                    
                    # Create colored segmentation mask if segmentation available - SAME AS ORIGINAL
                    if segmentation_mask is not None:
                        # Create label picture for this cell - SAME LOGIC as original
                        label_picture = np.zeros_like(segmentation_mask)
                        
                        # Search in a radius around the consensus position
                        search_radius = 20
                        y_min = max(0, y - search_radius)
                        y_max = min(segmentation_mask.shape[0], y + search_radius)
                        x_min = max(0, x - search_radius)
                        x_max = min(segmentation_mask.shape[1], x + search_radius)
                        
                        # Find segmented pixels in this region and match to consensus position
                        region = segmentation_mask[y_min:y_max, x_min:x_max]
                        if np.any(region > 0):
                            # Create mask for segmented region
                            region_mask = (region > 0).astype(np.uint8) * 255
                            
                            # Ensure single channel for contour detection - SAME AS ORIGINAL
                            if len(region_mask.shape) > 2:
                                region_mask = cv2.cvtColor(region_mask, cv2.COLOR_BGR2GRAY)
                            
                            try:
                                contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                if contours:
                                    # Use largest contour
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    
                                    # Offset contour to full image coordinates
                                    largest_contour = largest_contour + np.array([[x_min, y_min]])
                                    
                                    # Draw filled contour on mask - SAME AS ORIGINAL
                                    mask = np.zeros_like(vis_img[:, :, 0])
                                    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                                    
                                    # Create colored mask for this cell - SAME AS ORIGINAL
                                    colored_mask = self.get_coloured_mask(mask, cell_id)
                                    vis_img = cv2.addWeighted(vis_img, 1, colored_mask, 0.4, 0)
                                    
                                    # Draw bounding box for iPS cells - BLUE color - SAME AS ORIGINAL
                                    if is_ips:
                                        x_box, y_box, w_box, h_box = cv2.boundingRect(largest_contour)
                                        cv2.rectangle(vis_img, (x_box, y_box), (x_box + w_box, y_box + h_box), (255, 0, 0), 2)
                                        self.draw_ips_marker(vis_img, (x, y), cell_id)
                                        ips_count += 1
                                    else:
                                        self.draw_normal_marker(vis_img, (x, y), cell_id)
                                    
                                    cell_count += 1
                                        
                            except cv2.error as e:
                                # Fallback: just draw marker at consensus position - SAME AS ORIGINAL
                                print(f"OpenCV error in frame {frame_idx}: {e}")
                                if is_ips:
                                    cv2.rectangle(vis_img, (x-15, y-15), (x+15, y+15), (255, 0, 0), 2)
                                    self.draw_ips_marker(vis_img, (x, y), cell_id)
                                    ips_count += 1
                                else:
                                    self.draw_normal_marker(vis_img, (x, y), cell_id)
                                cell_count += 1
                        else:
                            # No segmentation found, just draw marker at consensus position
                            if is_ips:
                                cv2.rectangle(vis_img, (x-15, y-15), (x+15, y+15), (255, 0, 0), 2)
                                self.draw_ips_marker(vis_img, (x, y), cell_id)
                                ips_count += 1
                            else:
                                self.draw_normal_marker(vis_img, (x, y), cell_id)
                            cell_count += 1
                    else:
                        # No segmentation available, just draw marker at consensus position
                        if is_ips:
                            cv2.rectangle(vis_img, (x-15, y-15), (x+15, y+15), (255, 0, 0), 2)
                            self.draw_ips_marker(vis_img, (x, y), cell_id)
                            ips_count += 1
                        else:
                            self.draw_normal_marker(vis_img, (x, y), cell_id)
                        cell_count += 1
            
            # Add frame information with YELLOW BOLD font
            cv2.putText(vis_img, f"Frame {frame_idx}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Yellow bold
            
            # Add cell count information with YELLOW BOLD font
            cv2.putText(vis_img, f"iPS: {ips_count}/{cell_count}", (10, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Yellow bold
            
            return vis_img
            
        except Exception as e:
            print(f"Error creating frame visualization {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_all_frame_visualizations(self, start_frame=0, end_frame=400):
        """Create visualizations for all frames with trajectory lines"""
        print(f"Creating consensus visualizations for ALL frames: {start_frame} to {end_frame}")
        print(f"Output format: 6-digit zero-padded .tif files (000000.tif - {end_frame:06d}.tif)")
        print(f"Using optimized consensus data (no expensive position mapping)")
        
        # Get total number of available test frames
        kit_ge_test_files = sorted([f for f in os.listdir(self.kit_ge_test_path) 
                                  if f.endswith('.tif') or f.endswith('.tiff')])
        max_available_frame = len(kit_ge_test_files) - 1
        
        actual_end_frame = min(end_frame, max_available_frame)
        print(f"Available test frames: 0 to {max_available_frame}")
        print(f"Will create visualizations for frames: {start_frame} to {actual_end_frame}")
        
        # Create all frame visualizations first (without trajectories)
        trace_images = []
        for frame_idx in range(start_frame, actual_end_frame + 1):
            show_preprocessed = frame_idx >= 275
            vis_img = self.create_frame_visualization(frame_idx, show_preprocessed_overlay=show_preprocessed)
            
            if vis_img is not None:
                trace_images.append(vis_img)
            else:
                # Create a blank image as placeholder
                if trace_images:
                    blank_img = np.zeros_like(trace_images[0])
                    trace_images.append(blank_img)
                else:
                    # Default size if no previous images
                    blank_img = np.zeros((736, 768, 3), dtype=np.uint8)
                    trace_images.append(blank_img)
            
            # Progress indicator
            if (frame_idx - start_frame) % 50 == 0:
                print(f"Progress: {frame_idx}/{actual_end_frame}")
        
        # Draw trajectory lines on all frames
        if trace_images:
            self.draw_trajectory_lines(trace_images)
        
        # Save all frames
        successful_saves = 0
        failed_saves = 0
        
        for i, vis_img in enumerate(trace_images):
            frame_idx = start_frame + i
            output_path = os.path.join(self.visualization_path, f"{frame_idx:06d}.tif")
            success = cv2.imwrite(output_path, vis_img)
            
            if success:
                successful_saves += 1
                if frame_idx % 50 == 0:  # Print progress every 50 frames
                    print(f"Progress: Frame {frame_idx}/{actual_end_frame} saved as {frame_idx:06d}.tif")
            else:
                failed_saves += 1
                print(f"Failed to save frame {frame_idx}")
        
        print(f"\nCONSENSUS FRAME VISUALIZATION COMPLETE!")
        print(f" Summary:")
        print(f"   Successfully saved: {successful_saves} frames")
        print(f"   Failed: {failed_saves} frames")
        print(f"   Trajectory lines: Enabled (consensus lineage)")
        print(f"   Preprocessed overlay: GREEN background + WHITE cells")
        print(f"   iPS cells: Blue bounding boxes (consensus labels)")
        print(f"   Data source: consensus_retrospective_labels.csv")
        print(f"   Files saved as: 000000.tif to {actual_end_frame:06d}.tif")
        print(f"   Performance: OPTIMIZED - minutes instead of 24 hours!")
        
        return successful_saves, failed_saves
    
    def create_comparison_visualization(self, early_frame=275, late_frame=400):
        """Create side-by-side comparison"""
        print(f"Creating comparison: Frame {early_frame} vs Frame {late_frame}")
        
        early_vis = self.create_frame_visualization(early_frame, show_preprocessed_overlay=True)
        late_vis = self.create_frame_visualization(late_frame, show_preprocessed_overlay=True)
        
        if early_vis is not None and late_vis is not None:
            # Ensure same height
            h1, w1 = early_vis.shape[:2]
            h2, w2 = late_vis.shape[:2]
            
            target_height = min(h1, h2)
            early_vis = cv2.resize(early_vis, (int(w1 * target_height / h1), target_height))
            late_vis = cv2.resize(late_vis, (int(w2 * target_height / h2), target_height))
            
            # Combine side by side
            combined = np.hstack((early_vis, late_vis))
            
            # Add titles
            cv2.putText(combined, f"Early Frame {early_frame}", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(combined, f"Late Frame {late_frame}", 
                       (early_vis.shape[1] + 50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            comparison_path = os.path.join(self.visualization_path, "consensus_comparison_early_vs_late.tif")
            cv2.imwrite(comparison_path, combined)
            print(f"Saved comparison: {comparison_path}")
        else:
            print("Failed to create comparison visualization")
    
    def create_summary_report(self):
        """Create summary image"""
        print("Creating summary image...")
        
        summary_img = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(summary_img, f"Optimized Consensus iPS Detection - FOV {self.fov}", 
                   (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Statistics
        if self.consensus_labels_df is not None:
            total_cells = len(self.consensus_labels_df['Cell ID'].unique())
            ips_count = len(self.ips_cells)
            ips_percentage = (ips_count / total_cells) * 100 if total_cells > 0 else 0
            
            y_pos = 120
            stats = [
                f"Method: OPTIMIZED Consensus Visualization",
                f"Total Consensus Cells: {total_cells}",
                f"Consensus iPS Cells: {ips_count}",
                f"iPS Percentage: {ips_percentage:.1f}%",
                f"Tracking Data: {len(self.tracking_info)} cells",
                f"Frames Visualized: 0-400 (all frames)",
                f"Output Format: 6-digit .tif files",
                f"Data Source: consensus_retrospective_labels.csv",
                f"Segmentation: KIT-GE track_result boundaries",
                f"Performance: FIXED 24-hour runtime issue!",
                f"Expected Runtime: Minutes instead of hours",
                f"Optimization: Direct consensus data usage"
            ]
            
            for stat in stats:
                cv2.putText(summary_img, stat, (70, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 35
        
        summary_path = os.path.join(self.visualization_path, "optimized_consensus_summary.tif")
        #cv2.imwrite(summary_img, summary_path)
        cv2.imwrite(summary_path, summary_img)
        print(f"Saved summary: {summary_path}")
    
    def run_visualization(self, visualize_all_frames=True, key_frames=None):
        """Run complete visualization pipeline"""
        print("="*70)
        print(f"OPTIMIZED CONSENSUS VISUALIZATION - FOV {self.fov}")
        print("PERFORMANCE FIXES APPLIED:")
        print("   Removed expensive position mapping")
        print("   Direct use of consensus_retrospective_labels.csv")
        print("   Pre-organized data structures")
        print("   Expected runtime: MINUTES instead of 24 hours!")
        print("="*70)
        
        if self.consensus_labels_df is None:
            print(" No consensus data found. Run consensus labeling first!")
            return False
        
        # Create frame visualizations
        if visualize_all_frames:
            print("\n Creating ALL FRAME consensus visualizations (0-400) with trajectories...")
            successful, failed = self.create_all_frame_visualizations(start_frame=0, end_frame=400)
            if failed > 0:
                print(f" Warning: {failed} frames failed to render")
        
        # Create comparison visualization
        print("\n Creating comparison visualization...")
        self.create_comparison_visualization()
        
        # Create summary report
        print("\nCreating summary report...")
        self.create_summary_report()
        
        print(f"\nVisualization completed!")
        print(f"Results saved in: {self.visualization_path}")
        print(f" Performance: OPTIMIZED")
        
        return True


def visualize_single_fov_consensus(base_path, fov, reference_model="KIT-GE", visualize_all_frames=True, key_frames=None):
    """Visualize single FOV with optimized consensus data"""
    try:
        visualizer = ConsensusVisualizerOptimized(base_path, fov, reference_model)
        return visualizer.run_visualization(visualize_all_frames, key_frames)
    except Exception as e:
        print(f" Error visualizing consensus FOV {fov}: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_all_fovs_consensus(base_path=".", reference_model="KIT-GE", visualize_all_frames=True, key_frames=None):
    """Visualize all FOVs with optimized consensus data"""
    consensus_dir = os.path.join(base_path, "consensus")
    
    if not os.path.exists(consensus_dir):
        print(f" Directory not found: {consensus_dir}")
        return
    
    # Find FOVs with consensus data
    fov_dirs = []
    for d in os.listdir(consensus_dir):
        if os.path.isdir(os.path.join(consensus_dir, d)) and d.isdigit():
            fov_num = int(d)
            if 2 <= fov_num <= 54:
                consensus_labels_path = os.path.join(consensus_dir, d, "Labelled", "consensus_retrospective_labels.csv")
                if os.path.exists(consensus_labels_path):
                    fov_dirs.append(d)
    
    fov_dirs = sorted(fov_dirs, key=int)
    print(f" Found {len(fov_dirs)} FOVs with consensus data")
    
    successful = 0
    for fov in fov_dirs:
        print(f"\n{'='*50}")
        print(f" Visualizing Consensus FOV {fov}")
        print(f"{'='*50}")
        
        success = visualize_single_fov_consensus(base_path, fov, reference_model, visualize_all_frames, key_frames)
        if success:
            successful += 1
            print(f" Consensus FOV {fov} visualization completed")
        else:
            print(f" Consensus FOV {fov} visualization failed")
    
    print(f"\n{'='*70}")
    print(" CONSENSUS VISUALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully visualized: {successful}/{len(fov_dirs)} FOVs")
    print(f" Performance: OPTIMIZED - no more 24-hour runtime!")
    print(f" Data workflow:")
    print(f"   1. KIT-GE segmentation (track_result) for boundaries")
    print(f"   2. KIT-GE test images for visualization base")
    print(f"   3. Preprocessed green overlay where available")
    print(f"   4. Consensus res_track.txt for lineage")
    print(f"   5. consensus_retrospective_labels.csv for positions/IDs")


if __name__ == "__main__":
    print(" Testing consensus visualization!")
    print(" Performance issue!")
    print(" Using consensus_retrospective_labels.csv directly")
    print(" Optimal runtime")
    visualize_single_fov_consensus(".", "2", "KIT-GE", visualize_all_frames=True)
