import os
import pandas as pd
import numpy as np
import cv2
from typing import List, Tuple, Set

class iPSVisualizerProper:
    """
    iPS Visualization system with trajectory lines and improved styling
    FEATURES:
    1. Red trajectory lines showing cell movement across frames
    2. Yellow bold fonts for frame information and iPS counts
    3. Blue bounding boxes for iPS cells (no circles)
    4. FIXED OpenCV contour detection errors
    5. Robust image loading for various TIFF formats
    6. Uses preprocessed GREEN background with WHITE cells
    """
    
    def __init__(self, base_path: str, fov: str):
        self.base_path = base_path
        self.fov = fov
        
        # Paths
        self.fov_path = os.path.join(base_path, "nuclear_dataset", fov)
        self.test_path = os.path.join(self.fov_path, "test")
        self.preprocessed_green_path = os.path.join(self.fov_path, "Preprocessed_green")  # Changed from green_signal
        self.track_result_path = os.path.join(self.fov_path, "track_result")
        self.labelled_path = os.path.join(self.fov_path, "Labelled")
        self.visualization_path = os.path.join(self.fov_path, "Visualization")
        
        # Create visualization folder
        os.makedirs(self.visualization_path, exist_ok=True)
        
        # Load labelled data and tracking data
        self.labelled_df = None
        self.tracking_info = {}
        self.ips_cells = set()
        self.tracking_data_lines = []
        self.load_data()
    
    def safe_array_to_scalar(self, value, default=0):
        """Safely convert numpy array or other value to scalar"""
        try:
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return value.item()
                elif value.size > 1:
                    return value.flat[0]  # Take first element
                else:
                    return default
            else:
                return value
        except Exception:
            return default
    
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
                print("💡 For better TIFF support, install tifffile: pip install tifffile")
            except Exception as e:
                print(f"⚠️ tifffile error for {image_path}: {e}")
            
            # Try with PIL as last resort
            try:
                from PIL import Image
                pil_img = Image.open(image_path)
                img = np.array(pil_img)
                if img is not None:
                    return img
            except ImportError:
                print("💡 For additional format support, install pillow: pip install pillow")
            except Exception as e:
                print(f"⚠️ PIL error for {image_path}: {e}")
            
            print(f"⚠️ Could not load image with any method: {image_path}")
            return None
            
        except Exception as e:
            print(f"⚠️ Error loading {image_path}: {e}")
            return None
    
    def load_data(self):
        """Load labelled data and tracking information"""
        # Load labelled CSV
        labelled_csv_path = os.path.join(self.labelled_path, "arranged_retrospective_labels.csv")
        if os.path.exists(labelled_csv_path):
            self.labelled_df = pd.read_csv(labelled_csv_path)
            print(f"Loaded retrospective labelled data: {len(self.labelled_df)} records")
            
            # Get iPS cells (Label = 1)
            self.ips_cells = set(self.labelled_df[self.labelled_df['Label'] == 1]['Cell ID'].unique())
            print(f"Found {len(self.ips_cells)} iPS cells")
            
            label_counts = self.labelled_df['Label'].value_counts().to_dict()
            print(f"Label distribution: {label_counts}")
        else:
            print(f"Warning: No retrospective labelled data found at {labelled_csv_path}")
            return False
        
        # Load tracking data from res_track.txt
        track_file = os.path.join(self.track_result_path, "res_track.txt")
        if os.path.exists(track_file):
            with open(track_file, "r") as f:
                self.tracking_data_lines = f.readlines()
            
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
            
            print(f"Loaded tracking info for {len(self.tracking_info)} cells")
        else:
            print(f"Warning: res_track.txt not found at {track_file}")
        
        return True
    
    def get_center(self, frame_idx, cell_id):
        """Get centroid of a cell in a specific frame"""
        track_files = sorted([f for f in os.listdir(self.track_result_path) 
                             if f.endswith('.tif') or f.endswith('.tiff')])
        
        if frame_idx >= len(track_files):
            return None
        
        track_img_path = os.path.join(self.track_result_path, track_files[frame_idx])
        result_picture = self.load_image_robust(track_img_path)
        
        if result_picture is None:
            return None
        
        # Ensure single channel
        if len(result_picture.shape) > 2:
            result_picture = cv2.cvtColor(result_picture, cv2.COLOR_BGR2GRAY)
        
        # Apply same cropping as visualization
        if result_picture.shape[0] >= 741 and result_picture.shape[1] >= 769:
            result_picture = result_picture[5:741, 1:769]
        
        # Create mask for this specific cell ID - ENSURE SINGLE CHANNEL
        label_picture = ((result_picture == cell_id) * 255).astype(np.uint8)
        
        # CRITICAL FIX: Ensure single channel for contour detection
        if len(label_picture.shape) > 2:
            label_picture = cv2.cvtColor(label_picture, cv2.COLOR_BGR2GRAY)
        
        try:
            contours, _ = cv2.findContours(label_picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Use the largest contour if multiple exist
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy)
        except cv2.error as e:
            print(f"OpenCV error in get_center for cell {cell_id}, frame {frame_idx}: {e}")
            return None
        
        return None
    
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
        """Draw a marker for iPS cells - white bounding box only, smaller font"""
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
    
    def draw_trajectory_lines(self, trace_images):
        """Draw trajectory lines on all frames"""
        print("🔴 Drawing trajectory lines...")
        
        # Process tracking data to draw trajectories
        lines = [line.strip('\n') for line in self.tracking_data_lines]
        
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
                
            cell_id = int(parts[0])
            start_frame = int(parts[1])
            end_frame = int(parts[2])
            parent_id = int(parts[3])
            
            # Process for cells that have tracks (multiple frames)
            if start_frame != end_frame and start_frame < len(trace_images) and end_frame < len(trace_images):
                # Get starting center and place red dot
                center = self.get_center(start_frame, cell_id)
                if center:
                    cv2.circle(trace_images[start_frame], center, 2, (0, 0, 255), -1)  # Red dot at start
                start_point = center
                
                # Connect tracked centroids across frames with red lines
                for i in range(start_frame + 1, min(end_frame + 1, len(trace_images))):
                    center = self.get_center(i, cell_id)
                    if center:
                        cv2.circle(trace_images[i], center, 2, (0, 0, 255), -1)  # Red dot at each point
                        
                        # Draw red line from start_point to current center on all frames from start to current
                        if start_point:
                            for j in range(start_frame, i):
                                if j < len(trace_images):
                                    cv2.line(trace_images[j], start_point, center, (0, 0, 255), 1)  # Red trajectory line
                        start_point = center
    
    def create_frame_visualization(self, frame_idx, show_preprocessed_overlay=True):
        """Create visualization for a single frame using preprocessed overlay"""
        try:
            # Load original image
            test_files = sorted([f for f in os.listdir(self.test_path) 
                                if f.endswith('.tif') or f.endswith('.tiff')])
            
            if frame_idx >= len(test_files):
                print(f"❌ Frame {frame_idx} not available (max: {len(test_files)-1})")
                return None
            
            test_img_path = os.path.join(self.test_path, test_files[frame_idx])
            original_img = self.load_image_robust(test_img_path)
            
            if original_img is None:
                print(f"❌ Could not load image: {test_img_path}")
                return None
            
            # Ensure it's color image
            if len(original_img.shape) == 2:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
            elif len(original_img.shape) == 3 and original_img.shape[2] == 4:
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR)
            
            # Apply the same cropping as in generate_trace.py
            if original_img.shape[0] >= 741 and original_img.shape[1] >= 769:
                original_img = original_img[5:741, 1:769]
            
            # Create visualization image
            vis_img = original_img.copy()
            
            # Add preprocessed overlay if available (CHANGED FROM GREEN_SIGNAL)
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
                        vis_img = cv2.addWeighted(vis_img, 0.6, preprocessed_img, 0.4, 0)
            
            # Load tracking image
            track_files = sorted([f for f in os.listdir(self.track_result_path) 
                                 if f.endswith('.tif') or f.endswith('.tiff')])
            
            if frame_idx < len(track_files):
                track_img_path = os.path.join(self.track_result_path, track_files[frame_idx])
                track_img = self.load_image_robust(track_img_path)
                
                if track_img is not None:
                    # Ensure single channel
                    if len(track_img.shape) > 2:
                        track_img = cv2.cvtColor(track_img, cv2.COLOR_BGR2GRAY)
                    
                    # Apply same cropping
                    if track_img.shape[0] >= 741 and track_img.shape[1] >= 769:
                        track_img = track_img[5:741, 1:769]
                    
                    # CRITICAL FIX: Create proper single-channel mask for contour detection
                    label_picture = ((track_img >= 1) * 255).astype(np.uint8)
                    
                    # Ensure single channel
                    if len(label_picture.shape) > 2:
                        label_picture = cv2.cvtColor(label_picture, cv2.COLOR_BGR2GRAY)
                    
                    try:
                        contours, _ = cv2.findContours(label_picture, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for contour in contours:
                            # Get centroid
                            M = cv2.moments(contour)
                            if M['m00'] == 0:
                                continue
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            
                            # Get cell ID from tracking image
                            if cy < track_img.shape[0] and cx < track_img.shape[1]:
                                cell_id = track_img[cy, cx]
                                cell_id = self.safe_array_to_scalar(cell_id)
                                cell_id = int(cell_id)
                                
                                # Validate cell ID exists in tracking data
                                if cell_id in self.tracking_info:
                                    # Check if cell should be present in this frame
                                    start_frame = self.tracking_info[cell_id]['start_frame']
                                    end_frame = self.tracking_info[cell_id]['end_frame']
                                    
                                    if start_frame <= frame_idx <= end_frame:
                                        # Create colored mask for this cell (full segmentation)
                                        mask = np.zeros_like(vis_img[:, :, 0])
                                        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                                        colored_mask = self.get_coloured_mask(mask, cell_id)
                                        vis_img = cv2.addWeighted(vis_img, 1, colored_mask, 0.4, 0)
                                        
                                        # Draw bounding box for iPS cells - BLUE color
                                        if cell_id in self.ips_cells:
                                            x, y, w, h = cv2.boundingRect(contour)
                                            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue bounding box
                                            self.draw_ips_marker(vis_img, (cx, cy), cell_id)
                                        else:
                                            self.draw_normal_marker(vis_img, (cx, cy), cell_id)
                    
                    except cv2.error as e:
                        print(f"⚠️ OpenCV error in frame {frame_idx}: {e}")
                        print("Skipping contour detection for this frame")
            
            # Add frame information with YELLOW BOLD font
            cv2.putText(vis_img, f"Frame {frame_idx}", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Yellow bold
            
            # Add cell count information with YELLOW BOLD font  
            frame_cells = self.labelled_df[self.labelled_df['Frame'] == frame_idx] if self.labelled_df is not None else pd.DataFrame()
            if not frame_cells.empty:
                ips_count = len(frame_cells[frame_cells['Label'] == 1])
                total_count = len(frame_cells)
                cv2.putText(vis_img, f"iPS: {ips_count}/{total_count}", (10, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Yellow bold - same size as frame
            
            return vis_img
            
        except Exception as e:
            print(f"❌ Error creating frame visualization {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_all_frame_visualizations(self, start_frame=0, end_frame=400):
        """Create visualizations for all frames with trajectory lines"""
        print(f"🎬 Creating visualizations for ALL frames: {start_frame} to {end_frame}")
        print(f"📁 Output format: 6-digit zero-padded .tif files (000000.tif - {end_frame:06d}.tif)")
        print(f"🔬 Using preprocessed GREEN background + WHITE cells overlay")
        
        # Get total number of available test frames
        test_files = sorted([f for f in os.listdir(self.test_path) 
                            if f.endswith('.tif') or f.endswith('.tiff')])
        max_available_frame = len(test_files) - 1
        
        actual_end_frame = min(end_frame, max_available_frame)
        print(f"📊 Available test frames: 0 to {max_available_frame}")
        print(f"🎯 Will create visualizations for frames: {start_frame} to {actual_end_frame}")
        
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
                    print(f"✅ Progress: Frame {frame_idx}/{actual_end_frame} saved as {frame_idx:06d}.tif")
            else:
                failed_saves += 1
                print(f"❌ Failed to save frame {frame_idx}")
        
        print(f"\n🎬 ALL FRAME VISUALIZATION COMPLETE!")
        print(f"📊 Summary:")
        print(f"   ✅ Successfully saved: {successful_saves} frames")
        print(f"   ❌ Failed: {failed_saves} frames")
        print(f"   🔴 Trajectory lines: Enabled")
        print(f"   🔬 Preprocessed overlay: GREEN background + WHITE cells")
        print(f"   📁 Total frames processed: {actual_end_frame - start_frame + 1}")
        print(f"   💾 Files saved as: 000000.tif to {actual_end_frame:06d}.tif")
        
        return successful_saves, failed_saves
    
    def create_comparison_visualization(self, early_frame=275, late_frame=400):
        """Create side-by-side comparison"""
        print(f"📊 Creating comparison: Frame {early_frame} vs Frame {late_frame}")
        
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
            
            # Add titles with smaller font
            cv2.putText(combined, f"Early Frame {early_frame}", (50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(combined, f"Late Frame {late_frame}", 
                       (early_vis.shape[1] + 50, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            comparison_path = os.path.join(self.visualization_path, "comparison_early_vs_late.tif")
            cv2.imwrite(comparison_path, combined)
            print(f"✅ Saved comparison: {comparison_path}")
        else:
            print("❌ Failed to create comparison visualization")
    
    def create_summary_report(self):
        """Create summary image"""
        print("📋 Creating summary image...")
        
        summary_img = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(summary_img, f"iPS Detection Summary - FOV {self.fov}", 
                   (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Statistics
        if self.labelled_df is not None:
            total_cells = len(self.labelled_df['Cell ID'].unique())
            ips_count = len(self.ips_cells)
            ips_percentage = (ips_count / total_cells) * 100 if total_cells > 0 else 0
            
            y_pos = 120
            stats = [
                f"Method: WHITE Preprocessed Cell Detection",
                f"Total Cells: {total_cells}",
                f"iPS Cells: {ips_count}",
                f"iPS Percentage: {ips_percentage:.1f}%",
                f"Tracking Data: {len(self.tracking_info)} cells",
                f"Frames Visualized: 0-400 (all frames)",
                f"Output Format: 6-digit .tif files",
                f"Overlay: Preprocessed GREEN + WHITE cells",
                f"Features: Full segmentation + trajectories + preprocessing"
            ]
            
            for stat in stats:
                cv2.putText(summary_img, stat, (70, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_pos += 35
        
        summary_path = os.path.join(self.visualization_path, "detection_summary.tif")
        cv2.imwrite(summary_path, summary_img)
        print(f"✅ Saved summary: {summary_path}")
    
    def run_visualization(self, visualize_all_frames=True, key_frames=None):
        """Run complete visualization pipeline"""
        print("="*70)
        print(f"🎨 iPS VISUALIZATION WITH PREPROCESSING - FOV {self.fov}")
        print("🔧 FIXES: OpenCV contour errors + better error handling")
        print("🖼️ FIXES: Robust image loading for all TIFF formats")
        print("🔬 NEW: Uses preprocessed GREEN background + WHITE cells")
        print("🎯 UPDATED: Yellow bold fonts for frame info and iPS counts")
        print("="*70)
        
        if self.labelled_df is None:
            print("❌ No labelled data found. Run labeling system first!")
            return False
        
        # Create frame visualizations
        if visualize_all_frames:
            print("\n🎬 Creating ALL FRAME visualizations (0-400) with trajectories...")
            successful, failed = self.create_all_frame_visualizations(start_frame=0, end_frame=400)
            if failed > 0:
                print(f"⚠️ Warning: {failed} frames failed to render")
        
        # Create comparison visualization
        print("\n📊 Creating comparison visualization...")
        self.create_comparison_visualization()
        
        # Create summary report
        print("\n📋 Creating summary report...")
        self.create_summary_report()
        
        print(f"\n✅ Visualization completed!")
        print(f"📁 Results saved in: {self.visualization_path}")
        print(f"🔬 Overlay: Preprocessed GREEN background + WHITE cells")
        print(f"🎯 Frame info: Yellow bold fonts")
        
        return True


def visualize_single_fov_proper(base_path, fov, visualize_all_frames=True, key_frames=None):
    """Visualize single FOV with working trajectory method"""
    try:
        visualizer = iPSVisualizerProper(base_path, fov)
        return visualizer.run_visualization(visualize_all_frames, key_frames)
    except Exception as e:
        print(f"❌ Error visualizing FOV {fov}: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_all_fovs_proper(base_path=".", visualize_all_frames=True, key_frames=None):
    """Visualize all FOVs with working trajectory method"""
    nuclear_dataset_dir = os.path.join(base_path, "nuclear_dataset")
    
    if not os.path.exists(nuclear_dataset_dir):
        print(f"❌ Directory not found: {nuclear_dataset_dir}")
        return
    
    # Find FOVs with labelled data
    fov_dirs = []
    for d in os.listdir(nuclear_dataset_dir):
        if os.path.isdir(os.path.join(nuclear_dataset_dir, d)) and d.isdigit():
            fov_num = int(d)
            if 2 <= fov_num <= 54:
                labelled_path = os.path.join(nuclear_dataset_dir, d, "Labelled", "arranged_retrospective_labels.csv")
                if os.path.exists(labelled_path):
                    fov_dirs.append(d)
    
    fov_dirs = sorted(fov_dirs, key=int)
    print(f"🔍 Found {len(fov_dirs)} FOVs with labelled data")
    
    successful = 0
    for fov in fov_dirs:
        print(f"\n{'='*50}")
        print(f"🎨 Visualizing FOV {fov}")
        print(f"{'='*50}")
        
        success = visualize_single_fov_proper(base_path, fov, visualize_all_frames, key_frames)
        if success:
            successful += 1
            print(f"✅ FOV {fov} visualization completed")
        else:
            print(f"❌ FOV {fov} visualization failed")
    
    print(f"\n{'='*70}")
    print("🎬 ALL FOVs VISUALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully visualized: {successful}/{len(fov_dirs)} FOVs")
    print(f"✨ Each FOV: 0-400 frames with trajectories")
    print(f"📁 Format: 6-digit .tif files (000000.tif - 000400.tif)")
    print(f"🎨 Features: Colored cells + blue iPS boxes + red trajectories")
    print(f"🔬 Overlay: Preprocessed GREEN background + WHITE cells")
    print(f"🖼️ Robust: Handles all TIFF format variants")
    print(f"🎯 Font: Yellow bold for frame info and iPS counts")


if __name__ == "__main__":
    # Test visualization with trajectory lines and preprocessed overlay
    print("🎬 Testing: visualization with trajectory lines and preprocessed overlay!")
    print("✨ Features:")
    print("   - Red trajectory lines showing cell movement")
    print("   - Yellow bold fonts for frame info and iPS counts")
    print("   - Blue bounding boxes for iPS cells")
    print("   - Full colored cell segmentation")
    print("   - All frames 0-400 as 000000.tif to 000400.tif")
    print("   - Robust image loading for all TIFF formats")
    print("   - Preprocessed GREEN background + WHITE cells overlay")
    visualize_single_fov_proper(".", "2", visualize_all_frames=True)
