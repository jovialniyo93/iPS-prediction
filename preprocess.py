#!/usr/bin/env python3
"""
Green Fluorescence Preprocessing Pipeline - Only segment when cells are present
Based on the paper's methodology: CLAHE → Bilateral → Noise Removal → Thresholding

PIPELINE WITH CELL PRESENCE DETECTION:
1. Original green fluorescence
2. CLAHE (contrast enhancement) 
3. Bilateral filtering (noise suppression + edge preservation)
4. Noise removal (morphological operations)
5. Cell presence detection - only threshold if bright cells are detected
6. Thresholding (only when cells are present) - GREEN BACKGROUND + WHITE cells
7. If no cells detected, save as green background only

Usage:
    python preprocess.py --fov 2
    python preprocess.py --all
"""

import os
import cv2
import numpy as np
import argparse
from typing import Tuple, List
import sys

class GreenFluorescenceProcessor:
    """
    FIXED Green fluorescence preprocessing with intelligent cell detection
    Only performs thresholding when bright cells are actually present
    """
    
    def __init__(self, base_path: str, fov: str):
        self.base_path = base_path
        self.fov = fov
        
        # Paths
        self.fov_path = os.path.join(base_path, "nuclear_dataset", fov)
        self.green_signal_path = os.path.join(self.fov_path, "green_signal")
        
        # Output directories
        self.clahe_path = os.path.join(self.fov_path, "clahe")
        self.bilateral_filter_path = os.path.join(self.fov_path, "bilateral_filter")
        self.noise_removal_path = os.path.join(self.fov_path, "noise_removal")
        self.thresholded_path = os.path.join(self.fov_path, "thresholded")
        self.preprocessed_green_path = os.path.join(self.fov_path, "Preprocessed_green")
        
        # Create output directories
        for path in [self.clahe_path, self.bilateral_filter_path, 
                    self.noise_removal_path, self.thresholded_path, self.preprocessed_green_path]:
            os.makedirs(path, exist_ok=True)
    
    def safe_load_image(self, image_path: str) -> np.ndarray:
        """Safely load image with multiple fallback methods"""
        try:
            # Try OpenCV first
            img = cv2.imread(image_path, -1)
            if img is not None:
                return img
            
            # Try tifffile for complex TIFF formats
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
                print("💡 Install tifffile for better TIFF support: pip install tifffile")
            except Exception:
                pass
            
            print(f"❌ Could not load image: {image_path}")
            return None
            
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")
            return None
    
    def remove_uneven_illumination(self, img, blur_kernel_size=501):
        """EXACT copy from process_mask_2.py"""
        if blur_kernel_size % 2 == 0:
            blur_kernel_size = blur_kernel_size + 1
        img_f = img.astype(np.float32)
        img_mean = np.mean(img_f)
        img_blur = cv2.GaussianBlur(img_f, (blur_kernel_size, blur_kernel_size), 0)
        result = np.maximum(np.minimum((img_f - img_blur) + img_mean, 255), 0).astype(np.uint8)
        return result
    
    def binarization(self, img, threshold):
        """EXACT copy from process_mask_2.py"""
        img = np.uint8(img)
        _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        return img
    
    def useAreaFilter(self, img, area_size):
        """EXACT copy from process_mask_2.py"""
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_new = np.stack((img, img, img), axis=2)
        for cont in contours:
            area = cv2.contourArea(cont)
            if area < area_size:
                img_new = cv2.fillConvexPoly(img_new, cont, (0, 0, 0))
        img = img_new[:, :, 0]
        return img
    
    def median_filter(self, img, median_blur_size):
        """EXACT copy from process_mask_2.py"""
        if median_blur_size % 2 == 0:
            median_blur_size += 1
        img = cv2.medianBlur(img, median_blur_size)
        return img
    
    def detect_bright_cells(self, processed_img, min_brightness_threshold=80, min_cell_area=30):
        """
        NEW: Detect if there are actually bright cells present in the image
        Returns True if bright cells are detected, False otherwise
        """
        try:
            # Calculate image statistics
            mean_intensity = np.mean(processed_img)
            max_intensity = np.max(processed_img)
            std_intensity = np.std(processed_img)
            
            # Check if there's sufficient contrast and bright regions
            if max_intensity < min_brightness_threshold:
                return False, 0, "Max intensity too low"
            
            if std_intensity < 15:  # Very low variation suggests no cells
                return False, 0, "Low contrast - uniform background"
            
            # Try a conservative threshold to see if we have bright objects
            conservative_threshold = max(mean_intensity + 2 * std_intensity, min_brightness_threshold)
            conservative_threshold = min(conservative_threshold, 200)
            
            # Apply conservative threshold
            test_binary = self.binarization(processed_img, conservative_threshold)
            
            # Find contours to check for cell-like objects
            contours, _ = cv2.findContours(test_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return False, 0, "No bright regions found"
            
            # Count significant bright objects
            significant_objects = 0
            total_bright_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_cell_area:  # Minimum area for a cell
                    significant_objects += 1
                    total_bright_area += area
            
            # Calculate bright area ratio
            total_pixels = processed_img.shape[0] * processed_img.shape[1]
            bright_ratio = total_bright_area / total_pixels
            
            # Decision criteria
            has_cells = (significant_objects >= 1 and 
                        bright_ratio > 0.0005 and  # At least 0.05% bright area
                        bright_ratio < 0.4)        # But not too much (likely noise)
            
            reason = f"{significant_objects} objects, {bright_ratio:.3%} bright area"
            
            return has_cells, significant_objects, reason
            
        except Exception as e:
            print(f"❌ Error in cell detection: {e}")
            return False, 0, "Detection error"
    
    def create_green_background_from_processed(self, processed_gray):
        """Create GREEN background using processed grayscale results"""
        result = np.zeros((processed_gray.shape[0], processed_gray.shape[1], 3), dtype=np.uint8)
        result[:, :, 1] = processed_gray  # Green channel gets the processed version
        return result
    
    def create_green_background_with_white_cells(self, processed_gray, cell_mask):
        """Create GREEN background from processed image with WHITE cells"""
        result = np.zeros((processed_gray.shape[0], processed_gray.shape[1], 3), dtype=np.uint8)
        result[:, :, 1] = processed_gray  # Green channel shows processed background
        
        # Set cells to WHITE where mask is 255
        cell_pixels = cell_mask == 255
        result[cell_pixels] = [255, 255, 255]  # White cells
        
        return result
    
    def create_thresholded_green_background(self, processed_gray, cell_mask):
        """Create thresholded result showing processed background in GREEN + WHITE cells"""
        result = np.zeros((processed_gray.shape[0], processed_gray.shape[1], 3), dtype=np.uint8)
        result[:, :, 1] = processed_gray  # Green channel shows the processed background
        
        # Overlay WHITE cells on top
        cell_pixels = cell_mask == 255
        result[cell_pixels] = [255, 255, 255]  # White cells
        
        return result
    
    def process_single_image(self, image_path: str, output_filename: str) -> bool:
        """
        FIXED: Process single green fluorescence image with intelligent cell detection
        Only performs thresholding when bright cells are actually present
        """
        try:
            # Load original image
            original_img = self.safe_load_image(image_path)
            if original_img is None:
                return False
            
            print(f"   Processing: {output_filename}")
            
            # Convert to grayscale if needed
            if len(original_img.shape) > 2:
                gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = original_img.copy()
            
            # PROCESSING CHAIN - each step builds on previous
            
            # STEP 1: Apply CLAHE (Contrast-Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray_img)
            
            # Save CLAHE result with GREEN BACKGROUND
            clahe_green_result = self.create_green_background_from_processed(clahe_img)
            clahe_output = os.path.join(self.clahe_path, output_filename)
            cv2.imwrite(clahe_output, clahe_green_result)
            
            # STEP 2: Apply Bilateral Filtering ON CLAHE RESULT
            bilateral_img = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)
            
            # Save bilateral filter result with GREEN BACKGROUND
            bilateral_green_result = self.create_green_background_from_processed(bilateral_img)
            bilateral_output = os.path.join(self.bilateral_filter_path, output_filename)
            cv2.imwrite(bilateral_output, bilateral_green_result)
            
            # STEP 3: Noise Removal ON BILATERAL RESULT
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened_img = cv2.morphologyEx(bilateral_img, cv2.MORPH_OPEN, kernel_open)
            
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            noise_removed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel_close)
            
            noise_removed_img = self.median_filter(noise_removed_img, 3)
            
            # Save noise removal result with GREEN BACKGROUND
            noise_removal_green_result = self.create_green_background_from_processed(noise_removed_img)
            noise_removal_output = os.path.join(self.noise_removal_path, output_filename)
            cv2.imwrite(noise_removal_output, noise_removal_green_result)
            
            # STEP 4: NEW - Check if bright cells are present
            has_cells, cell_count, detection_reason = self.detect_bright_cells(noise_removed_img)
            
            if has_cells:
                # Cells detected - proceed with thresholding
                print(f"      ✅ Cells detected: {detection_reason}")
                
                # Calculate adaptive threshold
                mean_intensity = np.mean(noise_removed_img)
                std_intensity = np.std(noise_removed_img)
                
                threshold_value = mean_intensity + 1.5 * std_intensity
                threshold_value = min(max(threshold_value, 60), 200)
                
                # Apply thresholding to create binary mask
                binary_mask = self.binarization(noise_removed_img, threshold_value)
                
                # Apply area filtering and morphological cleanup
                binary_mask = self.useAreaFilter(binary_mask, 50)
                
                kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_final)
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_final)
                
                # Save thresholded result with GREEN BACKGROUND + WHITE cells
                thresholded_green_result = self.create_thresholded_green_background(noise_removed_img, binary_mask)
                thresholded_output = os.path.join(self.thresholded_path, output_filename)
                cv2.imwrite(thresholded_output, thresholded_green_result)
                
                # Create final result - GREEN processed background + WHITE cells
                final_result = self.create_green_background_with_white_cells(noise_removed_img, binary_mask)
                
                # Quality statistics
                total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
                white_pixels = np.sum(binary_mask == 255)
                cell_ratio = white_pixels / total_pixels
                
                if '000' in output_filename or output_filename.endswith('0.tif'):
                    print(f"      📊 Cell coverage: {cell_ratio:.1%} ({white_pixels} pixels)")
                    print(f"      🎯 Threshold used: {threshold_value:.1f}")
                
            else:
                # No cells detected - save as green background only
                print(f"      ⚪ No cells detected: {detection_reason}")
                
                # Create empty binary mask
                binary_mask = np.zeros_like(noise_removed_img)
                
                # Save thresholded result as pure green background (no white cells)
                thresholded_green_result = self.create_green_background_from_processed(noise_removed_img)
                thresholded_output = os.path.join(self.thresholded_path, output_filename)
                cv2.imwrite(thresholded_output, thresholded_green_result)
                
                # Final result is also pure green background
                final_result = self.create_green_background_from_processed(noise_removed_img)
            
            # Save final result
            preprocessed_output = os.path.join(self.preprocessed_green_path, output_filename)
            cv2.imwrite(preprocessed_output, final_result)
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {output_filename}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_all_images(self) -> Tuple[int, int]:
        """Process all green fluorescence images"""
        print(f"\n🔬 FIXED GREEN FLUORESCENCE PREPROCESSING - FOV {self.fov}")
        print("="*70)
        print("🎯 NEW FEATURE: Only segment when bright cells are actually present")
        print("📄 Processing pipeline:")
        print("1. 📸 Original green fluorescence")
        print("2. 🌟 CLAHE (contrast enhancement)")
        print("3. 🔧 Bilateral filtering (noise reduction)")
        print("4. 🧹 Noise removal (morphological operations)")
        print("5. 🔍 Cell presence detection (NEW)")
        print("6. 🔢 Thresholding ONLY if cells detected")
        print("7. 🎨 RESULT: GREEN background + WHITE cells (when present)")
        print("="*70)
        
        if not os.path.exists(self.green_signal_path):
            print(f"❌ Green signal folder not found: {self.green_signal_path}")
            return 0, 0
        
        green_files = sorted([f for f in os.listdir(self.green_signal_path) 
                             if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])
        
        if not green_files:
            print(f"❌ No image files found in {self.green_signal_path}")
            return 0, 0
        
        print(f"📁 Found {len(green_files)} green fluorescence images")
        
        successful = 0
        failed = 0
        images_with_cells = 0
        images_without_cells = 0
        
        for i, filename in enumerate(green_files):
            input_path = os.path.join(self.green_signal_path, filename)
            
            success = self.process_single_image(input_path, filename)
            
            if success:
                successful += 1
                # Check if cells were detected by examining the final result
                final_path = os.path.join(self.preprocessed_green_path, filename)
                final_img = self.safe_load_image(final_path)
                if final_img is not None:
                    # Check if there are any white pixels (cells)
                    if len(final_img.shape) > 2:
                        white_pixels = np.sum(np.all(final_img == [255, 255, 255], axis=2))
                        if white_pixels > 0:
                            images_with_cells += 1
                        else:
                            images_without_cells += 1
            else:
                failed += 1
            
            # Progress report
            if (i + 1) % 25 == 0 or (i + 1) == len(green_files):
                print(f"📈 Progress: {i + 1}/{len(green_files)} images processed")
        
        print(f"\n✅ FIXED PREPROCESSING COMPLETE!")
        print(f"📊 Results:")
        print(f"   ✅ Successfully processed: {successful} images")
        print(f"   ❌ Failed: {failed} images")
        print(f"   🔬 Images with cells detected: {images_with_cells}")
        print(f"   ⚪ Images without cells (green background only): {images_without_cells}")
        print(f"📁 Output folders:")
        print(f"   🌟 clahe/: CLAHE")
        print(f"   🔧 bilateral_filter/: Bilateral filtered")
        print(f"   🧹 noise_removal/: Noise removed")
        print(f"   🔢 thresholded/: Segmented (only when cells present)")
        print(f"   🎯 Preprocessed_green/: FINAL results")
        print(f"🔍 Smart detection prevents false cell segmentation!")
        
        return successful, failed
    
    def create_comparison_montage(self) -> bool:
        """Create comparison montage showing processing effects"""
        try:
            green_files = sorted([f for f in os.listdir(self.green_signal_path) 
                                 if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])
            
            if not green_files:
                return False
            
            # Find one image with cells and one without for comparison
            sample_with_cells = None
            sample_without_cells = None
            
            for filename in green_files[:min(10, len(green_files))]:
                final_path = os.path.join(self.preprocessed_green_path, filename)
                final_img = self.safe_load_image(final_path)
                if final_img is not None:
                    if len(final_img.shape) > 2:
                        white_pixels = np.sum(np.all(final_img == [255, 255, 255], axis=2))
                        if white_pixels > 0 and sample_with_cells is None:
                            sample_with_cells = filename
                        elif white_pixels == 0 and sample_without_cells is None:
                            sample_without_cells = filename
                
                if sample_with_cells and sample_without_cells:
                    break
            
            # Create montage for the sample with cells
            if sample_with_cells:
                sample_file = sample_with_cells
            else:
                sample_file = green_files[0]
            
            # Load all processing stages
            original_path = os.path.join(self.green_signal_path, sample_file)
            clahe_path = os.path.join(self.clahe_path, sample_file)
            bilateral_path = os.path.join(self.bilateral_filter_path, sample_file)
            noise_removal_path = os.path.join(self.noise_removal_path, sample_file)
            thresholded_path = os.path.join(self.thresholded_path, sample_file)
            final_path = os.path.join(self.preprocessed_green_path, sample_file)
            
            images = []
            labels = ["Original Green", "CLAHE ", "Bilateral Filtered", "Noise Removed", "Smart Threshold", "FINAL Result"]
            paths = [original_path, clahe_path, bilateral_path, noise_removal_path, thresholded_path, final_path]
            
            for path in paths:
                img = self.safe_load_image(path)
                if img is not None:
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img = cv2.resize(img, (250, 250))
                    images.append(img)
                else:
                    blank = np.zeros((250, 250, 3), dtype=np.uint8)
                    images.append(blank)
            
            # Create montage
            if len(images) == 6:
                # Add labels to images
                for i, (img, label) in enumerate(zip(images, labels)):
                    cv2.putText(img, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(img, f"Step {i+1}", (5, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Arrange in 2 rows
                top_row = np.hstack((images[0], images[1], images[2]))
                bottom_row = np.hstack((images[3], images[4], images[5]))
                montage = np.vstack((top_row, bottom_row))
                
                # Add title
                title_height = 100
                title_img = np.zeros((title_height, montage.shape[1], 3), dtype=np.uint8)
                cv2.putText(title_img, f"FIXED: Smart Cell Detection - FOV {self.fov} - Sample: {sample_file}", 
                           (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(title_img, "CLAHE → Bilateral → Noise Removal → Smart Detection → Threshold (if cells present)", 
                           (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(title_img, "Only segments when bright cells are actually detected", 
                           (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(title_img, "Prevents false segmentation on uniform green backgrounds", 
                           (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
                
                final_montage = np.vstack((title_img, montage))
                
                # Save montage
                montage_path = os.path.join(self.fov_path, "fixed_smart_detection_comparison.png")
                cv2.imwrite(montage_path, final_montage)
                
                print(f"✅ Smart detection comparison montage saved: {montage_path}")
                return True
            
        except Exception as e:
            print(f"❌ Error creating comparison montage: {e}")
        
        return False


def preprocess_single_fov(base_path: str, fov: str) -> bool:
    """Preprocess green fluorescence for a single FOV"""
    try:
        processor = GreenFluorescenceProcessor(base_path, fov)
        successful, failed = processor.process_all_images()
        
        # Create comparison montage
        processor.create_comparison_montage()
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Error preprocessing FOV {fov}: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_all_fovs(base_path: str = ".") -> None:
    """Preprocess green fluorescence for all FOVs"""
    nuclear_dataset_dir = os.path.join(base_path, "nuclear_dataset")
    
    if not os.path.exists(nuclear_dataset_dir):
        print(f"❌ Directory not found: {nuclear_dataset_dir}")
        return
    
    # Find FOV directories with green_signal
    fov_dirs = []
    for d in os.listdir(nuclear_dataset_dir):
        if os.path.isdir(os.path.join(nuclear_dataset_dir, d)) and d.isdigit():
            fov_num = int(d)
            if 2 <= fov_num <= 54:
                green_signal_path = os.path.join(nuclear_dataset_dir, d, "green_signal")
                if os.path.exists(green_signal_path):
                    fov_dirs.append(d)
    
    fov_dirs = sorted(fov_dirs, key=int)
    print(f"🔍 Found {len(fov_dirs)} FOVs with green signal data")
    
    successful_fovs = 0
    total_with_cells = 0
    total_without_cells = 0
    
    for fov in fov_dirs:
        print(f"\n{'='*60}")
        print(f"🔬 Preprocessing FOV {fov}")
        print(f"{'='*60}")
        
        success = preprocess_single_fov(base_path, fov)
        if success:
            successful_fovs += 1
            print(f"✅ FOV {fov} preprocessing completed")
        else:
            print(f"❌ FOV {fov} preprocessing failed")
    
    print(f"\n{'='*70}")
    print("🔬 FIXED GREEN FLUORESCENCE PREPROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successfully preprocessed: {successful_fovs}/{len(fov_dirs)} FOVs")
    print(f"🎯 NEW FEATURE: Smart cell detection prevents false segmentation")
    print(f"📄 Used intelligent methodology:")
    print(f"   Original → CLAHE → Bilateral → Noise Removal → Cell Detection → Threshold (if cells present)")
    print(f"📁 Results saved in respective FOV folders")
    print(f"🔍 Images without cells saved as pure green background")
    print(f"⚪ Images with cells show WHITE cells on GREEN background")
    print(f"✅ No more false segmentation on uniform backgrounds!")
    print(f"💡 Ready for segmentation-only retrospective labeling!")


def main():
    """Main function for FIXED green background preprocessing script"""
    parser = argparse.ArgumentParser(description="FIXED Green Background Preprocessing Pipeline with Smart Cell Detection")
    parser.add_argument("--fov", type=str, help="Process specific FOV number (e.g., '2')")
    parser.add_argument("--all", action="store_true", help="Process all FOVs (2-54)")
    parser.add_argument("--base-path", type=str, default=".", help="Base directory path")
    
    args = parser.parse_args()
    
    if not args.fov and not args.all:
        print("❌ Error: Must specify either --fov <number> or --all")
        parser.print_help()
        sys.exit(1)
    
    if args.fov and args.all:
        print("❌ Error: Cannot specify both --fov and --all")
        sys.exit(1)
    
    print("🔬 FIXED GREEN BACKGROUND PREPROCESSING PIPELINE")
    print("="*70)
    print("🎯 NEW: Smart cell detection - only segments when cells are present")
    print("📄 Intelligent processing methodology:")
    print("   Original → CLAHE → Bilateral → Noise Removal → Cell Detection → Threshold (conditional)")
    print("✅ Key Fix: Prevents false segmentation on uniform green backgrounds")
    print("🔍 Like frame 000275.tif - will show pure green background if no cells detected")
    print("⚪ Only segments bright cells when they are actually present")
    print("="*70)
    
    try:
        if args.fov:
            print(f"🔬 Preprocessing FOV {args.fov}")
            success = preprocess_single_fov(args.base_path, args.fov)
            if success:
                print(f"\n🎉 Successfully preprocessed FOV {args.fov}")
                print(f"📁 Check results in: nuclear_dataset/{args.fov}/")
                print("🎯 Result: Smart detection - only segments when cells present!")
            else:
                print(f"\n❌ Preprocessing failed for FOV {args.fov}")
                sys.exit(1)
        else:
            print("🔬 Preprocessing all FOVs")
            preprocess_all_fovs(args.base_path)
    
    except KeyboardInterrupt:
        print("\n⚠️ Preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
