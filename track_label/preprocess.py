#!/usr/bin/env python3
"""
Green Fluorescence Preprocessing Pipeline for Consensus Workflow

CONSENSUS MODIFICATIONS:
1. REFERENCE MODEL: Uses KIT-GE as reference for original images
2. INPUT SOURCE: KIT-GE/nuclear_dataset/FOV/green_signal/
3. OUTPUT TARGET: consensus/FOV/Preprocessed_green/
4. VALIDATION IMAGES: Uses KIT-GE images for preprocessing validation
5. CONSENSUS INTEGRATION: Designed to work with consensus labeling system

PIPELINE WITH CELL PRESENCE DETECTION:
1. Original green fluorescence from KIT-GE reference model
2. CLAHE (contrast enhancement) 
3. Bilateral filtering (noise suppression + edge preservation)
4. Noise removal (morphological operations)
5. Cell presence detection - only threshold if bright cells are detected
6. Thresholding (only when cells are present) - GREEN BACKGROUND + WHITE cells
7. If no cells detected, save as green background only
8. Save results to consensus folder structure

Usage:
    python preprocess.py --consensus-fov 2
    python preprocess.py --consensus-all
"""

import os
import cv2
import numpy as np
import argparse
from typing import Tuple, List
import sys

class ConsensusGreenFluorescenceProcessor:
    """
    Consensus Green fluorescence preprocessing with KIT-GE reference model
    Uses KIT-GE images as source, saves to consensus folder structure
    """
    
    def __init__(self, base_path: str, fov: str, reference_model: str = "KIT-GE"):
        self.base_path = base_path
        self.fov = fov
        self.reference_model = reference_model
        
        # Input paths (from KIT-GE reference model)
        self.reference_fov_path = os.path.join(base_path, reference_model, "nuclear_dataset", fov)
        self.green_signal_path = os.path.join(self.reference_fov_path, "green_signal")
        
        # Output paths (to consensus folder)
        self.consensus_fov_path = os.path.join(base_path, "consensus", fov)
        self.clahe_path = os.path.join(self.consensus_fov_path, "clahe")
        self.bilateral_filter_path = os.path.join(self.consensus_fov_path, "bilateral_filter")
        self.noise_removal_path = os.path.join(self.consensus_fov_path, "noise_removal")
        self.thresholded_path = os.path.join(self.consensus_fov_path, "thresholded")
        self.preprocessed_green_path = os.path.join(self.consensus_fov_path, "Preprocessed_green")
        
        # Create output directories
        for path in [self.clahe_path, self.bilateral_filter_path, 
                    self.noise_removal_path, self.thresholded_path, self.preprocessed_green_path]:
            os.makedirs(path, exist_ok=True)
        
        print(f"Consensus preprocessing setup for FOV {fov}")
        print(f"Input source: {self.green_signal_path}")
        print(f"Output target: {self.preprocessed_green_path}")
    
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
                print("Install tifffile for better TIFF support: pip install tifffile")
            except Exception:
                pass
            
            print(f"Could not load image: {image_path}")
            return None
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
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
        Detect if there are actually bright cells present in the image
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
            print(f" Error in cell detection: {e}")
            return False, 0, "Detection error"
    
    def create_green_background_from_processed(self, processed_gray):
        """Create GREEN background using processed grayscale results"""
        result = np.zeros((processed_gray.shape[0], processed_gray.shape[1], 3), dtype=np.uint8)
        result[:, :, 1] = processed_gray  # Green channel gets the processed version
        return result
    
    def create_green_background_with_white_cells(self, processed_gray, cell_mask):
        """Create GREEN background from processed image with WHITE cells"""
        result = np.zeros((processed_gray.shape[0], processed_gray.shape[1], 3), dtype=np.uint8)
        
        # Start with processed gray as green background
        result[:, :, 1] = processed_gray  # Green channel shows processed background
        
        # Set cells to WHITE where mask is 255 (same as preprocess__.py)
        cell_pixels = cell_mask == 255
        result[cell_pixels] = [255, 255, 255]  # White cells
        
        return result
    
    def process_single_image(self, image_path: str) -> bool:
        """Process a single consensus image with smart cell detection"""
        try:
            filename = os.path.basename(image_path)
            print(f" Processing consensus image: {filename}")
            
            # Load original image from KIT-GE
            original_img = self.safe_load_image(image_path)
            if original_img is None:
                print(f" Failed to load consensus image: {filename}")
                return False
            
            # Step 1: CLAHE enhancement
            if len(original_img.shape) == 3:
                original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = original_img.copy()
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(original_gray)
            
            # Save CLAHE result with GREEN BACKGROUND (same as preprocess__.py)
            clahe_green_result = self.create_green_background_from_processed(clahe_img)
            clahe_path = os.path.join(self.clahe_path, filename)
            cv2.imwrite(clahe_path, clahe_green_result)
            
            # Step 2: Bilateral filtering
            bilateral_img = cv2.bilateralFilter(clahe_img, 9, 75, 75)
            
            # Save bilateral filter result with GREEN BACKGROUND (same as preprocess__.py)
            bilateral_green_result = self.create_green_background_from_processed(bilateral_img)
            bilateral_path = os.path.join(self.bilateral_filter_path, filename)
            cv2.imwrite(bilateral_path, bilateral_green_result)
            
            # Step 3: Noise removal
            kernel = np.ones((3, 3), np.uint8)
            noise_removed = cv2.morphologyEx(bilateral_img, cv2.MORPH_OPEN, kernel)
            noise_removed = cv2.morphologyEx(noise_removed, cv2.MORPH_CLOSE, kernel)
            
            # Save noise removal result with GREEN BACKGROUND (same as preprocess__.py)
            noise_removal_green_result = self.create_green_background_from_processed(noise_removed)
            noise_removal_path = os.path.join(self.noise_removal_path, filename)
            cv2.imwrite(noise_removal_path, noise_removal_green_result)
            
            # Step 4: Smart cell detection
            has_cells, cell_count, reason = self.detect_bright_cells(noise_removed)
            
            if has_cells:
                print(f" Cells detected: {reason}")
                
                # Apply thresholding since cells are present
                mean_intensity = np.mean(noise_removed)
                std_intensity = np.std(noise_removed)
                
                # Dynamic thresholding
                threshold_value = max(mean_intensity + 1.5 * std_intensity, 80)
                threshold_value = min(threshold_value, 180)
                
                # Apply threshold
                cell_mask = self.binarization(noise_removed, threshold_value)
                
                # Apply area filter to remove small noise
                cell_mask = self.useAreaFilter(cell_mask, 30)
                
                # Apply median filter for smoothing
                cell_mask = self.median_filter(cell_mask, 3)
                
                # Create final result: GREEN background + WHITE cells
                final_result = self.create_green_background_with_white_cells(noise_removed, cell_mask)
                
                # Save thresholded result with GREEN BACKGROUND + WHITE cells (same as preprocess__.py)
                thresholded_green_result = self.create_green_background_with_white_cells(noise_removed, cell_mask)
                thresholded_path = os.path.join(self.thresholded_path, filename)
                cv2.imwrite(thresholded_path, thresholded_green_result)
                
            else:
                print(f"   No cells detected: {reason} - using green background only")
                
                # Create pure green background (no thresholding)
                final_result = self.create_green_background_from_processed(noise_removed)
                
                # Save thresholded result as pure green background (same as preprocess__.py)
                thresholded_green_result = self.create_green_background_from_processed(noise_removed)
                thresholded_path = os.path.join(self.thresholded_path, filename)
                cv2.imwrite(thresholded_path, thresholded_green_result)
            
            # Save final consensus result
            final_path = os.path.join(self.preprocessed_green_path, filename)
            cv2.imwrite(final_path, final_result)
            
            return True
            
        except Exception as e:
            print(f" Error processing consensus image {filename}: {e}")
            return False
    
    def process_all_images(self):
        """Process all images in the KIT-GE green_signal folder for consensus"""
        if not os.path.exists(self.green_signal_path):
            print(f" KIT-GE green signal folder not found: {self.green_signal_path}")
            return 0, 0
        
        # Get all TIFF files
        image_files = [f for f in os.listdir(self.green_signal_path) 
                      if f.lower().endswith(('.tif', '.tiff'))]
        
        if not image_files:
            print(f" No TIFF files found in KIT-GE green signal folder")
            return 0, 0
        
        image_files.sort()
        print(f" Found {len(image_files)} images in KIT-GE reference model")
        
        successful = 0
        failed = 0
        
        for i, filename in enumerate(image_files):
            print(f"\n [{i+1}/{len(image_files)}] Processing consensus: {filename}")
            
            image_path = os.path.join(self.green_signal_path, filename)
            
            if self.process_single_image(image_path):
                successful += 1
                print(f"   Success")
            else:
                failed += 1
                print(f"   Failed")
        
        print(f"\n Consensus preprocessing summary:")
        print(f"    Successful: {successful}")
        print(f"    Failed: {failed}")
        print(f"    KIT-GE reference: {self.green_signal_path}")
        print(f"    Consensus output: {self.preprocessed_green_path}")
        
        return successful, failed
    
    def create_consensus_comparison_montage(self):
        """Create comparison montage for consensus preprocessing"""
        try:
            # Get a sample file
            image_files = [f for f in os.listdir(self.green_signal_path) 
                          if f.lower().endswith(('.tif', '.tiff'))]
            if not image_files:
                return False
            
            # Use first available image
            sample_file = image_files[0]
            
            # Paths to each processing step
            original_path = os.path.join(self.green_signal_path, sample_file)
            clahe_path = os.path.join(self.clahe_path, sample_file)
            bilateral_path = os.path.join(self.bilateral_filter_path, sample_file)
            noise_removal_path = os.path.join(self.noise_removal_path, sample_file)
            thresholded_path = os.path.join(self.thresholded_path, sample_file)
            final_path = os.path.join(self.preprocessed_green_path, sample_file)
            
            images = []
            labels = ["Original (KIT-GE)", "CLAHE ", "Bilateral Filtered", "Noise Removed", "Smart Threshold", "CONSENSUS Result"]
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
                title_height = 120
                title_img = np.zeros((title_height, montage.shape[1], 3), dtype=np.uint8)
                cv2.putText(title_img, f"CONSENSUS: Smart Cell Detection - FOV {self.fov} - Sample: {sample_file}", 
                           (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(title_img, f"Source: {self.reference_model} → Target: consensus/", 
                           (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
                cv2.putText(title_img, "CLAHE → Bilateral → Noise Removal → Smart Detection → Threshold (if cells present)", 
                           (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(title_img, "Only segments when bright cells are actually detected in KIT-GE reference", 
                           (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(title_img, "Results saved to consensus folder for consensus labeling system", 
                           (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 255, 255), 2)
                
                final_montage = np.vstack((title_img, montage))
                
                # Save montage
                montage_path = os.path.join(self.consensus_fov_path, "consensus_smart_detection_comparison.png")
                cv2.imwrite(montage_path, final_montage)
                
                print(f" Consensus comparison montage saved: {montage_path}")
                return True
            
        except Exception as e:
            print(f" Error creating consensus comparison montage: {e}")
        
        return False


def preprocess_consensus_single_fov(base_path: str, fov: str, reference_model: str = "KIT-GE") -> bool:
    """Preprocess green fluorescence for a single consensus FOV using reference model"""
    try:
        processor = ConsensusGreenFluorescenceProcessor(base_path, fov, reference_model)
        successful, failed = processor.process_all_images()
        
        # Create comparison montage
        processor.create_consensus_comparison_montage()
        
        return successful > 0
        
    except Exception as e:
        print(f" Error preprocessing consensus FOV {fov}: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_consensus_all_fovs(base_path: str = ".", reference_model: str = "KIT-GE") -> None:
    """Preprocess green fluorescence for all consensus FOVs using reference model"""
    consensus_base_dir = os.path.join(base_path, "consensus")
    
    if not os.path.exists(consensus_base_dir):
        print(f" Consensus directory not found: {consensus_base_dir}")
        return
    
    # Find consensus FOVs that need preprocessing
    fov_dirs = []
    for d in os.listdir(consensus_base_dir):
        if os.path.isdir(os.path.join(consensus_base_dir, d)) and d.isdigit():
            # Check if corresponding KIT-GE data exists
            kit_ge_green_path = os.path.join(base_path, reference_model, "nuclear_dataset", d, "green_signal")
            if os.path.exists(kit_ge_green_path):
                fov_dirs.append(d)
    
    fov_dirs = sorted(fov_dirs, key=int)
    print(f" Found {len(fov_dirs)} consensus FOVs with {reference_model} reference data")
    
    successful_fovs = 0
    
    for fov in fov_dirs:
        print(f"\n{'='*60}")
        print(f" Consensus Preprocessing FOV {fov}")
        print(f"{'='*60}")
        
        success = preprocess_consensus_single_fov(base_path, fov, reference_model)
        if success:
            successful_fovs += 1
            print(f" Consensus FOV {fov} preprocessing completed")
        else:
            print(f" Consensus FOV {fov} preprocessing failed")
    
    print(f"\n{'='*70}")
    print(" CONSENSUS GREEN FLUORESCENCE PREPROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f" Successfully preprocessed: {successful_fovs}/{len(fov_dirs)} FOVs")
    print(f" Reference Model: {reference_model}")
    print(f" Input Source: {reference_model}/nuclear_dataset/FOV/green_signal/")
    print(f" Output Target: consensus/FOV/Preprocessed_green/")
    print(f" Smart cell detection prevents false segmentation")
    print(f" Methodology:")
    print(f"   Original → CLAHE → Bilateral → Noise Removal → Cell Detection → Threshold (if cells present)")
    print(f" Results saved in respective consensus FOV folders")
    print(f" Images without cells saved as pure green background")
    print(f" Images with cells show WHITE cells on GREEN background")
    print(f" Ready for consensus lineage labeling!")


# Legacy functions for compatibility (redirect to consensus versions)
def preprocess_single_fov(base_path: str, fov: str) -> bool:
    """Legacy function - redirects to consensus preprocessing"""
    print("️ Using legacy preprocess_single_fov - redirecting to consensus version")
    return preprocess_consensus_single_fov(base_path, fov)


def preprocess_all_fovs(base_path: str = ".") -> None:
    """Legacy function - redirects to consensus preprocessing"""
    print("️ Using legacy preprocess_all_fovs - redirecting to consensus version")
    preprocess_consensus_all_fovs(base_path)


def main():
    """Main function for consensus preprocessing script"""
    parser = argparse.ArgumentParser(description="Consensus Green Background Preprocessing Pipeline with KIT-GE Reference")
    parser.add_argument("--consensus-fov", type=str, help="Process specific consensus FOV number (e.g., '2')")
    parser.add_argument("--consensus-all", action="store_true", help="Process all consensus FOVs")
    parser.add_argument("--fov", type=str, help="Legacy option (redirects to consensus)")
    parser.add_argument("--all", action="store_true", help="Legacy option (redirects to consensus)")
    parser.add_argument("--base-path", type=str, default=".", help="Base directory path")
    parser.add_argument("--reference-model", type=str, default="KIT-GE", help="Reference model for images")
    
    args = parser.parse_args()
    
    # Handle legacy arguments
    if args.fov:
        print(" Legacy --fov detected, using consensus preprocessing")
        args.consensus_fov = args.fov
    if args.all:
        print(" Legacy --all detected, using consensus preprocessing")
        args.consensus_all = True
    
    if not args.consensus_fov and not args.consensus_all:
        print(" Error: Must specify either --consensus-fov <number> or --consensus-all")
        parser.print_help()
        sys.exit(1)
    
    if args.consensus_fov and args.consensus_all:
        print(" Error: Cannot specify both --consensus-fov and --consensus-all")
        sys.exit(1)
    
    print(" CONSENSUS GREEN BACKGROUND PREPROCESSING PIPELINE")
    print("="*70)
    print(f" Reference Model: {args.reference_model}")
    print(f" Input Source: {args.reference_model}/nuclear_dataset/FOV/green_signal/")
    print(f" Output Target: consensus/FOV/Preprocessed_green/")
    print(f" Smart cell detection - only segments when cells are present")
    print(f" Methodology:")
    print(f"   Original → CLAHE → Bilateral → Noise Removal → Cell Detection → Threshold (conditional)")
    print(f" Key Feature: Uses {args.reference_model} as reference for preprocessing consensus data")
    print("="*70)
    
    try:
        if args.consensus_fov:
            print(f" Consensus preprocessing FOV {args.consensus_fov}")
            success = preprocess_consensus_single_fov(args.base_path, args.consensus_fov, args.reference_model)
            if success:
                print(f"\n Successfully preprocessed consensus FOV {args.consensus_fov}")
                print(f" Check results in: consensus/{args.consensus_fov}/Preprocessed_green/")
                print(f" Reference: {args.reference_model} images used for preprocessing")
                print(f" Results: Saved to consensus folder for consensus labeling")
            else:
                print(f"\n Consensus preprocessing failed for FOV {args.consensus_fov}")
                sys.exit(1)
        else:
            print(f" Consensus preprocessing all FOVs")
            preprocess_consensus_all_fovs(args.base_path, args.reference_model)
    
    except KeyboardInterrupt:
        print("\n Consensus preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error during consensus preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
