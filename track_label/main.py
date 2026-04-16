#!/usr/bin/env python3
"""
Main Pipeline for iPS Cell Detection with Consensus Data

CONSENSUS WORKFLOW MODIFICATIONS:
1. PREPROCESSING: Uses KIT-GE reference model for images (test, green_signal)
2. Preserves ground truth blue markers from consensus first frame
3. Segmentation-based detection from consensus last frames 
4. BACKWARD TRACING: Uses consensus lineage data
5. FORWARD TRACING: Uses consensus lineage data  
6. FALLBACK MODE: Spatial tracking when no consensus parent-child relationships
7. CROSS-VALIDATION: Compare backward vs forward results on consensus data
8. Visualization with consensus lineage overlay
9. Robust image loading for all TIFF formats
10. Comprehensive error handling and validation

CONSENSUS DATA STRUCTURE:
- Input: consensus/FOV/consensus_data.csv (instead of track_result/arranged.csv)
- Lineage: consensus/FOV/res_track.txt (instead of track_result/res_track.txt) 
- Images: KIT-GE/nuclear_dataset/FOV/test/ and green_signal/ (reference model)
- Output: consensus/FOV/Labelled/ (all results saved in consensus)

Usage:
    python main.py --fov 2                      # Process single FOV with consensus tracking
    python main.py --all                        # Process all FOVs with consensus tracking
    python main.py --visualize-only --fov 2     # Only visualize consensus
    python main.py --complete-tracking --fov 2  # Only consensus tracking
    python main.py --preprocess-only --fov 2    # Only preprocessing from KIT-GE reference
"""

import argparse
import sys
import os

def check_consensus_prerequisites(base_path: str, fov: str = None):
    """Check if required consensus files and folders exist"""
    issues = []
    warnings = []
    
    consensus_dir = os.path.join(base_path, "consensus")
    if not os.path.exists(consensus_dir):
        issues.append(f"consensus directory not found at {consensus_dir}")
        return issues, warnings
    
    # Check KIT-GE reference model
    kit_ge_dir = os.path.join(base_path, "KIT-GE", "nuclear_dataset")
    if not os.path.exists(kit_ge_dir):
        issues.append(f"KIT-GE reference model not found at {kit_ge_dir}")
        return issues, warnings
    
    # Check specific FOV or all FOVs in consensus
    if fov:
        fov_dirs = [fov]
    else:
        fov_dirs = [d for d in os.listdir(consensus_dir) 
                   if os.path.isdir(os.path.join(consensus_dir, d)) and d.isdigit()]
    
    for fov_dir in fov_dirs:
        consensus_fov_path = os.path.join(consensus_dir, fov_dir)
        kit_ge_fov_path = os.path.join(kit_ge_dir, fov_dir)
        
        # Check consensus folder structure
        if not os.path.exists(consensus_fov_path):
            issues.append(f"FOV {fov_dir}: Missing consensus folder")
            continue
        
        # Check KIT-GE reference folders
        required_reference_folders = ["test", "green_signal"]
        for folder in required_reference_folders:
            folder_path = os.path.join(kit_ge_fov_path, folder)
            if not os.path.exists(folder_path):
                issues.append(f"FOV {fov_dir}: Missing KIT-GE {folder} folder")
            else:
                # Check if folders have image files
                files = [f for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.tiff')]
                if not files:
                    other_formats = [f for f in os.listdir(folder_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    if other_formats:
                        warnings.append(f"FOV {fov_dir}: KIT-GE {folder} contains {len(other_formats)} images in non-TIFF format")
                    else:
                        issues.append(f"FOV {fov_dir}: KIT-GE {folder} folder has no supported image files")
                else:
                    # Test image loading
                    test_file = os.path.join(folder_path, files[0])
                    try:
                        import cv2
                        test_img = cv2.imread(test_file, -1)
                        if test_img is None:
                            warnings.append(f"FOV {fov_dir}: KIT-GE {folder} images may be in unsupported format")
                        else:
                            print(f" FOV {fov_dir}: KIT-GE {folder} OK ({len(files)} .tif files)")
                    except Exception as e:
                        warnings.append(f"FOV {fov_dir}: Error testing KIT-GE image format in {folder}: {e}")
        
        # Check consensus_data.csv
        consensus_data_path = os.path.join(consensus_fov_path, "consensus_data.csv")
        if not os.path.exists(consensus_data_path):
            issues.append(f"FOV {fov_dir}: Missing consensus_data.csv")
        else:
            try:
                import pandas as pd
                df = pd.read_csv(consensus_data_path)
                required_columns = ['Cell ID', 'Label', 'Frame', 'X', 'Y']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    issues.append(f"FOV {fov_dir}: consensus_data.csv missing columns: {missing_cols}")
                
                # Check for manual annotations (blue markers)
                manual_ips = len(df[df['Label'] == 1])
                print(f" FOV {fov_dir}: consensus_data.csv OK ({len(df)} records, {manual_ips} blue markers)")
            except Exception as e:
                issues.append(f"FOV {fov_dir}: Error reading consensus_data.csv: {e}")
        
        # Check consensus res_track.txt
        consensus_res_track_path = os.path.join(consensus_fov_path, "res_track.txt")
        if not os.path.exists(consensus_res_track_path):
            issues.append(f"FOV {fov_dir}: Missing consensus res_track.txt (required for consensus tracking)")
        else:
            try:
                with open(consensus_res_track_path, 'r') as f:
                    lines = f.readlines()
                if len(lines) < 10:
                    warnings.append(f"FOV {fov_dir}: consensus res_track.txt seems too small ({len(lines)} lines)")
                else:
                    # Count parent-child relationships
                    parent_count = 0
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 4 and int(parts[3]) != 0:
                            parent_count += 1
                    
                    if parent_count > 0:
                        print(f" FOV {fov_dir}: consensus res_track.txt OK ({len(lines)} lines, {parent_count} parent-child relationships)")
                    else:
                        print(f"️ FOV {fov_dir}: consensus res_track.txt OK ({len(lines)} lines, {parent_count} parent-child relationships)")
                        warnings.append(f"FOV {fov_dir}: No consensus parent-child relationships - will use spatial tracking fallback")
            except Exception as e:
                warnings.append(f"FOV {fov_dir}: Error reading consensus res_track.txt: {e}")
    
    return issues, warnings


def run_consensus_pipeline_single_fov(base_path: str, fov: str, detection_start_frame: int = 350, 
                                    detection_end_frame: int = 400, skip_preprocess: bool = False):
    """
    Run complete consensus pipeline for a single FOV
    """
    print("="*100)
    print(f" CONSENSUS iPSC PIPELINE - FOV {fov}")
    print("="*100)
    print("CONSENSUS WORKFLOW IMPLEMENTATION:")
    print("KIT-GE reference model for images (test, green_signal)")
    print("Ground truth from consensus first frame")
    print("Segmented cell detection from consensus last frames")
    print("BACKWARD TRACING: Using consensus lineage data")
    print("FORWARD TRACING: Using consensus lineage data")
    print("CONSENSUS SPATIAL FALLBACK: When no consensus lineage available")
    print("CROSS-VALIDATION: Compare both methods on consensus")
    print("COMPREHENSIVE LABELING: All validated consensus lineages")
    print("Visualization with consensus tracking")
    print("Robust image loading for all TIFF formats")
    print("="*100)
    
    # Import modules
    try:
        from preprocess import preprocess_consensus_single_fov
        from labeling import label_single_fov_consensus_tracking
        from visualization import visualize_single_fov_consensus 
    except ImportError as e:
        print(f" Error importing modules: {e}")
        return False
    
    success = True
    
    try:
        # Step 1: Preprocessing (if not skipped)
        if not skip_preprocess:
            print(f"\n STEP 1: CONSENSUS PREPROCESSING - FOV {fov}")
            print("="*60)
            print(" Using KIT-GE reference model for images")
            print(" Processing from: KIT-GE/nuclear_dataset/{fov}/green_signal/")
            print("Saving to: consensus/{fov}/Preprocessed_green/")
            
            preprocess_success = preprocess_consensus_single_fov(base_path, fov)
            
            if preprocess_success:
                print(f" Consensus preprocessing completed for FOV {fov}")
            else:
                print(f" Consensus preprocessing failed for FOV {fov}")
                success = False
        else:
            print(f"\n️ SKIPPING: Consensus preprocessing for FOV {fov}")
        
        # Step 2: Consensus tracking
        print(f"\n️ STEP 2: CONSENSUS TRACKING - FOV {fov}")
        print("="*60)
        print(" Using consensus_data.csv instead of arranged.csv")
        print(" Using consensus lineage data from consensus/res_track.txt")
        print("️ Using KIT-GE reference model for image validation")
        
        tracking_success, metadata = label_single_fov_consensus_tracking(
            fov, "KIT-GE", detection_start_frame, detection_end_frame
        )
        
        if tracking_success:
            print(f" Consensus tracking completed for FOV {fov}")
            if metadata:
                print(f" Results: {metadata.get('ips_lineage_count', 0)} iPS cells "
                      f"({metadata.get('ips_percentage', 0):.1f}%)")
        else:
            print(f" Consensus tracking failed for FOV {fov}")
            success = False
        
        # Step 3: Consensus visualization
        print(f"\n STEP 3: CONSENSUS VISUALIZATION - FOV {fov}")
        print("="*60)
        print(" Using consensus data for visualization")
        print("️ Using KIT-GE reference images")
        print(" Saving to: consensus/{fov}/Visualization/")
        
        viz_success = visualize_single_fov_consensus(base_path, fov, "KIT-GE")
        
        if viz_success:
            print(f" Consensus visualization completed for FOV {fov}")
        else:
            print(f" Consensus visualization failed for FOV {fov}")
            success = False
            
    except Exception as e:
        print(f" Error in consensus pipeline for FOV {fov}: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    return success


def run_consensus_pipeline_all_fovs(base_path: str, detection_start_frame: int = 350, 
                                   detection_end_frame: int = 400, skip_preprocess: bool = False):
    """Run consensus pipeline for all FOVs"""
    print("="*100)
    print(" CONSENSUS iPSC PIPELINE - ALL FOVs")
    print("="*100)
    
    # Find consensus FOVs
    consensus_dir = os.path.join(base_path, "consensus")
    if not os.path.exists(consensus_dir):
        print(f" Consensus directory not found: {consensus_dir}")
        return False
    
    fov_dirs = []
    for d in os.listdir(consensus_dir):
        if os.path.isdir(os.path.join(consensus_dir, d)) and d.isdigit():
            # Check if consensus_data.csv exists
            consensus_data_path = os.path.join(consensus_dir, d, "consensus_data.csv")
            if os.path.exists(consensus_data_path):
                fov_dirs.append(d)
    
    fov_dirs = sorted(fov_dirs, key=int)
    print(f" Found {len(fov_dirs)} consensus FOVs to process")
    
    successful = 0
    failed = 0
    
    for fov in fov_dirs:
        print(f"\n{'='*80}")
        print(f" Processing Consensus FOV {fov}")
        print(f"{'='*80}")
        
        try:
            success = run_consensus_pipeline_single_fov(
                base_path, fov, detection_start_frame, detection_end_frame, skip_preprocess
            )
            
            if success:
                successful += 1
                print(f" Consensus FOV {fov} completed successfully")
            else:
                failed += 1
                print(f" Consensus FOV {fov} failed")
                
        except Exception as e:
            failed += 1
            print(f" Error processing consensus FOV {fov}: {e}")
    
    print(f"\n{'='*100}")
    print(" CONSENSUS PIPELINE SUMMARY")
    print(f"{'='*100}")
    print(f" Successful: {successful}")
    print(f" Failed: {failed}")
    print(f" Success Rate: {successful}/{len(fov_dirs)} ({100*successful/len(fov_dirs) if fov_dirs else 0:.1f}%)")
    print(" CONSENSUS FEATURES APPLIED:")
    print("    consensus_data.csv input instead of arranged.csv")
    print("   consensus lineage data from consensus/res_track.txt")
    print("   ️ KIT-GE reference model for images and validation")
    print("    All results saved in consensus folder structure")
    
    return successful > 0


def main():
    """Main function for consensus workflow"""
    parser = argparse.ArgumentParser(description="iPS Cell Detection Pipeline with Consensus Data")
    parser.add_argument("--fov", type=str, help="Process specific FOV number (e.g., '2')")
    parser.add_argument("--all", action="store_true", help="Process all consensus FOVs")
    parser.add_argument("--visualize-only", action="store_true", help="Only run visualization on consensus")
    parser.add_argument("--complete-tracking", action="store_true", help="Only run consensus tracking")
    parser.add_argument("--preprocess-only", action="store_true", help="Only run consensus preprocessing")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--base-path", type=str, default=".", help="Base directory path (default: current directory)")
    parser.add_argument("--detection-start", type=int, default=350, help="Start frame for detection (default: 350)")
    parser.add_argument("--detection-end", type=int, default=400, help="End frame for detection (default: 400)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.fov and not args.all:
        print(" Error: Must specify either --fov <number> or --all")
        parser.print_help()
        sys.exit(1)
    
    if args.fov and args.all:
        print(" Error: Cannot specify both --fov and --all")
        sys.exit(1)
    
    # Check for conflicting options
    exclusive_options = [args.visualize_only, args.complete_tracking, args.preprocess_only]
    if sum(exclusive_options) > 1:
        print(" Error: Cannot specify multiple exclusive options")
        sys.exit(1)
    
    print(" Checking consensus prerequisites...")
    issues, warnings = check_consensus_prerequisites(args.base_path, args.fov)
    
    if issues:
        print(" Consensus prerequisites check failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n Make sure you have:")
        print("  - consensus/ folder with FOV directories")
        print("  - Each consensus FOV has: consensus_data.csv and res_track.txt")
        print("  - KIT-GE/nuclear_dataset/ with FOV folders containing test/ and green_signal/")
        print("  - consensus_data.csv with blue markers (Label=1)")
        sys.exit(1)
    
    if warnings:
        print("️ Warnings found:")
        for warning in warnings:
            print(f"  - {warning}")
        print("The consensus system will automatically handle these issues with fallback modes")
        print("Continuing with processing...")
    
    print(" Consensus prerequisites check passed")
    
    # Import check
    print(" Checking consensus modules...")
    try:
        if args.preprocess_only:
            from preprocess import preprocess_consensus_single_fov, preprocess_consensus_all_fovs
        elif args.visualize_only:
            # Use consensus visualization functions
            from visualization import visualize_single_fov_consensus, visualize_all_fovs_consensus
        elif args.complete_tracking:
            from labeling import label_single_fov_consensus_tracking, label_all_fovs_consensus_tracking
        else:
            from preprocess import preprocess_consensus_single_fov, preprocess_consensus_all_fovs
            from labeling import label_single_fov_consensus_tracking, label_all_fovs_consensus_tracking
            from visualization import visualize_single_fov_consensus, visualize_all_fovs_consensus  
        print(" Consensus modules loaded successfully")
    except ImportError as e:
        print(f" Error importing consensus modules: {e}")
        print(" Please make sure you have:")
        print("  - preprocess.py (with consensus preprocessing functions)")
        print("  - labeling.py (with consensus tracking functions)")
        print("  - visualization.py (with consensus visualization functions)")
        sys.exit(1)
    
    # Run consensus pipeline
    try:
        if args.fov:
            # Single FOV processing
            if args.preprocess_only:
                print(f" Running consensus preprocessing only for FOV {args.fov}")
                from preprocess import preprocess_consensus_single_fov
                success = preprocess_consensus_single_fov(args.base_path, args.fov)
            elif args.visualize_only:
                print(f" Running consensus visualization only for FOV {args.fov}")
                from visualization import visualize_single_fov_consensus
                success = visualize_single_fov_consensus(args.base_path, args.fov, "KIT-GE")
            elif args.complete_tracking:
                print(f"️ Running consensus tracking only for FOV {args.fov}")
                from labeling import label_single_fov_consensus_tracking
                success, _ = label_single_fov_consensus_tracking(args.fov, "KIT-GE", args.detection_start, args.detection_end)
            else:
                print(f" Running complete consensus pipeline for FOV {args.fov}")
                success = run_consensus_pipeline_single_fov(
                    args.base_path, args.fov, args.detection_start, args.detection_end, args.skip_preprocess
                )
            
            if success:
                print(f"\n Successfully completed consensus processing for FOV {args.fov}")
                print(f" Check results in: consensus/{args.fov}/")
                if not args.visualize_only:
                    print(" Consensus features applied:")
                    print("    KIT-GE reference model preprocessing")
                    print("    Ground truth from consensus first frame")
                    print("    Backward tracing using consensus lineage")
                    print("    Forward tracing using consensus lineage")
                    print("    Consensus spatial tracking fallback")
                    print("    Cross-validation of consensus results")
                    print("    consensus_data.csv input source")
                    print("    All outputs saved to consensus folder")
                if not args.complete_tracking and not args.preprocess_only:
                    print(" Consensus visualization features:")
                    print("    Complete consensus lineage trajectory logic")
                    print("    All frames 000000.tif to 000400.tif")
                    print("    Blue bounding boxes for consensus iPS cells")
                    print("    Bold text for frame info and labeling")
                    print("    Preprocessed GREEN background + WHITE cells overlay")
                    print("   ️ Robust image format handling")
            else:
                print(f"\n Consensus processing failed for FOV {args.fov}")
                sys.exit(1)
        
        else:  # args.all
            if args.preprocess_only:
                print(" Running consensus preprocessing for all FOVs")
                from preprocess import preprocess_consensus_all_fovs
                preprocess_consensus_all_fovs(args.base_path)
            elif args.visualize_only:
                print(" Running consensus visualization for all FOVs")
                from visualization import visualize_all_fovs_consensus
                visualize_all_fovs_consensus("KIT-GE")
            elif args.complete_tracking:
                print("️ Running consensus tracking for all FOVs")
                from labeling import label_all_fovs_consensus_tracking
                label_all_fovs_consensus_tracking("KIT-GE", args.detection_start, args.detection_end)
            else:
                print(" Running complete consensus pipeline for all FOVs")
                success = run_consensus_pipeline_all_fovs(
                    args.base_path, args.detection_start, args.detection_end, args.skip_preprocess
                )
                if not success:
                    sys.exit(1)
            
            print("\n Successfully completed consensus processing for all FOVs")
            print(" Check individual FOV results in consensus/")
            print(" Consensus features applied to all FOVs:")
            print("    KIT-GE reference model preprocessing")
            print("    Ground truth from consensus first frames")
            print("    Backward tracing using consensus lineage data")
            print("    Forward tracing using consensus lineage data")
            print("    Consensus spatial tracking fallback")
            print("    Cross-validation of consensus results")
            print("    consensus_data.csv input sources")
            print("   All outputs saved to consensus folders")
    
    except KeyboardInterrupt:
        print("\n️ Consensus processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error during consensus processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
