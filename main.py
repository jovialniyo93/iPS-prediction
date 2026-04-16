#!/usr/bin/env python3
"""
Main Pipeline for iPS Cell Detection with Complete Lineage Tracking

COMPLETE ENHANCED IMPLEMENTATION:
1. 🔬 PREPROCESSING: Smart green fluorescence segmentation (WHITE cells on GREEN background)
2. 🔵 Preserves ground truth blue markers from first frame
3. 🔬 Segmentation-based detection from last frames (most reliable)
4. 🔙 BACKWARD TRACING: Last frame → First frame using parent relationships (when available)
5. 🔜 FORWARD TRACING: Blue markers → Last frame using child relationships (when available)
6. 🔄 FALLBACK MODE: Spatial tracking when no parent-child relationships exist
7. ✅ CROSS-VALIDATION: Compare backward vs forward results
8. 🎨 Enhanced visualization with complete lineage overlay
9. 🖼️ Robust image loading for all TIFF formats
10. 🛡️ Comprehensive error handling and validation

This pipeline addresses the critical issue of missing parent-child relationships:
1. 🔬 Detects if lineage data is available in res_track.txt
2. 🔄 Automatically falls back to spatial tracking when lineage data is missing
3. 🔬 Uses segmentation + spatial proximity instead of parent-child relationships
4. ✅ Still preserves blue markers as ground truth
5. 🎨 Creates visualizations with all available information

Usage:
    python main.py --fov 2                      # Process single FOV with enhanced tracking
    python main.py --all                        # Process all FOVs with enhanced tracking
    python main.py --visualize-only --fov 2     # Only visualize
    python main.py --complete-tracking --fov 2  # Only enhanced tracking
    python main.py --preprocess-only --fov 2    # Only preprocessing
"""

import argparse
import sys
import os

def check_prerequisites(base_path: str, fov: str = None):
    """Check if required files and folders exist with enhanced validation"""
    issues = []
    warnings = []
    
    nuclear_dataset_dir = os.path.join(base_path, "nuclear_dataset")
    if not os.path.exists(nuclear_dataset_dir):
        issues.append(f"nuclear_dataset directory not found at {nuclear_dataset_dir}")
        return issues, warnings
    
    # Check specific FOV or all FOVs
    if fov:
        fov_dirs = [fov]
    else:
        fov_dirs = [d for d in os.listdir(nuclear_dataset_dir) 
                   if os.path.isdir(os.path.join(nuclear_dataset_dir, d)) and d.isdigit()]
    
    for fov_dir in fov_dirs:
        fov_path = os.path.join(nuclear_dataset_dir, fov_dir)
        
        # Check required folders
        required_folders = ["test", "green_signal", "track_result"]
        for folder in required_folders:
            folder_path = os.path.join(fov_path, folder)
            if not os.path.exists(folder_path):
                issues.append(f"FOV {fov_dir}: Missing {folder} folder")
            else:
                # Check if folders have files
                files = [f for f in os.listdir(folder_path) if f.endswith('.tif') or f.endswith('.tiff')]
                if not files:
                    other_formats = [f for f in os.listdir(folder_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    if other_formats:
                        warnings.append(f"FOV {fov_dir}: {folder} contains {len(other_formats)} images in non-TIFF format")
                    else:
                        issues.append(f"FOV {fov_dir}: {folder} folder has no supported image files")
                else:
                    # Test image loading
                    test_file = os.path.join(folder_path, files[0])
                    try:
                        import cv2
                        test_img = cv2.imread(test_file, -1)
                        if test_img is None:
                            warnings.append(f"FOV {fov_dir}: {folder} images may be in unsupported format")
                        else:
                            print(f"✅ FOV {fov_dir}: {folder} OK ({len(files)} .tif files)")
                    except Exception as e:
                        warnings.append(f"FOV {fov_dir}: Error testing image format in {folder}: {e}")
        
        # Check arranged.csv
        arranged_path = os.path.join(fov_path, "track_result", "arranged.csv")
        if not os.path.exists(arranged_path):
            issues.append(f"FOV {fov_dir}: Missing arranged.csv")
        else:
            try:
                import pandas as pd
                df = pd.read_csv(arranged_path)
                required_columns = ['Cell ID', 'Label', 'Frame', 'X', 'Y']
                missing_cols = [col for col in required_columns if col not in df.columns]
                if missing_cols:
                    issues.append(f"FOV {fov_dir}: arranged.csv missing columns: {missing_cols}")
                
                # Check for manual annotations (blue markers)
                manual_ips = len(df[df['Label'] == 1])
                print(f"✅ FOV {fov_dir}: arranged.csv OK ({len(df)} records, {manual_ips} blue markers)")
            except Exception as e:
                issues.append(f"FOV {fov_dir}: Error reading arranged.csv: {e}")
        
        # Check res_track.txt (analyze lineage data)
        res_track_path = os.path.join(fov_path, "track_result", "res_track.txt")
        if not os.path.exists(res_track_path):
            issues.append(f"FOV {fov_dir}: Missing res_track.txt (required for tracking)")
        else:
            try:
                with open(res_track_path, 'r') as f:
                    lines = f.readlines()
                if len(lines) < 10:
                    warnings.append(f"FOV {fov_dir}: res_track.txt seems too small ({len(lines)} lines)")
                else:
                    # Count parent-child relationships
                    parent_count = 0
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 4 and int(parts[3]) != 0:
                            parent_count += 1
                    
                    if parent_count > 0:
                        print(f"✅ FOV {fov_dir}: res_track.txt OK ({len(lines)} lines, {parent_count} parent-child relationships)")
                    else:
                        print(f"⚠️ FOV {fov_dir}: res_track.txt OK ({len(lines)} lines, {parent_count} parent-child relationships)")
                        warnings.append(f"FOV {fov_dir}: No parent-child relationships - will use spatial tracking fallback")
            except Exception as e:
                warnings.append(f"FOV {fov_dir}: Error reading res_track.txt: {e}")
    
    return issues, warnings


def run_complete_pipeline_single_fov(base_path: str, fov: str, detection_start_frame: int = 350, 
                                    detection_end_frame: int = 400, skip_preprocess: bool = False):
    """
    Run complete pipeline for a single FOV with enhanced lineage tracking and fallback
    """
    print("="*100)
    print(f"🧬 ENHANCED iPS PIPELINE - COMPLETE TRACKING WITH FALLBACK - FOV {fov}")
    print("="*100)
    print("COMPLETE ENHANCED IMPLEMENTATION:")
    print("✅ 🔬 Smart green fluorescence preprocessing")
    print("✅ 🔵 Ground truth blue marker preservation")
    print("✅ 🔬 Segmented cell detection from last frames")
    print("✅ 🔙 BACKWARD TRACING: Last frame → First frame (when lineage available)")
    print("✅ 🔜 FORWARD TRACING: Blue markers → Last frame (when lineage available)")
    print("✅ 🔄 SPATIAL FALLBACK: When no parent-child relationships exist")
    print("✅ ✅ CROSS-VALIDATION: Compare both methods")
    print("✅ 🏷️ COMPREHENSIVE LABELING: All validated lineages")
    print("✅ 🎨 Enhanced visualization with complete tracking")
    print("✅ 🖼️ Robust image loading for all TIFF formats")
    print("="*100)
    
    # Import modules with correct names
    try:
        from preprocess import preprocess_single_fov
        from labeling import label_single_fov_complete_tracking  # FIXED: correct module name
        from visualization import visualize_single_fov_proper
    except ImportError as e:
        print(f"❌ Error: Required modules not found: {e}")
        print("💡 Please save the artifacts as:")
        print("  - preprocess.py (preprocessing system)")
        print("  - labeling.py (enhanced tracking system)")  # FIXED: correct module name
        print("  - visualization.py (visualization system)")
        return False
    
    # Step 1: Preprocessing
    if not skip_preprocess:
        print("\n" + "="*70)
        print("STEP 1: GREEN FLUORESCENCE PREPROCESSING")
        print("="*70)
        
        preprocess_success = preprocess_single_fov(base_path, fov)
        
        if not preprocess_success:
            print(f"❌ Preprocessing failed for FOV {fov}")
            return False
        
        print(f"✅ Preprocessing completed for FOV {fov}")
        print("📁 Created folders:")
        print("   🌟 clahe/ - CLAHE enhanced")
        print("   🔧 bilateral_filter/ - Bilateral filtered")
        print("   🧹 noise_removal/ - Noise removed")
        print("   🔢 thresholded/ - Smart segmentation")
        print("   🎯 Preprocessed_green/ - WHITE cells on GREEN background")
    else:
        print("\n" + "="*70)
        print("STEP 1: PREPROCESSING SKIPPED")
        print("="*70)
    
    # Step 2: Enhanced Tracking with Fallback
    print("\n" + "="*70)
    print("STEP 2: ENHANCED TRACKING WITH FALLBACK")
    print("="*70)
    
    success, metadata = label_single_fov_complete_tracking(base_path, fov, detection_start_frame, detection_end_frame)
    
    if not success:
        print(f"❌ Enhanced tracking failed for FOV {fov}")
        return False
    
    print(f"✅ Enhanced tracking completed for FOV {fov}")
    if metadata:
        method = metadata.get('method', 'unknown')
        has_lineage = metadata.get('has_lineage_data', False)
        
        print(f"📊 Results:")
        print(f"   🎯 Method: {method}")
        print(f"   🌳 Has lineage data: {has_lineage}")
        print(f"   🔵 Ground truth preserved: {len(metadata.get('ground_truth_blue_markers', []))}")
        print(f"   🔬 Segmented detected: {len(metadata.get('segmented_last_frame', []))}")
        print(f"   ✅ Final validated: {metadata.get('ips_lineage_count', 0)} ({metadata.get('ips_percentage', 0):.1f}%)")
        
        if has_lineage:
            print(f"   🔙 Backward traced: {metadata.get('backward_traced_lineages', 0)}")
            print(f"   🔜 Forward traced: {metadata.get('forward_traced_lineages', 0)}")
        else:
            print(f"   🔄 Spatial tracked: {metadata.get('spatial_tracked_lineages', 0)}")
    
    # Step 3: Enhanced Visualization
    print("\n" + "="*70)
    print("STEP 3: ENHANCED VISUALIZATION")
    print("="*70)
    
    viz_success = visualize_single_fov_proper(base_path, fov)
    
    if not viz_success:
        print(f"❌ Visualization failed for FOV {fov}")
        return False
    
    print(f"✅ Enhanced visualization completed for FOV {fov}")
    
    # Summary
    print("\n" + "="*100)
    print("ENHANCED PIPELINE SUMMARY")
    print("="*100)
    print(f"✅ FOV {fov} processing completed successfully!")
    print(f"📁 Preprocessed data: nuclear_dataset/{fov}/Preprocessed_green/")
    print(f"📁 Enhanced tracking data: nuclear_dataset/{fov}/Labelled/")
    print(f"📁 Enhanced visualizations: nuclear_dataset/{fov}/Visualization/")
    if metadata:
        method = metadata.get('method', 'unknown')
        print(f"📊 Final results:")
        print(f"   🎯 Tracking method: {method}")
        print(f"   🔵 Ground truth cells: {len(metadata.get('ground_truth_blue_markers', []))}")
        print(f"   🔬 Segmented cells: {len(metadata.get('segmented_last_frame', []))}")
        print(f"   ✅ Total iPS lineage: {metadata.get('ips_lineage_count', 0)}")
        print(f"   📈 Success rate: Enhanced tracking with automatic fallback!")
    print("🔧 Complete enhancements applied:")
    print("   ✅ 🔬 Smart green fluorescence preprocessing")
    print("   ✅ 🔙 Complete backward tracing (when lineage available)")
    print("   ✅ 🔜 Complete forward tracing (when lineage available)")
    print("   ✅ 🔄 Spatial tracking fallback (when no lineage)")
    print("   ✅ ✅ Cross-validation of tracing methods")
    print("   ✅ 🎨 Enhanced visualization with complete tracking")
    print("   ✅ 🖼️ Robust image loading for all TIFF variants")
    print("   ✅ 🛡️ Comprehensive error handling and validation")
    print("="*100)
    
    return True


def run_complete_pipeline_all_fovs(base_path: str = ".", detection_start_frame: int = 350, 
                                  detection_end_frame: int = 400, skip_preprocess: bool = False):
    """
    Run complete pipeline for all FOVs with enhanced tracking and fallback
    """
    print("="*100)
    print("🧬 ENHANCED iPS PIPELINE - COMPLETE TRACKING WITH FALLBACK - ALL FOVs")
    print("="*100)
    
    # Import modules with correct names
    try:
        from preprocess import preprocess_all_fovs
        from labeling import label_all_fovs_complete_tracking  # FIXED: correct module name
        from visualization import visualize_all_fovs_proper
    except ImportError as e:
        print(f"❌ Error: Required modules not found: {e}")
        print("💡 Please save the artifacts as separate Python files:")
        print("  - preprocess.py")
        print("  - labeling.py")  # FIXED: correct module name
        print("  - visualization.py")
        return False
    
    # Step 1: Preprocess all FOVs
    if not skip_preprocess:
        print("\n" + "="*70)
        print("STEP 1: PREPROCESSING ALL FOVs")
        print("="*70)
        
        preprocess_all_fovs(base_path)
    else:
        print("\n" + "="*70)
        print("STEP 1: PREPROCESSING SKIPPED")
        print("="*70)
    
    # Step 2: Enhanced Tracking for all FOVs
    print("\n" + "="*70)
    print("STEP 2: ENHANCED TRACKING ALL FOVs")
    print("="*70)
    
    label_all_fovs_complete_tracking(base_path, detection_start_frame, detection_end_frame)
    
    # Step 3: Enhanced Visualization for all FOVs
    print("\n" + "="*70)
    print("STEP 3: ENHANCED VISUALIZATION ALL FOVs")
    print("="*70)
    
    visualize_all_fovs_proper(base_path)
    
    print("\n" + "="*100)
    print("🎉 ALL FOVs ENHANCED PIPELINE COMPLETED")
    print("="*100)
    print("✨ All FOVs processed with complete enhancements:")
    print("   🔬 Smart green fluorescence preprocessing")
    print("   🔵 Ground truth blue marker preservation")
    print("   🔬 Segmented cell detection from last frames")
    print("   🔙 Complete backward tracing (when lineage available)")
    print("   🔜 Complete forward tracing (when lineage available)")
    print("   🔄 Spatial tracking fallback (when no lineage)")
    print("   ✅ Cross-validation of tracing methods")
    print("   🏷️ Comprehensive lineage labeling")
    print("   🎨 Enhanced visualization with complete tracking")
    print("   🔵 Blue bounding boxes for iPS cells")
    print("   📝 Bold text for frame info and labeling")
    print("   🖼️ Robust image format handling")
    print("   🛡️ Comprehensive error recovery")
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description="Enhanced iPS Cell Detection Pipeline - Complete Tracking with Fallback")
    parser.add_argument("--fov", type=str, help="Process specific FOV number (e.g., '2')")
    parser.add_argument("--all", action="store_true", help="Process all FOVs (2-54)")
    parser.add_argument("--visualize-only", action="store_true", help="Only run visualization")
    parser.add_argument("--complete-tracking", action="store_true", help="Only run enhanced tracking")
    parser.add_argument("--preprocess-only", action="store_true", help="Only run preprocessing")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--base-path", type=str, default=".", help="Base directory path (default: current directory)")
    parser.add_argument("--detection-start", type=int, default=350, help="Start frame for detection (default: 350)")
    parser.add_argument("--detection-end", type=int, default=400, help="End frame for detection (default: 400)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.fov and not args.all:
        print("❌ Error: Must specify either --fov <number> or --all")
        parser.print_help()
        sys.exit(1)
    
    if args.fov and args.all:
        print("❌ Error: Cannot specify both --fov and --all")
        sys.exit(1)
    
    # Check for conflicting options
    exclusive_options = [args.visualize_only, args.complete_tracking, args.preprocess_only]
    if sum(exclusive_options) > 1:
        print("❌ Error: Cannot specify multiple exclusive options")
        sys.exit(1)
    
    print("🔍 Checking prerequisites...")
    issues, warnings = check_prerequisites(args.base_path, args.fov)
    
    if issues:
        print("❌ Prerequisites check failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("\n💡 Make sure you have:")
        print("  - nuclear_dataset folder with FOV directories (2-54)")
        print("  - Each FOV has: test/, green_signal/, track_result/ folders")
        print("  - arranged.csv in track_result/ folder with blue markers (Label=1)")
        print("  - res_track.txt in track_result/ folder")
        sys.exit(1)
    
    if warnings:
        print("⚠️ Warnings found:")
        for warning in warnings:
            print(f"  - {warning}")
        print("🔄 The enhanced system will automatically handle these issues with fallback modes")
        print("Continuing with processing...")
    
    print("✅ Prerequisites check passed")
    
    # Import check with correct module names
    print("🔍 Checking modules...")
    try:
        if args.preprocess_only:
            from preprocess import preprocess_single_fov, preprocess_all_fovs
        elif args.visualize_only:
            from visualization import visualize_single_fov_proper, visualize_all_fovs_proper
        elif args.complete_tracking:
            from labeling import label_single_fov_complete_tracking, label_all_fovs_complete_tracking  # FIXED
        else:
            from preprocess import preprocess_single_fov, preprocess_all_fovs
            from labeling import label_single_fov_complete_tracking, label_all_fovs_complete_tracking  # FIXED
            from visualization import visualize_single_fov_proper, visualize_all_fovs_proper
        print("✅ Modules loaded successfully")
    except ImportError as e:
        print(f"❌ Error importing modules: {e}")
        print("💡 Please make sure you have:")
        print("  - preprocess.py (preprocessing system)")
        print("  - labeling.py (enhanced tracking system)")  # FIXED
        print("  - visualization.py (visualization system)")
        sys.exit(1)
    
    # Run pipeline
    try:
        if args.fov:
            # Single FOV processing
            if args.preprocess_only:
                print(f"🔬 Running preprocessing only for FOV {args.fov}")
                success = preprocess_single_fov(args.base_path, args.fov)
            elif args.visualize_only:
                print(f"🎨 Running visualization only for FOV {args.fov}")
                success = visualize_single_fov_proper(args.base_path, args.fov)
            elif args.complete_tracking:
                print(f"🏷️ Running enhanced tracking only for FOV {args.fov}")
                success, _ = label_single_fov_complete_tracking(args.base_path, args.fov, args.detection_start, args.detection_end)
            else:
                print(f"🚀 Running complete enhanced pipeline for FOV {args.fov}")
                success = run_complete_pipeline_single_fov(
                    args.base_path, args.fov, args.detection_start, args.detection_end, args.skip_preprocess
                )
            
            if success:
                print(f"\n🎉 Successfully completed enhanced processing for FOV {args.fov}")
                print(f"📁 Check results in: nuclear_dataset/{args.fov}/")
                if not args.visualize_only:
                    print("📊 Enhanced features applied:")
                    print("   🔬 Smart green fluorescence preprocessing")
                    print("   🔵 Ground truth blue marker preservation")
                    print("   🔙 Complete backward tracing (when available)")
                    print("   🔜 Complete forward tracing (when available)")
                    print("   🔄 Spatial tracking fallback (when no lineage)")
                    print("   ✅ Cross-validation of results")
                if not args.complete_tracking and not args.preprocess_only:
                    print("🎨 Enhanced visualization features:")
                    print("   🔧 Complete lineage trajectory logic")
                    print("   📁 All frames 000000.tif to 000400.tif")
                    print("   🔵 Blue bounding boxes for iPS cells")
                    print("   📝 Bold text for frame info and labeling")
                    print("   🔬 Preprocessed GREEN background + WHITE cells overlay")
                    print("   🖼️ Robust image format handling")
            else:
                print(f"\n❌ Enhanced processing failed for FOV {args.fov}")
                sys.exit(1)
        
        else:  # args.all
            if args.preprocess_only:
                print("🔬 Running preprocessing only for all FOVs")
                preprocess_all_fovs(args.base_path)
            elif args.visualize_only:
                print("🎨 Running visualization only for all FOVs")
                visualize_all_fovs_proper(args.base_path)
            elif args.complete_tracking:
                print("🏷️ Running enhanced tracking for all FOVs")
                label_all_fovs_complete_tracking(args.base_path, args.detection_start, args.detection_end)
            else:
                print("🚀 Running complete enhanced pipeline for all FOVs")
                run_complete_pipeline_all_fovs(
                    args.base_path, args.detection_start, args.detection_end, args.skip_preprocess
                )
            
            print("\n🎉 Successfully completed enhanced processing for all FOVs")
            print("📁 Check individual FOV results in nuclear_dataset/")
            print("✨ Enhanced features applied to all FOVs:")
            print("   🔬 Smart green fluorescence preprocessing")
            print("   🔵 Ground truth blue marker preservation")
            print("   🔙 Complete backward tracing (when lineage available)")
            print("   🔜 Complete forward tracing (when lineage available)")
            print("   🔄 Spatial tracking fallback (when no lineage)")
            print("   ✅ Cross-validation of tracing methods")
            print("   🏷️ Comprehensive lineage labeling")
            print("   🎨 Enhanced visualization with complete tracking")
            print("   🔵 Blue bounding boxes for iPS cells")
            print("   📝 Bold text for frame info and labeling")
            print("   🖼️ Robust image format handling")
    
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during enhanced processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
