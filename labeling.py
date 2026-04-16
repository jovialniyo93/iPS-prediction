#!/usr/bin/env python3
"""
iPS Cell Labeling System with Complete Lineage Tracking

FIXES APPLIED:
1. Fixed detailed_lineage.csv "Is iPS" column to correctly map Label column (1="Yes", 0="No")
2. Fixed Excel sorting to show Cell IDs in ascending order
3. Removed all "enhanced" terminology

COMPLETE IMPLEMENTATION:
1. 🔵 FIRST FRAME PURITY: Only experimentalist ground truth in frame 0
2. 🔄 ADAPTIVE MIDDLE FRAME VALIDATION: Three-method validation system
3. 🔗 ROBUST GAP HANDLING: Tolerates minor gaps while maintaining lineage integrity
4. 🚫 FAILED PROGENITOR DETECTION: Identifies GT cells that don't produce iPS
5. 🛡️ FALLBACK MECHANISMS: Multiple validation layers prevent implementation failure

TEMPORAL VALIDATION METHODOLOGY (as per SVG diagram):
1. Progressive Gap Tolerance: Based on frame position
2. Missing Segmentation Tolerance: Based on total missing frames
3. Combined Pattern: Pattern-based validation
Final Tolerance = MIN(Progressive, Segmentation, Combined)
"""

import os
import pandas as pd
import numpy as np
import cv2
from typing import List, Tuple, Set, Dict
from collections import defaultdict, deque
import csv

class iPSLabelerWithCompleteLineageTracking:
    """
    iPS Labeling System with Improved Middle Frame Validation
    
    KEY IMPROVEMENTS:
    1. 🔵 FIRST FRAME PURITY: Only experimentalist ground truth in frame 0
    2. 🔄 ADAPTIVE MIDDLE FRAME VALIDATION: Three-method validation system
    3. 🔗 ROBUST GAP HANDLING: Tolerates minor gaps while maintaining lineage integrity
    4. 🚫 FAILED PROGENITOR DETECTION: Identifies GT cells that don't produce iPS
    5. 🛡️ FALLBACK MECHANISMS: Multiple validation layers prevent implementation failure
    
    TEMPORAL VALIDATION METHODOLOGY (as per SVG diagram):
    1. Progressive Gap Tolerance: Based on frame position
    2. Missing Segmentation Tolerance: Based on total missing frames
    3. Combined Pattern: Pattern-based validation
    Final Tolerance = MIN(Progressive, Segmentation, Combined)
    """
    
    def __init__(self, base_path: str, fov: str):
        self.base_path = base_path
        self.fov = fov
        
        # Paths
        self.fov_path = os.path.join(base_path, "nuclear_dataset", fov)
        self.track_result_path = os.path.join(self.fov_path, "track_result")
        self.preprocessed_green_path = os.path.join(self.fov_path, "Preprocessed_green")
        self.labelled_path = os.path.join(self.fov_path, "Labelled")
        
        # Create Labelled folder
        os.makedirs(self.labelled_path, exist_ok=True)
        
        # Load data
        self.arranged_df = self.load_arranged_csv()
        
        # Tracking data structures
        self.tracking_data = {}  # cell_id -> {start_frame, end_frame, parent_id}
        self.parent_to_children = defaultdict(list)  # parent_id -> [child_ids]
        self.child_to_parent = {}  # child_id -> parent_id
        self.frame_to_cells = defaultdict(set)  # frame -> set of cell_ids
        self.cell_to_frames = defaultdict(set)  # cell_id -> set of frames
        
        # FIXED: Initialize division_frames BEFORE calling load_complete_tracking_data
        self.missing_cells_tracker = []  # Track missing cells by frame
        self.division_frames = {}  # Track division frames
        
        # Tracking mode
        self.has_lineage_data = False
        self.tracking_mode = "unknown"
        
        self.load_complete_tracking_data()
        
        # Detection results
        self.blue_marker_cells = set()  # Ground truth from first frame
        self.segmented_cells_last_frame = set()  # Detected from last frame
        self.backward_traced_lineages = set()  # All cells from backward tracing
        self.forward_traced_lineages = set()  # All cells from forward tracing
        self.spatial_tracked_lineages = set()  # Spatial tracking fallback
        self.final_ips_lineages = set()  # Final combined result
        
        # Validation tracking
        self.failed_progenitors = set()  # GT cells that don't connect to preprocessed
        self.disconnected_lineages = set()  # Lineages with gaps
        self.continuous_lineages = set()  # Validated continuous lineages
        self.middle_frame_validation_cache = {}  # Cache validation results
        self.validation_stats = defaultdict(int)  # Track validation statistics
    
    def load_arranged_csv(self):
        """Load the arranged.csv file"""
        arranged_path = os.path.join(self.track_result_path, "arranged.csv")
        
        if not os.path.exists(arranged_path):
            raise FileNotFoundError(f"arranged.csv not found at {arranged_path}")
            
        df = pd.read_csv(arranged_path)
        print(f"✅ Loaded arranged.csv: {len(df)} records")
        print(f"📊 Current Label distribution: {df['Label'].value_counts().to_dict()}")
        
        return df
    
    def load_complete_tracking_data(self):
        """Load comprehensive tracking data from res_track.txt"""
        track_path = os.path.join(self.track_result_path, "res_track.txt")
        
        if not os.path.exists(track_path):
            print(f"⚠️ Warning: res_track.txt not found at {track_path}")
            print("Will use spatial tracking fallback mode")
            self.tracking_mode = "no_tracking_file"
            return False
        
        print("📋 Loading comprehensive tracking data from res_track.txt...")
        
        try:
            with open(track_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"❌ Error reading res_track.txt: {e}")
            self.tracking_mode = "error_reading_file"
            return False
            
        print(f"📄 res_track.txt has {len(lines)} lines")
        
        parsed_count = 0
        parent_relationships = 0
        
        for line_num, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    cell_id = int(parts[0])
                    start_frame = int(parts[1])
                    end_frame = int(parts[2])
                    parent_id = int(parts[3])
                    
                    # Store tracking data
                    self.tracking_data[cell_id] = {
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'parent_id': parent_id
                    }
                    
                    # Build frame-cell mappings
                    for frame in range(start_frame, end_frame + 1):
                        self.frame_to_cells[frame].add(cell_id)
                        self.cell_to_frames[cell_id].add(frame)
                    
                    parsed_count += 1
                    
                    # Build parent-child relationships
                    if parent_id != 0:
                        self.parent_to_children[parent_id].append(cell_id)
                        self.child_to_parent[cell_id] = parent_id
                        parent_relationships += 1
                        
                        # Record division frame
                        if parent_id not in self.division_frames:
                            self.division_frames[parent_id] = {}
                        self.division_frames[parent_id][cell_id] = start_frame
                        
                except (ValueError, IndexError) as e:
                    if line_num < 10:  # Only print first few errors
                        print(f"Error parsing line {line_num}: {e}")
                    continue
        
        print(f"✅ Parsed {parsed_count} tracking records")
        print(f"🌳 Found {parent_relationships} parent-child relationships")
        
        # Determine tracking mode
        if parent_relationships > 0:
            self.has_lineage_data = True
            self.tracking_mode = "lineage_tracking"
            if self.frame_to_cells:
                print(f"📊 Frame coverage: {min(self.frame_to_cells.keys())} to {max(self.frame_to_cells.keys())}")
            print(f"👥 Found {len(self.parent_to_children)} parent cells")
            print(f"🎯 MODE: LINEAGE TRACKING")
        else:
            self.has_lineage_data = False
            self.tracking_mode = "segmentation_spatial"
            print(f"⚠️ WARNING: NO parent-child relationships found!")
            print(f"📄 MODE: SEGMENTATION + SPATIAL TRACKING (fallback)")
        
        return True
    
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
                pass
            except Exception:
                pass
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error loading {image_path}: {e}")
            return None
    
    def calculate_gap_pattern(self, missing_frames):
        """
        Calculate the gap pattern for Combined Pattern validation
        Returns a pattern score and whether it matches expected patterns
        """
        if not missing_frames:
            return 1.0, True, "No gaps"
        
        sorted_missing = sorted(missing_frames)
        gaps = []
        current_gap = 1
        
        # Calculate gap sizes
        for i in range(1, len(sorted_missing)):
            if sorted_missing[i] == sorted_missing[i-1] + 1:
                current_gap += 1
            else:
                gaps.append(current_gap)
                current_gap = 1
        gaps.append(current_gap)
        
        # Analyze pattern
        if len(gaps) == 1:
            # Single continuous gap
            return 0.8, True, f"Single gap of {gaps[0]} frames"
        
        # Check for alternating pattern (2,1,2,1...)
        alternating = True
        for i in range(len(gaps) - 1):
            if abs(gaps[i] - gaps[i+1]) > 2:
                alternating = False
                break
        
        if alternating:
            return 0.9, True, f"Alternating pattern: {gaps}"
        
        # Random gaps
        avg_gap = sum(gaps) / len(gaps)
        if avg_gap <= 2:
            return 0.7, True, f"Small gaps: avg {avg_gap:.1f}"
        else:
            return 0.5, False, f"Large gaps: avg {avg_gap:.1f}"
    
    def validate_middle_frame_continuity(self, cell_id):
        """
        🔄 IMPROVED: Three-method temporal validation as per SVG diagram
        
        Method 1: Progressive Gap Tolerance (based on frame position)
        Method 2: Missing Segmentation Tolerance (based on total missing)
        Method 3: Combined Pattern (pattern-based validation)
        Final Tolerance = MIN of all three methods
        """
        # Check cache first
        if cell_id in self.middle_frame_validation_cache:
            return self.middle_frame_validation_cache[cell_id]
        
        # Default validation for cells not in tracking data
        if cell_id not in self.tracking_data:
            result = (False, "No tracking data")
            self.middle_frame_validation_cache[cell_id] = result
            return result
        
        try:
            track_info = self.tracking_data[cell_id]
            start_frame = track_info['start_frame']
            end_frame = track_info['end_frame']
            parent_id = track_info.get('parent_id', 0)
            
            # Calculate expected and actual frames
            expected_frames = set(range(start_frame, end_frame + 1))
            actual_frames = self.cell_to_frames.get(cell_id, set())
            
            # Calculate missing frames and gaps
            missing_frames = expected_frames - actual_frames
            num_missing = len(missing_frames)
            lifespan = end_frame - start_frame + 1
            
            # Special case: cells in frame 0 (ground truth)
            if start_frame == 0 and cell_id in self.blue_marker_cells:
                validation_passed = True
                validation_reason = "Ground truth cell (frame 0)"
                self.continuous_lineages.add(cell_id)
                result = (validation_passed, validation_reason)
                self.middle_frame_validation_cache[cell_id] = result
                return result
            
            # Calculate gaps (consecutive missing frames)
            gaps = []
            if missing_frames:
                sorted_missing = sorted(missing_frames)
                current_gap = 1
                for i in range(1, len(sorted_missing)):
                    if sorted_missing[i] == sorted_missing[i-1] + 1:
                        current_gap += 1
                    else:
                        gaps.append(current_gap)
                        current_gap = 1
                gaps.append(current_gap)
            
            num_gaps = len(gaps)
            max_gap_size = max(gaps) if gaps else 0
            
            # METHOD 1: Progressive Gap Tolerance (based on frame position)
            progressive_tolerance = 0
            avg_frame = (start_frame + end_frame) / 2
            
            if avg_frame <= 150:
                progressive_tolerance = 1  # Frames 0-150: 1 gap allowed
            elif avg_frame <= 303:
                progressive_tolerance = 2  # Frames 150-303: 2 gaps allowed
            else:
                progressive_tolerance = 3  # Frames 303-400: 3 gaps allowed
            
            progressive_valid = num_gaps <= progressive_tolerance
            
            # METHOD 2: Missing Segmentation Tolerance (based on total missing frames)
            segmentation_tolerance = 0
            
            if num_missing >= 20:
                segmentation_tolerance = 1  # 20+ frames missing: 1 gap allowed
            elif num_missing >= 10:
                segmentation_tolerance = 2  # 10-19 frames missing: 2 gaps allowed
            else:
                segmentation_tolerance = 3  # <10 frames missing: 3 gaps allowed
            
            segmentation_valid = num_gaps <= segmentation_tolerance
            
            # METHOD 3: Combined Pattern validation
            pattern_score, pattern_valid, pattern_reason = self.calculate_gap_pattern(missing_frames)
            
            # Determine pattern tolerance based on score
            if pattern_score >= 0.9:
                pattern_tolerance = 3  # Good pattern: 3 gaps allowed
            elif pattern_score >= 0.7:
                pattern_tolerance = 2  # Moderate pattern: 2 gaps allowed
            else:
                pattern_tolerance = 1  # Poor pattern: 1 gap allowed
            
            combined_valid = num_gaps <= pattern_tolerance and pattern_valid
            
            # FINAL VALIDATION: MIN of all three methods
            final_tolerance = min(progressive_tolerance, segmentation_tolerance, pattern_tolerance)
            validation_passed = num_gaps <= final_tolerance
            
            # Create detailed validation reason
            validation_details = []
            validation_details.append(f"Gaps: {num_gaps} (max size: {max_gap_size})")
            validation_details.append(f"Progressive: {num_gaps}/{progressive_tolerance} {'✓' if progressive_valid else '✗'}")
            validation_details.append(f"Segmentation: {num_gaps}/{segmentation_tolerance} {'✓' if segmentation_valid else '✗'}")
            validation_details.append(f"Pattern: {num_gaps}/{pattern_tolerance} {'✓' if combined_valid else '✗'} ({pattern_reason})")
            validation_details.append(f"Final tolerance: {final_tolerance} gaps")
            
            validation_reason = " | ".join(validation_details)
            
            # Special validation for cells with parent/child relationships
            if not validation_passed and parent_id != 0:
                # Check if parent validates
                parent_valid, parent_reason = self.validate_middle_frame_continuity(parent_id)
                if parent_valid and num_gaps <= final_tolerance + 1:  # Allow one extra gap for child cells
                    validation_passed = True
                    validation_reason += " | Validated through parent lineage"
            
            # Update statistics
            if validation_passed:
                self.validation_stats['passed'] += 1
                self.continuous_lineages.add(cell_id)
            else:
                self.validation_stats['failed'] += 1
                self.disconnected_lineages.add(cell_id)
            
            result = (validation_passed, validation_reason)
            self.middle_frame_validation_cache[cell_id] = result
            return result
            
        except Exception as e:
            print(f"⚠️ Error validating cell {cell_id}: {e}")
            result = (False, f"Validation error: {e}")
            self.middle_frame_validation_cache[cell_id] = result
            return result
    
    def find_blue_markers_ground_truth(self):
        """🔵 Find blue markers from first frame as ground truth"""
        print(f"\n🔵 STEP 1: FINDING BLUE MARKERS (GROUND TRUTH)")
        
        # Get blue markers from arranged.csv (Label = 1)
        blue_markers = self.arranged_df[self.arranged_df['Label'] == 1]['Cell ID'].unique()
        self.blue_marker_cells = set(blue_markers)
        
        print(f"📋 Found {len(self.blue_marker_cells)} blue marker cells")
        
        # Validate blue markers in frame 0
        frame_0_blue_markers = set()
        if len(self.blue_marker_cells) > 0:
            blue_df = self.arranged_df[self.arranged_df['Cell ID'].isin(self.blue_marker_cells)]
            
            # Check frame 0 specifically
            frame_0_candidates = blue_df[blue_df['Frame'] == 0]['Cell ID'].unique()
            frame_0_blue_markers.update(frame_0_candidates)
            
            if frame_0_blue_markers:
                print(f"🎯 Frame 0 blue markers: {len(frame_0_blue_markers)} cells")
                self.blue_marker_cells = frame_0_blue_markers
            else:
                print(f"⚠️ No blue markers in frame 0, using all blue markers")
        
        return self.blue_marker_cells
    
    def find_segmented_cells_last_frame(self, detection_start_frame=350, detection_end_frame=400):
        """🔬 Find WHITE preprocessed cells from last frames"""
        print(f"\n🔬 STEP 2: FINDING WHITE PREPROCESSED CELLS ({detection_start_frame}-{detection_end_frame})")
        
        if not os.path.exists(self.preprocessed_green_path):
            print(f"⚠️ Preprocessed_green folder not found")
            return set()
        
        # Get preprocessed files
        preprocessed_files = sorted([f for f in os.listdir(self.preprocessed_green_path) 
                                    if f.endswith('.tif') or f.endswith('.tiff')])
        
        # Get tracking files for cell boundaries
        track_files = sorted([f for f in os.listdir(self.track_result_path) 
                             if f.endswith('.tif') or f.endswith('.tiff')])
        
        if not preprocessed_files:
            print(f"⚠️ No preprocessed files found")
            return set()
        
        segmented_candidates = set()
        detection_scores = defaultdict(list)
        
        # Process detection frames
        for frame_idx in range(detection_start_frame, min(detection_end_frame + 1, len(track_files))):
            preprocessed_frame_idx = frame_idx - 275  # Preprocessed starts at frame 275
            
            if preprocessed_frame_idx < 0 or preprocessed_frame_idx >= len(preprocessed_files):
                continue
            
            try:
                # Load preprocessed segmentation
                preprocessed_file_path = os.path.join(self.preprocessed_green_path, preprocessed_files[preprocessed_frame_idx])
                preprocessed_img = self.load_image_robust(preprocessed_file_path)
                
                # Load tracking mask
                track_file_path = os.path.join(self.track_result_path, track_files[frame_idx])
                track_img = self.load_image_robust(track_file_path)
                
                if preprocessed_img is None or track_img is None:
                    continue
                
                # Ensure single channel for tracking
                if len(track_img.shape) > 2:
                    track_img = cv2.cvtColor(track_img, cv2.COLOR_BGR2GRAY)
                
                # Detect white cells with adaptive thresholding
                if len(preprocessed_img.shape) > 2:
                    white_threshold = 240
                    white_mask = np.all(preprocessed_img >= white_threshold, axis=2).astype(np.uint8) * 255
                    
                    # Also check brightness
                    gray_preprocessed = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
                    bright_mask = (gray_preprocessed >= white_threshold).astype(np.uint8) * 255
                    white_mask = cv2.bitwise_or(white_mask, bright_mask)
                else:
                    white_threshold = 240
                    white_mask = (preprocessed_img >= white_threshold).astype(np.uint8) * 255
                
                # Resize if needed
                if white_mask.shape[:2] != track_img.shape[:2]:
                    white_mask = cv2.resize(white_mask, (track_img.shape[1], track_img.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST)
                
                # Find cells in this frame
                if self.has_lineage_data:
                    frame_cells = self.frame_to_cells.get(frame_idx, set())
                else:
                    frame_cells = set(np.unique(track_img[track_img > 0]))
                
                # Check each cell for overlap with white regions
                for cell_id in frame_cells:
                    try:
                        cell_id = int(cell_id)
                        if cell_id == 0:
                            continue
                        
                        # Get cell mask
                        cell_mask = (track_img == cell_id)
                        if not np.any(cell_mask):
                            continue
                        
                        # Check overlap
                        overlap = np.logical_and(cell_mask, white_mask == 255)
                        cell_pixels = np.sum(cell_mask)
                        overlap_pixels = np.sum(overlap)
                        
                        if cell_pixels == 0:
                            continue
                        
                        overlap_ratio = overlap_pixels / cell_pixels
                        
                        # Adaptive threshold based on cell size
                        min_overlap = 0.10 if cell_pixels < 100 else 0.15
                        
                        if overlap_ratio > min_overlap:
                            score = overlap_ratio * overlap_pixels
                            detection_scores[cell_id].append((frame_idx, score, overlap_ratio))
                    
                    except Exception:
                        continue
                
            except Exception as e:
                if frame_idx == detection_start_frame:
                    print(f"⚠️ Error processing frame {frame_idx}: {e}")
                continue
        
        # Validate candidates
        for cell_id, scores in detection_scores.items():
            if len(scores) >= 1:
                avg_score = np.mean([s[1] for s in scores])
                avg_overlap = np.mean([s[2] for s in scores])
                
                # Adaptive thresholds
                threshold_score = 10.0 if self.has_lineage_data else 8.0
                threshold_overlap = 0.15 if self.has_lineage_data else 0.10
                
                if avg_score >= threshold_score and avg_overlap >= threshold_overlap:
                    segmented_candidates.add(cell_id)
        
        self.segmented_cells_last_frame = segmented_candidates
        print(f"🔬 Detected {len(segmented_candidates)} WHITE/bright preprocessed cells")
        
        return segmented_candidates
    
    def detect_failed_progenitors(self):
        """🚫 Detect GT cells that don't connect to preprocessed cells"""
        print(f"\n🚫 DETECTING FAILED PROGENITORS")
        
        failed_progenitors = set()
        
        for gt_cell in self.blue_marker_cells:
            connected_to_preprocessed = False
            
            # Check direct connection
            if gt_cell in self.segmented_cells_last_frame:
                connected_to_preprocessed = True
            
            # Check through descendants
            if not connected_to_preprocessed and self.has_lineage_data:
                descendants = self.get_all_descendants(gt_cell)
                if descendants.intersection(self.segmented_cells_last_frame):
                    connected_to_preprocessed = True
            
            # Check through continuity
            if not connected_to_preprocessed and gt_cell in self.tracking_data:
                end_frame = self.tracking_data[gt_cell]['end_frame']
                if end_frame >= 250:
                    is_continuous, _ = self.validate_middle_frame_continuity(gt_cell)
                    if is_continuous:
                        connected_to_preprocessed = True
            
            if not connected_to_preprocessed:
                failed_progenitors.add(gt_cell)
        
        self.failed_progenitors = failed_progenitors
        successful_progenitors = self.blue_marker_cells - failed_progenitors
        
        if self.blue_marker_cells:
            success_rate = len(successful_progenitors) / len(self.blue_marker_cells) * 100
            print(f"📊 Progenitor Analysis:")
            print(f"   ✅ Successful: {len(successful_progenitors)} ({success_rate:.1f}%)")
            print(f"   🚫 Failed: {len(failed_progenitors)}")
        
        return failed_progenitors
    
    def backward_trace_complete_lineages(self, seed_cells):
        """🔙 Backward trace from seed cells with frame 0 protection"""
        print(f"\n🔙 STEP 3: BACKWARD TRACING")
        
        if not self.has_lineage_data:
            print("⚠️ No lineage data - using spatial fallback")
            return self.spatial_tracking_fallback(seed_cells)
        
        backward_lineages = set()
        frame_0_candidates = set()
        
        for seed_cell in seed_cells:
            try:
                lineage = self.get_complete_backward_lineage(seed_cell)
                
                # Separate frame 0 cells
                for cell_id in lineage:
                    if cell_id in self.tracking_data:
                        if self.tracking_data[cell_id]['start_frame'] == 0:
                            frame_0_candidates.add(cell_id)
                        else:
                            backward_lineages.add(cell_id)
                    else:
                        backward_lineages.add(cell_id)
                
            except Exception as e:
                print(f"⚠️ Error tracing cell {seed_cell}: {e}")
                if seed_cell in self.tracking_data and self.tracking_data[seed_cell]['start_frame'] != 0:
                    backward_lineages.add(seed_cell)
        
        # Validate frame 0 candidates
        validated_frame_0 = frame_0_candidates.intersection(self.blue_marker_cells)
        rejected_frame_0 = frame_0_candidates - self.blue_marker_cells
        
        if validated_frame_0:
            print(f"✅ Frame 0 validated: {len(validated_frame_0)} cells")
            backward_lineages.update(validated_frame_0)
        
        if rejected_frame_0:
            print(f"🚫 Frame 0 rejected: {len(rejected_frame_0)} cells (not ground truth)")
        
        self.backward_traced_lineages = backward_lineages
        print(f"🔙 Backward traced: {len(backward_lineages)} cells")
        
        return backward_lineages
    
    def forward_trace_from_ground_truth(self, ground_truth_cells):
        """🔜 Forward trace from ground truth cells"""
        print(f"\n🔜 STEP 4: FORWARD TRACING")
        
        if not self.has_lineage_data:
            print("⚠️ No lineage data - using spatial fallback")
            return self.spatial_tracking_fallback(ground_truth_cells)
        
        forward_lineages = set()
        
        for gt_cell in ground_truth_cells:
            try:
                lineage = self.get_complete_forward_lineage(gt_cell)
                
                # Validate continuity for lineage cells
                for cell_id in lineage:
                    is_continuous, _ = self.validate_middle_frame_continuity(cell_id)
                    if is_continuous or cell_id in self.blue_marker_cells:
                        forward_lineages.add(cell_id)
                
            except Exception as e:
                print(f"⚠️ Error tracing cell {gt_cell}: {e}")
                forward_lineages.add(gt_cell)
        
        self.forward_traced_lineages = forward_lineages
        print(f"🔜 Forward traced: {len(forward_lineages)} cells")
        
        return forward_lineages
    
    def cross_validate_and_combine(self):
        """✅ Cross-validate and combine results"""
        print(f"\n✅ STEP 5: CROSS-VALIDATION")
        
        if self.has_lineage_data:
            backward_set = self.backward_traced_lineages
            forward_set = self.forward_traced_lineages
        else:
            backward_set = self.spatial_tracked_lineages
            forward_set = self.spatial_tracked_lineages
        
        # Find overlaps
        both_methods = backward_set.intersection(forward_set)
        only_backward = backward_set - forward_set
        only_forward = forward_set - backward_set
        
        print(f"📊 Validation:")
        print(f"   Both methods: {len(both_methods)} cells")
        print(f"   Only backward: {len(only_backward)} cells")
        print(f"   Only forward: {len(only_forward)} cells")
        
        # Combine results
        combined_lineages = backward_set.union(forward_set)
        
        # Add segmented cells (with frame 0 protection)
        for cell_id in self.segmented_cells_last_frame:
            if cell_id in self.tracking_data:
                if self.tracking_data[cell_id]['start_frame'] != 0:
                    combined_lineages.add(cell_id)
            elif cell_id in self.blue_marker_cells:
                combined_lineages.add(cell_id)
        
        # Final validation
        validated_lineages = set()
        for cell_id in combined_lineages:
            if cell_id in self.blue_marker_cells:
                validated_lineages.add(cell_id)
            elif cell_id in self.tracking_data:
                if self.tracking_data[cell_id]['start_frame'] == 0:
                    if cell_id in self.blue_marker_cells:
                        validated_lineages.add(cell_id)
                else:
                    validated_lineages.add(cell_id)
            elif cell_id in self.arranged_df['Cell ID'].values:
                validated_lineages.add(cell_id)
        
        self.final_ips_lineages = validated_lineages
        print(f"✅ Final validated: {len(validated_lineages)} cells")
        
        return validated_lineages
    
    def spatial_tracking_fallback(self, seed_cells, max_distance=50):
        """🔄 Spatial tracking fallback"""
        print(f"\n🔄 SPATIAL TRACKING FALLBACK")
        
        if not seed_cells:
            return set()
        
        spatial_lineages = set()
        
        for seed_cell in seed_cells:
            try:
                lineage = self.get_spatial_lineage(seed_cell, max_distance)
                spatial_lineages.update(lineage)
            except Exception:
                spatial_lineages.add(seed_cell)
        
        self.spatial_tracked_lineages = spatial_lineages
        print(f"🔄 Spatially tracked: {len(spatial_lineages)} cells")
        
        return spatial_lineages
    
    def get_spatial_lineage(self, seed_cell, max_distance=50):
        """Get lineage using spatial proximity"""
        lineage = set([seed_cell])
        
        if seed_cell in self.cell_to_frames:
            seed_frames = self.cell_to_frames[seed_cell]
        else:
            return lineage
        
        for frame in seed_frames:
            try:
                nearby = self.find_nearby_cells_in_frame(seed_cell, frame, max_distance)
                lineage.update(nearby)
            except Exception:
                continue
        
        return lineage
    
    def find_nearby_cells_in_frame(self, reference_cell, frame, max_distance):
        """Find nearby cells in frame"""
        nearby = set()
        
        try:
            ref_center = self.get_cell_center_in_frame(reference_cell, frame)
            if ref_center is None:
                return nearby
            
            frame_cells = self.frame_to_cells.get(frame, set())
            for cell_id in frame_cells:
                if cell_id == reference_cell:
                    continue
                
                cell_center = self.get_cell_center_in_frame(cell_id, frame)
                if cell_center is None:
                    continue
                
                distance = np.sqrt((ref_center[0] - cell_center[0])**2 + 
                                 (ref_center[1] - cell_center[1])**2)
                
                if distance <= max_distance:
                    nearby.add(cell_id)
        
        except Exception:
            pass
        
        return nearby
    
    def get_cell_center_in_frame(self, cell_id, frame):
        """Get cell center in specific frame"""
        try:
            track_files = sorted([f for f in os.listdir(self.track_result_path) 
                                 if f.endswith('.tif') or f.endswith('.tiff')])
            
            if frame >= len(track_files):
                return None
            
            track_img_path = os.path.join(self.track_result_path, track_files[frame])
            track_img = self.load_image_robust(track_img_path)
            
            if track_img is None:
                return None
            
            if len(track_img.shape) > 2:
                track_img = cv2.cvtColor(track_img, cv2.COLOR_BGR2GRAY)
            
            cell_mask = (track_img == cell_id)
            if not np.any(cell_mask):
                return None
            
            y_coords, x_coords = np.where(cell_mask)
            center_x = int(np.mean(x_coords))
            center_y = int(np.mean(y_coords))
            
            return (center_x, center_y)
        
        except Exception:
            return None
    
    def get_complete_backward_lineage(self, cell_id):
        """Get complete backward lineage"""
        lineage = set()
        queue = deque([cell_id])
        visited = set()
        
        while queue:
            current_cell = queue.popleft()
            if current_cell in visited:
                continue
            
            visited.add(current_cell)
            lineage.add(current_cell)
            
            if current_cell in self.child_to_parent:
                parent_id = self.child_to_parent[current_cell]
                if parent_id != 0 and parent_id not in visited:
                    queue.append(parent_id)
        
        return lineage
    
    def get_complete_forward_lineage(self, cell_id):
        """Get complete forward lineage"""
        lineage = set()
        queue = deque([cell_id])
        visited = set()
        
        while queue:
            current_cell = queue.popleft()
            if current_cell in visited:
                continue
            
            visited.add(current_cell)
            lineage.add(current_cell)
            
            if current_cell in self.parent_to_children:
                for child_id in self.parent_to_children[current_cell]:
                    if child_id not in visited:
                        queue.append(child_id)
        
        return lineage
    
    def get_all_ancestors(self, cell_id):
        """Get all ancestor cells"""
        ancestors = set()
        current_id = cell_id
        max_iterations = 200
        
        for _ in range(max_iterations):
            if current_id not in self.child_to_parent:
                break
            parent_id = self.child_to_parent[current_id]
            if parent_id == 0:
                break
            ancestors.add(parent_id)
            current_id = parent_id
        
        return ancestors
    
    def get_all_descendants(self, cell_id):
        """Get all descendant cells"""
        descendants = set()
        queue = deque([cell_id])
        visited = set([cell_id])
        
        while queue:
            current_cell = queue.popleft()
            
            if current_cell in self.parent_to_children:
                for child_id in self.parent_to_children[current_cell]:
                    if child_id not in visited:
                        descendants.add(child_id)
                        visited.add(child_id)
                        queue.append(child_id)
        
        return descendants
    
    def calculate_generation_distance(self, ancestor_id, descendant_id):
        """Calculate how many generations apart two cells are"""
        current_id = descendant_id
        distance = 0
        max_iterations = 20
        
        for _ in range(max_iterations):
            if current_id == ancestor_id:
                return distance
            if current_id not in self.child_to_parent:
                break
            parent_id = self.child_to_parent[current_id]
            if parent_id == 0:
                break
            current_id = parent_id
            distance += 1
        
        return distance
    
    def create_lineage_tree_excel(self, cell_labels):
        """
        📊 Create Excel lineage tree showing complete ancestry and descendancy for each cell
        FIXED VERSION with comprehensive error handling and ASCENDING CELL ID SORTING
        """
        print(f"\n📊 Creating Excel lineage tree with complete ancestry/descendancy...")
        
        # Check dependencies first
        try:
            import pandas as pd
        except ImportError:
            print("❌ ERROR: pandas not installed. Install with: pip install pandas")
            return None
        
        try:
            import openpyxl
        except ImportError:
            print("❌ ERROR: openpyxl not installed. Install with: pip install openpyxl")
            return None
        
        # Ensure labelled directory exists
        try:
            os.makedirs(self.labelled_path, exist_ok=True)
            print(f"📁 Excel will be saved to: {self.labelled_path}")
        except Exception as e:
            print(f"❌ ERROR: Cannot create directory {self.labelled_path}: {e}")
            return None
        
        # Validate input data
        if not self.tracking_data:
            print("❌ ERROR: No tracking data available for Excel export")
            return None
        
        if not cell_labels:
            print("❌ ERROR: No cell labels available for Excel export")
            return None
        
        try:
            # Create comprehensive lineage tree data
            lineage_data = []
            print(f"📄 Processing {len(self.tracking_data)} cells for Excel export...")
            
            processed_count = 0
            for cell_id in sorted(self.tracking_data.keys()):
                try:
                    cell_info = {
                        'Cell_ID': int(cell_id),
                        'Label': 'iPS' if cell_labels.get(cell_id, 'normal') == 'iPS' else 'Normal',
                        'Start_Frame': int(self.tracking_data[cell_id]['start_frame']),
                        'End_Frame': int(self.tracking_data[cell_id]['end_frame']),
                        'Lifespan': int(self.tracking_data[cell_id]['end_frame'] - self.tracking_data[cell_id]['start_frame'] + 1),
                        'Direct_Parent': int(self.tracking_data[cell_id]['parent_id']) if self.tracking_data[cell_id]['parent_id'] != 0 else 'Root_Cell',
                        'Direct_Children': '',
                        'Complete_Ancestry_Chain': '',
                        'Complete_Descendancy_Tree': '',
                        'Generation_Level': 0,
                        'Total_Ancestors': 0,
                        'Total_Descendants': 0,
                        'Family_Tree_Position': ''
                    }
                    
                    # Calculate direct children - with error handling
                    try:
                        if cell_id in self.parent_to_children:
                            children = sorted([int(child) for child in self.parent_to_children[cell_id]])
                            cell_info['Direct_Children'] = ' → '.join(map(str, children))
                        else:
                            cell_info['Direct_Children'] = 'No_Children'
                    except Exception as e:
                        print(f"⚠️ Warning: Error processing children for cell {cell_id}: {e}")
                        cell_info['Direct_Children'] = 'Error_Processing'
                    
                    # Calculate complete ancestry chain - with error handling
                    try:
                        ancestry_chain = []
                        current_id = cell_id
                        generation = 0
                        max_iterations = 50  # Prevent infinite loops
                        
                        while current_id in self.child_to_parent and self.child_to_parent[current_id] != 0 and generation < max_iterations:
                            parent_id = self.child_to_parent[current_id]
                            ancestry_chain.append(int(parent_id))
                            current_id = parent_id
                            generation += 1
                        
                        cell_info['Total_Ancestors'] = len(ancestry_chain)
                        cell_info['Generation_Level'] = generation
                        
                        if ancestry_chain:
                            # Show complete path from root to current cell
                            full_path = ' → '.join(map(str, reversed(ancestry_chain))) + f' → {cell_id}'
                            cell_info['Complete_Ancestry_Chain'] = full_path
                        else:
                            cell_info['Complete_Ancestry_Chain'] = f'{cell_id} (Root_Cell)'
                    except Exception as e:
                        print(f"⚠️ Warning: Error processing ancestry for cell {cell_id}: {e}")
                        cell_info['Complete_Ancestry_Chain'] = f'{cell_id} (Error_Processing)'
                        cell_info['Generation_Level'] = 0
                        cell_info['Total_Ancestors'] = 0
                    
                    # Calculate complete descendancy tree - with error handling
                    try:
                        descendants = list(self.get_all_descendants(cell_id))
                        cell_info['Total_Descendants'] = len(descendants)
                        
                        if descendants:
                            # Group descendants by generation level
                            descendant_levels = {}
                            for desc_id in descendants:
                                try:
                                    desc_generation = self.calculate_generation_distance(cell_id, desc_id)
                                    if desc_generation not in descendant_levels:
                                        descendant_levels[desc_generation] = []
                                    descendant_levels[desc_generation].append(int(desc_id))
                                except Exception:
                                    # If generation calculation fails, put in generation 1
                                    if 1 not in descendant_levels:
                                        descendant_levels[1] = []
                                    descendant_levels[1].append(int(desc_id))
                            
                            # Build descendancy tree representation
                            tree_repr = []
                            for level in sorted(descendant_levels.keys()):
                                level_cells = sorted(descendant_levels[level])
                                tree_repr.append(f"Gen+{level}: {', '.join(map(str, level_cells))}")
                            
                            cell_info['Complete_Descendancy_Tree'] = ' | '.join(tree_repr)
                        else:
                            cell_info['Complete_Descendancy_Tree'] = 'No_Descendants'
                    except Exception as e:
                        print(f"⚠️ Warning: Error processing descendants for cell {cell_id}: {e}")
                        cell_info['Complete_Descendancy_Tree'] = 'Error_Processing'
                        cell_info['Total_Descendants'] = 0
                    
                    # Family tree position description
                    try:
                        if cell_info['Total_Ancestors'] == 0 and cell_info['Total_Descendants'] > 0:
                            position = f"Root_Ancestor (Gen_0) → {cell_info['Total_Descendants']} descendants"
                        elif cell_info['Total_Ancestors'] > 0 and cell_info['Total_Descendants'] > 0:
                            position = f"Intermediate_Cell (Gen_{generation}) → {cell_info['Total_Descendants']} descendants"
                        elif cell_info['Total_Ancestors'] > 0 and cell_info['Total_Descendants'] == 0:
                            position = f"Terminal_Cell (Gen_{generation}) → No descendants"
                        else:
                            position = "Isolated_Cell (No family connections)"
                        
                        cell_info['Family_Tree_Position'] = position
                    except Exception as e:
                        print(f"⚠️ Warning: Error creating position for cell {cell_id}: {e}")
                        cell_info['Family_Tree_Position'] = 'Error_Processing'
                    
                    lineage_data.append(cell_info)
                    processed_count += 1
                    
                    # Progress indicator
                    if processed_count % 100 == 0:
                        print(f"   📊 Processed {processed_count}/{len(self.tracking_data)} cells...")
                        
                except Exception as e:
                    print(f"⚠️ Warning: Error processing cell {cell_id}: {e}")
                    # Add basic cell info even if processing fails
                    lineage_data.append({
                        'Cell_ID': int(cell_id),
                        'Label': 'Error',
                        'Start_Frame': 0,
                        'End_Frame': 0,
                        'Lifespan': 0,
                        'Direct_Parent': 'Error',
                        'Direct_Children': 'Error',
                        'Complete_Ancestry_Chain': 'Error',
                        'Complete_Descendancy_Tree': 'Error',
                        'Generation_Level': 0,
                        'Total_Ancestors': 0,
                        'Total_Descendants': 0,
                        'Family_Tree_Position': 'Error_Processing'
                    })
                    continue
            
            print(f"✅ Successfully processed {processed_count} cells")
            
            # Validate we have data
            if not lineage_data:
                print("❌ ERROR: No lineage data created")
                return None
            
            # Create DataFrame with error handling
            try:
                lineage_df = pd.DataFrame(lineage_data)
                print(f"📊 Created DataFrame with {len(lineage_df)} rows and {len(lineage_df.columns)} columns")
            except Exception as e:
                print(f"❌ ERROR: Failed to create DataFrame: {e}")
                return None
            
            # FIXED: Sort by Cell_ID in ASCENDING order for better readability
            try:
                lineage_df = lineage_df.sort_values(['Cell_ID'])
                print("✅ DataFrame sorted by Cell_ID in ascending order")
            except Exception as e:
                print(f"⚠️ Warning: Could not sort DataFrame: {e}")
            
            # Create Excel file path
            excel_path = os.path.join(self.labelled_path, "lineage_tree_retrospective.xlsx")
            print(f"💾 Saving Excel file to: {excel_path}")
            
            # Save as Excel file with multiple sheets - with comprehensive error handling
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    print("📄 Creating Excel sheets...")
                    
                    # Main lineage tree sheet
                    try:
                        lineage_df.to_excel(writer, sheet_name='Complete_Lineage_Tree', index=False)
                        print("   ✅ Complete_Lineage_Tree sheet created")
                    except Exception as e:
                        print(f"   ❌ Error creating Complete_Lineage_Tree: {e}")
                    
                    # iPS lineages only sheet - ALSO SORTED BY CELL_ID
                    try:
                        ips_lineage_df = lineage_df[lineage_df['Label'] == 'iPS'].copy().sort_values(['Cell_ID'])
                        ips_lineage_df.to_excel(writer, sheet_name='iPS_Lineages_Only', index=False)
                        print(f"   ✅ iPS_Lineages_Only sheet created ({len(ips_lineage_df)} iPS cells)")
                    except Exception as e:
                        print(f"   ❌ Error creating iPS_Lineages_Only: {e}")
                    
                    # Root cells sheet (generation 0) - SORTED BY CELL_ID
                    try:
                        root_cells_df = lineage_df[lineage_df['Generation_Level'] == 0].copy().sort_values(['Cell_ID'])
                        root_cells_df.to_excel(writer, sheet_name='Root_Cells_Gen0', index=False)
                        print(f"   ✅ Root_Cells_Gen0 sheet created ({len(root_cells_df)} root cells)")
                    except Exception as e:
                        print(f"   ❌ Error creating Root_Cells_Gen0: {e}")
                    
                    # Terminal cells sheet (no descendants) - SORTED BY CELL_ID
                    try:
                        terminal_cells_df = lineage_df[lineage_df['Total_Descendants'] == 0].copy().sort_values(['Cell_ID'])
                        terminal_cells_df.to_excel(writer, sheet_name='Terminal_Cells', index=False)
                        print(f"   ✅ Terminal_Cells sheet created ({len(terminal_cells_df)} terminal cells)")
                    except Exception as e:
                        print(f"   ❌ Error creating Terminal_Cells: {e}")
                    
                    # Summary statistics sheet
                    try:
                        # Create summary with safe calculations
                        ips_count = len(lineage_df[lineage_df['Label'] == 'iPS']) if len(lineage_df) > 0 else 0
                        normal_count = len(lineage_df[lineage_df['Label'] == 'Normal']) if len(lineage_df) > 0 else 0
                        root_count = len(lineage_df[lineage_df['Generation_Level'] == 0]) if len(lineage_df) > 0 else 0
                        terminal_count = len(lineage_df[lineage_df['Total_Descendants'] == 0]) if len(lineage_df) > 0 else 0
                        max_generation = lineage_df['Generation_Level'].max() if len(lineage_df) > 0 else 0
                        avg_family_size = lineage_df['Total_Descendants'].mean() if len(lineage_df) > 0 else 0
                        largest_family = lineage_df['Total_Descendants'].max() if len(lineage_df) > 0 else 0
                        ips_percentage = (ips_count / len(lineage_df) * 100) if len(lineage_df) > 0 else 0
                        
                        summary_data = {
                            'Metric': [
                                'Total Cells Tracked',
                                'iPS Cells Identified',
                                'Normal Cells',
                                'iPS Percentage',
                                'Root Cells (Generation 0)',
                                'Terminal Cells (No Descendants)',
                                'Maximum Generation Level',
                                'Average Family Size',
                                'Largest Family Tree Size',
                                'Tracking Method Used',
                                'Excel Creation Date',
                                'FOV Number',
                                'Validation Method Applied',
                                'Cell ID Sorting'
                            ],
                            'Value': [
                                len(lineage_df),
                                ips_count,
                                normal_count,
                                f"{ips_percentage:.1f}%",
                                root_count,
                                terminal_count,
                                max_generation,
                                f"{avg_family_size:.1f}",
                                largest_family,
                                getattr(self, 'tracking_mode', 'unknown'),
                                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                str(getattr(self, 'fov', 'unknown')),
                                'Three-method temporal validation (Progressive, Segmentation, Combined)',
                                'Cell IDs sorted in ascending order'
                            ]
                        }
                        
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                        print("   ✅ Summary_Statistics sheet created")
                    except Exception as e:
                        print(f"   ❌ Error creating Summary_Statistics: {e}")
                
                # Verify file was created
                if os.path.exists(excel_path):
                    file_size = os.path.getsize(excel_path)
                    print(f"✅ Excel lineage tree saved successfully!")
                    print(f"📁 File location: {excel_path}")
                    print(f"📊 File size: {file_size:,} bytes")
                    print(f"📈 Contains {len(lineage_df)} cells with complete lineage information")
                    print(f"🔧 Applied three-method temporal validation for accuracy")
                    print(f"📋 Cell IDs sorted in ascending order for easy navigation")
                    
                    # Additional statistics
                    if len(lineage_df) > 0:
                        ips_count = len(lineage_df[lineage_df['Label'] == 'iPS'])
                        generations = lineage_df['Generation_Level'].nunique()
                        root_families = len(lineage_df[lineage_df['Generation_Level'] == 0])
                        max_gen = lineage_df['Generation_Level'].max()
                        
                        print(f"📈 iPS lineages: {ips_count} cells across {generations} generations")
                        print(f"🌳 Family trees: {root_families} root families, max {max_gen} generations deep")
                    
                    return excel_path
                else:
                    print(f"❌ ERROR: Excel file was not created at {excel_path}")
                    return None
                    
            except PermissionError as e:
                print(f"❌ ERROR: Permission denied when saving Excel file: {e}")
                print(f"💡 Try running as administrator or check file permissions for: {excel_path}")
                return None
            except Exception as e:
                print(f"❌ ERROR: Failed to save Excel file: {e}")
                print(f"💡 Check disk space and write permissions for: {self.labelled_path}")
                return None
                
        except Exception as e:
            print(f"❌ ERROR: Unexpected error in Excel creation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_detailed_lineage_csv(self, cell_labels):
        """
        📋 Create detailed lineage CSV similar to count.py output
        FIXED VERSION with correct "Is iPS" mapping to Label column and missing cell detection
        """
        print(f"\n📋 Creating detailed lineage CSV...")
        
        try:
            # First, load the updated arranged.csv to get the correct Label values
            updated_csv_path = os.path.join(self.labelled_path, "arranged_retrospective_labels.csv")
            
            # Create a mapping from Cell_ID to Label for quick lookup
            cell_id_to_label = {}
            if os.path.exists(updated_csv_path):
                try:
                    updated_df = pd.read_csv(updated_csv_path)
                    for _, row in updated_df.iterrows():
                        cell_id_to_label[row['Cell ID']] = row['Label']
                    print(f"✅ Loaded label mapping from updated CSV: {len(cell_id_to_label)} mappings")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load updated CSV for label mapping: {e}")
                    print("Will use cell_labels dictionary instead")
            
            # Prepare detailed lineage data
            detailed_lineage_data = []
            
            # First, detect missing cells by analyzing frame presence
            frame_cell_presence = {}
            
            # Get tracking files to determine actual cell presence
            track_files = sorted([f for f in os.listdir(self.track_result_path) 
                                 if f.endswith('.tif') or f.endswith('.tiff')])
            
            # Analyze which cells are actually present in each frame
            for frame_idx in range(len(track_files)):
                try:
                    track_img_path = os.path.join(self.track_result_path, track_files[frame_idx])
                    track_img = self.load_image_robust(track_img_path)
                    
                    if track_img is not None:
                        if len(track_img.shape) > 2:
                            track_img = cv2.cvtColor(track_img, cv2.COLOR_BGR2GRAY)
                        
                        present_cells = set(np.unique(track_img))
                        if 0 in present_cells:
                            present_cells.remove(0)  # Remove background
                        frame_cell_presence[frame_idx] = present_cells
                    else:
                        frame_cell_presence[frame_idx] = set()
                except Exception:
                    frame_cell_presence[frame_idx] = set()
            
            # Process each cell in tracking data
            for cell_id in sorted(self.tracking_data.keys()):
                try:
                    track_info = self.tracking_data[cell_id]
                    start_frame = track_info['start_frame']
                    end_frame = track_info['end_frame']
                    parent_id = track_info['parent_id']
                    
                    # FIXED: Determine if cell is iPS based on Label column from CSV
                    is_ips_value = "No"  # Default
                    
                    # First try to get from updated CSV (most accurate)
                    if cell_id in cell_id_to_label:
                        label_value = cell_id_to_label[cell_id]
                        is_ips_value = "Yes" if label_value == 1 else "No"
                    else:
                        # Fallback to cell_labels dictionary
                        is_ips_value = "Yes" if cell_labels.get(cell_id, 'normal') == 'iPS' else "No"
                    
                    # Check if cell has children
                    has_children = cell_id in self.parent_to_children and len(self.parent_to_children[cell_id]) > 0
                    
                    # Determine if cell is missing in any expected frames
                    is_missing = False
                    for frame_idx in range(start_frame, end_frame + 1):
                        if frame_idx in frame_cell_presence and cell_id not in frame_cell_presence[frame_idx]:
                            is_missing = True
                            break
                    
                    # Get children IDs
                    children_ids = []
                    if cell_id in self.parent_to_children:
                        children_ids = sorted([int(child) for child in self.parent_to_children[cell_id]])
                    
                    # Get validation information
                    is_continuous, validation_reason = self.validate_middle_frame_continuity(cell_id)
                    
                    # Create detailed entry
                    detailed_entry = {
                        "Cell ID": int(cell_id),
                        "Start Frame": int(start_frame),
                        "End Frame": int(end_frame),
                        "Parent ID": int(parent_id) if parent_id != 0 else 0,
                        "Is iPS": is_ips_value,  # FIXED: Now correctly maps to Label column
                        "Has Children": "Yes" if has_children else "No",
                        "Is Missing": "Yes" if is_missing else "No",
                        "Children IDs": ",".join(map(str, children_ids)) if children_ids else "None",
                        "Is Continuous": "Yes" if is_continuous else "No",
                        "Validation Reason": validation_reason,
                        "Generation Level": 0,
                        "Total Ancestors": 0,
                        "Total Descendants": len(children_ids) if has_children else 0,
                        "In Blue Markers": "Yes" if cell_id in self.blue_marker_cells else "No",
                        "In Segmented": "Yes" if cell_id in self.segmented_cells_last_frame else "No",
                        "In Final iPS": "Yes" if cell_id in self.final_ips_lineages else "No"
                    }
                    
                    # Calculate generation level
                    try:
                        current_id = cell_id
                        generation = 0
                        max_iterations = 50
                        
                        while current_id in self.child_to_parent and self.child_to_parent[current_id] != 0 and generation < max_iterations:
                            parent_id_gen = self.child_to_parent[current_id]
                            current_id = parent_id_gen
                            generation += 1
                        
                        detailed_entry["Generation Level"] = generation
                        
                        # Calculate total ancestors
                        ancestors = self.get_all_ancestors(cell_id)
                        detailed_entry["Total Ancestors"] = len(ancestors)
                        
                        # Calculate total descendants (all, not just direct children)
                        descendants = self.get_all_descendants(cell_id)
                        detailed_entry["Total Descendants"] = len(descendants)
                        
                    except Exception as e:
                        print(f"⚠️ Warning: Error calculating lineage metrics for cell {cell_id}: {e}")
                    
                    detailed_lineage_data.append(detailed_entry)
                    
                except Exception as e:
                    print(f"⚠️ Warning: Error processing cell {cell_id} for detailed CSV: {e}")
                    continue
            
            # FIXED: Sort the detailed lineage data by Cell ID in ascending order
            detailed_lineage_data.sort(key=lambda x: x["Cell ID"])
            
            # Save detailed lineage CSV
            detailed_csv_path = os.path.join(self.labelled_path, "detailed_lineage.csv")
            
            try:
                with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as f:
                    if detailed_lineage_data:
                        fieldnames = list(detailed_lineage_data[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(detailed_lineage_data)
                
                print(f"✅ Detailed lineage CSV saved successfully!")
                print(f"📁 File location: {detailed_csv_path}")
                print(f"📊 Contains {len(detailed_lineage_data)} cell records")
                print(f"📋 Cell IDs sorted in ascending order")
                print(f"✅ 'Is iPS' column correctly maps to Label column (1='Yes', 0='No')")
                
                # Summary statistics
                if detailed_lineage_data:
                    ips_cells = [entry for entry in detailed_lineage_data if entry["Is iPS"] == "Yes"]
                    missing_cells = [entry for entry in detailed_lineage_data if entry["Is Missing"] == "Yes"]
                    continuous_cells = [entry for entry in detailed_lineage_data if entry["Is Continuous"] == "Yes"]
                    
                    print(f"📈 Summary:")
                    print(f"   iPS cells: {len(ips_cells)}")
                    print(f"   Missing cells: {len(missing_cells)}")
                    print(f"   Continuous cells: {len(continuous_cells)}")
                
                return detailed_csv_path
                
            except Exception as e:
                print(f"❌ ERROR: Failed to save detailed lineage CSV: {e}")
                return None
                
        except Exception as e:
            print(f"❌ ERROR: Unexpected error in detailed CSV creation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_comprehensive_labels(self):
        """🏷️ Create comprehensive labels with validation"""
        print(f"\n🏷️ STEP 6: CREATING LABELS")
        
        # Get all cells
        all_cells = set(self.tracking_data.keys())
        all_cells.update(self.arranged_df['Cell ID'].unique())
        
        cell_labels = {}
        validation_stats = defaultdict(int)
        
        for cell_id in all_cells:
            label = 'normal'
            
            if cell_id in self.final_ips_lineages:
                # Validate the cell
                if cell_id in self.blue_marker_cells:
                    label = 'iPS'
                    validation_stats['ground_truth'] += 1
                elif cell_id in self.tracking_data:
                    is_continuous, reason = self.validate_middle_frame_continuity(cell_id)
                    if is_continuous:
                        label = 'iPS'
                        validation_stats['continuous'] += 1
                    else:
                        label = 'normal'
                        validation_stats['gap_excluded'] += 1
                elif cell_id in self.arranged_df['Cell ID'].values:
                    label = 'iPS'
                    validation_stats['other'] += 1
            
            cell_labels[cell_id] = label
            if label == 'iPS':
                validation_stats['total_ips'] += 1
        
        ips_count = validation_stats['total_ips']
        total_count = len(cell_labels)
        
        print(f"📊 LABELING RESULTS:")
        print(f"   Total cells: {total_count}")
        print(f"   iPS cells: {ips_count} ({ips_count/total_count*100:.1f}%)")
        print(f"   Ground truth: {validation_stats['ground_truth']}")
        print(f"   Continuous: {validation_stats['continuous']}")
        print(f"   Gap excluded: {validation_stats['gap_excluded']}")
        
        return cell_labels
    
    def update_and_save_results(self, cell_labels):
        """💾 Save results with metadata and create lineage files"""
        print(f"\n💾 STEP 7: SAVING RESULTS")
        
        # Update arranged.csv
        updated_df = self.arranged_df.copy()
        
        def get_label(cell_id):
            return 1 if cell_labels.get(cell_id, 'normal') == 'iPS' else 0
        
        updated_df['Label'] = updated_df['Cell ID'].apply(get_label)
        
        # Save CSV
        labelled_csv_path = os.path.join(self.labelled_path, "arranged_retrospective_labels.csv")
        updated_df.to_csv(labelled_csv_path, index=False)
        print(f"✅ Saved: {labelled_csv_path}")
        
        # Create comprehensive Excel lineage tree
        print(f"\n📊 CREATING LINEAGE TREE EXCEL...")
        excel_path = self.create_lineage_tree_excel(cell_labels)
        
        # Create detailed lineage CSV
        print(f"\n📋 CREATING DETAILED LINEAGE CSV...")
        detailed_csv_path = self.create_detailed_lineage_csv(cell_labels)
        
        # Create metadata
        ips_cells = [cid for cid, label in cell_labels.items() if label == 'iPS']
        
        metadata = {
            'fov': str(self.fov),
            'method': self.tracking_mode,
            'has_lineage_data': self.has_lineage_data,
            'ground_truth_blue_markers': sorted(list(self.blue_marker_cells)),
            'segmented_last_frame': sorted(list(self.segmented_cells_last_frame)),
            'backward_traced_lineages': len(self.backward_traced_lineages),
            'forward_traced_lineages': len(self.forward_traced_lineages),
            'spatial_tracked_lineages': len(self.spatial_tracked_lineages),
            'final_ips_lineages': sorted(ips_cells),
            'total_cells': len(cell_labels),
            'ips_lineage_count': len(ips_cells),
            'ips_percentage': (len(ips_cells) / len(cell_labels)) * 100 if cell_labels else 0,
            'failed_progenitors': sorted(list(self.failed_progenitors)),
            'validation_stats': dict(self.validation_stats),
            'validation_methodology': {
                'progressive_gap_tolerance': 'Frames 0-150: 1 gap, 150-303: 2 gaps, 303-400: 3 gaps',
                'missing_segmentation_tolerance': '20+ missing: 1 gap, 10-19: 2 gaps, <10: 3 gaps',
                'combined_pattern': 'Pattern-based validation with alternating checks',
                'final_tolerance': 'MIN(Progressive, Segmentation, Combined)'
            },
            'lineage_files_created': {
                'excel_path': excel_path,
                'detailed_csv_path': detailed_csv_path
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(self.labelled_path, "retrospective_metadata.txt")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"Lineage Tracking - FOV {self.fov}\n")
            f.write("="*80 + "\n")
            f.write(f"Method: {metadata['method']}\n")
            f.write(f"Has lineage data: {metadata['has_lineage_data']}\n\n")
            f.write(f"TEMPORAL VALIDATION METHODOLOGY:\n")
            f.write(f"  Progressive Gap Tolerance: {metadata['validation_methodology']['progressive_gap_tolerance']}\n")
            f.write(f"  Missing Segmentation Tolerance: {metadata['validation_methodology']['missing_segmentation_tolerance']}\n")
            f.write(f"  Combined Pattern: {metadata['validation_methodology']['combined_pattern']}\n")
            f.write(f"  Final Tolerance: {metadata['validation_methodology']['final_tolerance']}\n\n")
            f.write(f"Ground truth blue markers: {len(self.blue_marker_cells)}\n")
            f.write(f"Segmented cells: {len(self.segmented_cells_last_frame)}\n")
            f.write(f"Failed progenitors: {len(self.failed_progenitors)}\n\n")
            f.write(f"Backward traced: {metadata['backward_traced_lineages']}\n")
            f.write(f"Forward traced: {metadata['forward_traced_lineages']}\n")
            f.write(f"Spatial tracked: {metadata['spatial_tracked_lineages']}\n\n")
            f.write(f"Total cells: {metadata['total_cells']}\n")
            f.write(f"iPS cells: {metadata['ips_lineage_count']}\n")
            f.write(f"iPS percentage: {metadata['ips_percentage']:.1f}%\n\n")
            f.write(f"Validation stats: {metadata['validation_stats']}\n\n")
            f.write(f"LINEAGE FILES CREATED:\n")
            f.write(f"  📊 lineage_tree_retrospective.xlsx - Complete family tree with ancestry/descendancy\n")
            f.write(f"  📋 detailed_lineage.csv - Detailed tracking information with validation\n")
            f.write(f"  📄 arranged_retrospective_labels.csv - Updated cell labels\n")
            f.write(f"  📋 retrospective_metadata.txt - This summary\n\n")
            f.write(f"FIXES APPLIED:\n")
            f.write(f"  ✅ Excel sheets sorted by Cell_ID in ascending order\n")
            f.write(f"  ✅ detailed_lineage.csv 'Is iPS' correctly maps Label column (1='Yes', 0='No')\n")
            f.write(f"  ✅ All cell records sorted by Cell_ID for easy navigation\n")
        
        print(f"✅ Saved: {metadata_path}")
        
        return labelled_csv_path, metadata_path, metadata
    
    def run_complete_lineage_tracking(self, detection_start_frame=350, detection_end_frame=400):
        """🚀 Run complete tracking pipeline"""
        print("="*100)
        print(f"🧬 LINEAGE TRACKING - FOV {self.fov}")
        print("="*100)
        print("TEMPORAL VALIDATION METHODOLOGY (as per SVG diagram):")
        print("  Method 1: Progressive Gap Tolerance (frame-based)")
        print("  Method 2: Missing Segmentation Tolerance (missing-based)")
        print("  Method 3: Combined Pattern (pattern-based)")
        print("  Final Tolerance = MIN of all three methods")
        print("="*100)
        
        try:
            # Step 1: Find blue markers
            ground_truth_cells = self.find_blue_markers_ground_truth()
            
            # Step 2: Find segmented cells
            segmented_cells = self.find_segmented_cells_last_frame(detection_start_frame, detection_end_frame)
            
            # Step 2.5: Detect failed progenitors
            failed_progenitors = self.detect_failed_progenitors()
            
            if not segmented_cells and not ground_truth_cells:
                print("⚠️ No cells detected")
                cell_labels = {cell_id: 'normal' for cell_id in self.arranged_df['Cell ID'].unique()}
                metadata = {
                    'fov': str(self.fov),
                    'method': 'no_detection',
                    'has_lineage_data': self.has_lineage_data,
                    'total_cells': len(cell_labels),
                    'ips_lineage_count': 0,
                    'ips_percentage': 0.0
                }
                return False, metadata
            
            # Step 3: Backward tracing
            backward_lineages = self.backward_trace_complete_lineages(segmented_cells)
            
            # Step 4: Forward tracing
            forward_lineages = self.forward_trace_from_ground_truth(ground_truth_cells)
            
            # Step 5: Cross-validate
            final_lineages = self.cross_validate_and_combine()
            
            # Step 6: Create labels
            cell_labels = self.create_comprehensive_labels()
            
            # Step 7: Save results (includes Excel and CSV creation)
            csv_path, metadata_path, metadata = self.update_and_save_results(cell_labels)
            
            print(f"\n🎉 TRACKING COMPLETED!")
            print(f"📊 Summary:")
            print(f"   Ground truth: {len(ground_truth_cells)}")
            print(f"   Segmented: {len(segmented_cells)}")
            print(f"   Final iPS: {metadata['ips_lineage_count']} ({metadata['ips_percentage']:.1f}%)")
            print(f"   Method: {metadata['method']}")
            print(f"   Validation: Three-method temporal validation applied")
            print(f"📂 Results saved in: {self.labelled_path}")
            print(f"📊 Excel lineage tree: lineage_tree_retrospective.xlsx")
            print(f"📋 Detailed CSV: detailed_lineage.csv")
            print(f"✅ FIXES: Excel sorted by Cell_ID, CSV 'Is iPS' correctly mapped")
            
            return True, metadata
            
        except Exception as e:
            print(f"❌ Error in tracking: {e}")
            import traceback
            traceback.print_exc()
            return False, None


def label_single_fov_complete_tracking(base_path, fov, detection_start_frame=350, detection_end_frame=400):
    """Run tracking for single FOV"""
    try:
        labeler = iPSLabelerWithCompleteLineageTracking(base_path, fov)
        success, metadata = labeler.run_complete_lineage_tracking(detection_start_frame, detection_end_frame)
        return success, metadata
    except Exception as e:
        print(f"❌ Error with FOV {fov}: {e}")
        return False, None


def label_all_fovs_complete_tracking(base_path=".", detection_start_frame=350, detection_end_frame=400):
    """Run tracking for all FOVs"""
    nuclear_dataset_dir = os.path.join(base_path, "nuclear_dataset")
    
    if not os.path.exists(nuclear_dataset_dir):
        print(f"❌ Directory not found: {nuclear_dataset_dir}")
        return
    
    # Find FOVs
    fov_dirs = []
    for d in os.listdir(nuclear_dataset_dir):
        if os.path.isdir(os.path.join(nuclear_dataset_dir, d)) and d.isdigit():
            fov_num = int(d)
            if 2 <= fov_num <= 54:
                fov_dirs.append(d)
    
    fov_dirs = sorted(fov_dirs, key=int)
    print(f"📁 Found {len(fov_dirs)} FOVs")
    
    successful = 0
    
    for fov in fov_dirs:
        print(f"\n{'='*80}")
        print(f"🧬 Processing FOV {fov}")
        print(f"{'='*80}")
        
        success, metadata = label_single_fov_complete_tracking(base_path, fov, detection_start_frame, detection_end_frame)
        
        if success:
            successful += 1
            if metadata:
                ips_count = metadata.get('ips_lineage_count', 0)
                ips_pct = metadata.get('ips_percentage', 0)
                print(f"✅ FOV {fov}: {ips_count} iPS cells ({ips_pct:.1f}%)")
        else:
            print(f"❌ FOV {fov} failed")
    
    print(f"\n{'='*100}")
    print("🧬 ALL FOVs SUMMARY")
    print(f"{'='*100}")
    print(f"✅ Successfully processed: {successful}/{len(fov_dirs)} FOVs")
    print("Validation Methodology: Three-method temporal validation (Progressive, Segmentation, Combined)")
    print("📊 Excel lineage trees and detailed CSV files created for each FOV")
    print("✅ FIXES: Excel sorted by Cell_ID, CSV 'Is iPS' correctly mapped")


if __name__ == "__main__":
    print("🚀 Testing tracking with three-method temporal validation!")
    print("📊 Now includes lineage_tree_retrospective.xlsx and detailed_lineage.csv generation!")
    print("✅ FIXES: Excel sorted by Cell_ID, CSV 'Is iPS' correctly mapped")
    label_single_fov_complete_tracking(".", "2")
