#!/usr/bin/env python3
"""
Consensus-Based iPS Cell Labeling System with Backward Tracing

BACKWARD TRACING APPROACH:
1. CONSENSUS DATA SOURCE: Uses consensus_data.csv instead of arranged.csv
2. CONSENSUS TRACKING: Uses consensus/FOV/res_track.txt for lineage data
3. CONSENSUS OUTPUT: All results saved in consensus folder
4. FIRST FRAME PURITY: GT from first frame still maintained
5. BACKWARD TRACING: Uses preprocessed green segmentation as ground truth
   - Preprocessed green white regions = successfully induced iPS cells
   - Find cells at those positions in frames 275-400
   - Mark them as iPS (Label=1)
   - Track their ancestors backward
6. ADAPTIVE VALIDATION: Same three-method validation system
7. ROBUST GAP HANDLING: Same gap tolerance methodology
8. FAILED PROGENITOR DETECTION: Applied to consensus results
9. FALLBACK MECHANISMS: Multiple validation layers
"""

import os
import pandas as pd
import numpy as np
import cv2
from typing import List, Tuple, Set, Dict
from collections import defaultdict, deque
import csv

class ConsensusIPSLabeler:
    """
    Consensus-Based iPS Labeling System with Backward Tracing
    """
    
    def __init__(self, fov: str):
        self.fov = fov
        
        # PATHS FOR CONSENSUS
        self.consensus_path = os.path.join("consensus", fov)
        self.preprocessed_green_path = os.path.join(self.consensus_path, "Preprocessed_green")
        
        # Output paths (to consensus folder)
        self.labelled_path = os.path.join(self.consensus_path, "Labelled")
        os.makedirs(self.labelled_path, exist_ok=True)
        
        # Load consensus data
        self.consensus_df = self.load_consensus_csv()
        
        # Tracking data structures
        self.tracking_data = {}  # cell_id -> {start_frame, end_frame, parent_id}
        self.parent_to_children = defaultdict(list)  # parent_id -> [child_ids]
        self.child_to_parent = {}  # child_id -> parent_id
        self.frame_to_cells = defaultdict(set)  # frame -> set of cell_ids
        self.cell_to_frames = defaultdict(set)  # cell_id -> set of frames
        self.division_frames = {}  # Track division frames
        
        # Initialize tracking data
        self.missing_cells_tracker = []
        self.has_lineage_data = False
        self.tracking_mode = "unknown"
        
        self.load_consensus_tracking_data()
        
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
    
    def load_consensus_csv(self):
        """Load the consensus_data.csv file"""
        consensus_data_path = os.path.join(self.consensus_path, "consensus_data.csv")
        
        if not os.path.exists(consensus_data_path):
            raise FileNotFoundError(f"consensus_data.csv not found at {consensus_data_path}")
        
        df = pd.read_csv(consensus_data_path)
        print(f" Loaded consensus_data.csv: {len(df)} records")
        print(f" Current Label distribution: {df['Label'].value_counts().to_dict()}")
        
        return df
    
    def load_consensus_tracking_data(self):
        """Load tracking data from consensus res_track.txt"""
        consensus_track_path = os.path.join(self.consensus_path, "res_track.txt")
        
        if not os.path.exists(consensus_track_path):
            print(f"️ Warning: consensus res_track.txt not found at {consensus_track_path}")
            print("Will use spatial tracking fallback mode")
            self.tracking_mode = "no_tracking_file"
            return False
        
        print(" Loading consensus tracking data from res_track.txt...")
        
        try:
            with open(consensus_track_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f" Error reading consensus res_track.txt: {e}")
            self.tracking_mode = "error_reading_file"
            return False
        
        print(f" consensus res_track.txt has {len(lines)} lines")
        
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
                    if line_num < 10:
                        print(f"Error parsing line {line_num}: {e}")
                    continue
        
        print(f" Parsed {parsed_count} consensus tracking records")
        print(f" Found {parent_relationships} parent-child relationships")
        
        # Determine tracking mode
        if parent_relationships > 0:
            self.has_lineage_data = True
            self.tracking_mode = "consensus_lineage_tracking"
            if self.frame_to_cells:
                print(f" Frame coverage: {min(self.frame_to_cells.keys())} to {max(self.frame_to_cells.keys())}")
            print(f" Found {len(self.parent_to_children)} parent cells")
            print(f" MODE: CONSENSUS LINEAGE TRACKING")
        else:
            self.has_lineage_data = False
            self.tracking_mode = "consensus_spatial"
            print(f"️ WARNING: NO parent-child relationships found!")
            print(f" MODE: CONSENSUS SPATIAL TRACKING (fallback)")
        
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
            print(f"️ Error loading {image_path}: {e}")
            return None
    
    def calculate_gap_pattern(self, missing_frames):
        """Calculate gap pattern for Combined Pattern validation"""
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
            return 0.8, True, f"Single gap of {gaps[0]} frames"
        
        # Check for alternating pattern
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
        📄 Three-method temporal validation for consensus data
        Same methodology but applied to consensus results
        """
        # Check cache first
        if cell_id in self.middle_frame_validation_cache:
            return self.middle_frame_validation_cache[cell_id]
        
        # Default validation for cells not in tracking data
        if cell_id not in self.tracking_data:
            result = (False, "No consensus tracking data")
            self.middle_frame_validation_cache[cell_id] = result
            return result
        
        try:
            track_info = self.tracking_data[cell_id]
            start_frame = track_info['start_frame']
            end_frame = track_info['end_frame']
            
            # Calculate expected and actual frames
            expected_frames = set(range(start_frame, end_frame + 1))
            actual_frames = self.cell_to_frames.get(cell_id, set())
            
            # Calculate missing frames and gaps
            missing_frames = expected_frames - actual_frames
            num_missing = len(missing_frames)
            
            # Special case: cells in frame 0 (ground truth)
            if start_frame == 0 and cell_id in self.blue_marker_cells:
                validation_passed = True
                validation_reason = "Consensus ground truth cell (frame 0)"
                self.continuous_lineages.add(cell_id)
                result = (validation_passed, validation_reason)
                self.middle_frame_validation_cache[cell_id] = result
                return result
            
            # Calculate gaps
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
            
            # METHOD 1: Progressive Gap Tolerance
            avg_frame = (start_frame + end_frame) / 2
            if avg_frame <= 150:
                progressive_tolerance = 1
            elif avg_frame <= 303:
                progressive_tolerance = 2
            else:
                progressive_tolerance = 3
            
            progressive_valid = num_gaps <= progressive_tolerance
            
            # METHOD 2: Missing Segmentation Tolerance
            if num_missing >= 20:
                segmentation_tolerance = 1
            elif num_missing >= 10:
                segmentation_tolerance = 2
            else:
                segmentation_tolerance = 3
            
            segmentation_valid = num_gaps <= segmentation_tolerance
            
            # METHOD 3: Combined Pattern validation
            pattern_score, pattern_valid, pattern_reason = self.calculate_gap_pattern(missing_frames)
            
            if pattern_score >= 0.9:
                pattern_tolerance = 3
            elif pattern_score >= 0.7:
                pattern_tolerance = 2
            else:
                pattern_tolerance = 1
            
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
            print(f" Error validating cell {cell_id}: {e}")
            result = (False, f"Validation error: {e}")
            self.middle_frame_validation_cache[cell_id] = result
            return result
    
    def calculate_cell_properties(self, image, cell_mask, voxel_size=(1, 1)):
        """Calculate cell properties (area, volume, centroid) - same as features.py"""
        if not np.any(cell_mask):
            return None
        
        # Get coordinates of cell pixels
        coords = np.column_stack(np.where(cell_mask))
        
        # Calculate volume (area in 2D)
        volume = np.sum(cell_mask) * np.prod(voxel_size)
        
        # Calculate area (perimeter-based area for consistency)
        uint8_mask = cell_mask.astype(np.uint8)
        contours, _ = cv2.findContours(uint8_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours and len(contours[0]) >= 5:
            perimeter = cv2.arcLength(contours[0], True)
            area = perimeter * np.mean(voxel_size)
        else:
            area = volume  # Fallback
        
        # Calculate centroid (Y, X format to match features.py)
        if coords.size > 0:
            centroid = np.mean(coords, axis=0)  # Returns [Y, X]
        else:
            centroid = np.array([0, 0])
        
        return {
            'area': area,
            'volume': volume,
            'centroid': centroid,  # [Y, X] format
            'x': centroid[1],  # X coordinate
            'y': centroid[0],  # Y coordinate
        }
    
    def find_blue_markers_ground_truth(self):
        """ Find blue markers from consensus data frame 0"""
        print(f"\n STEP 1: FINDING CONSENSUS BLUE MARKERS (GROUND TRUTH)")
        
        # Get blue markers from consensus_data.csv (Label = 1) in frame 0
        frame_0_data = self.consensus_df[self.consensus_df['Frame'] == 0]
        blue_markers = frame_0_data[frame_0_data['Label'] == 1]['Cell ID'].unique()
        self.blue_marker_cells = set(blue_markers)
        
        print(f" Found {len(self.blue_marker_cells)} consensus blue marker cells")
        
        # Validate blue markers in frame 0
        frame_0_blue_markers = set()
        if len(self.blue_marker_cells) > 0:
            blue_df = self.consensus_df[self.consensus_df['Cell ID'].isin(self.blue_marker_cells)]
            
            # Check frame 0 specifically
            frame_0_candidates = blue_df[blue_df['Frame'] == 0]['Cell ID'].unique()
            frame_0_blue_markers.update(frame_0_candidates)
            
            if frame_0_blue_markers:
                print(f" Frame 0 consensus blue markers: {len(frame_0_blue_markers)} cells")
                self.blue_marker_cells = frame_0_blue_markers
            else:
                print(f"️ No consensus blue markers in frame 0, using all blue markers")
        
        return self.blue_marker_cells
    
    def find_segmented_cells_last_frame_fixed(self, detection_start_frame=275, detection_end_frame=400):
        """
        Find WHITE preprocessed cells using direct segmentation overlap
        MATCHES THE ORIGINAL labeling__.py APPROACH WITH STRICTER CRITERIA
        """
        print(f"\n STEP 2: FINDING iPS CELLS FROM PREPROCESSED GREEN SEGMENTATION")
        print(f" USING DIRECT SEGMENTATION OVERLAP (ORIGINAL METHOD)")
        print(f" STRICT CRITERIA: ≥2 frames, score≥15, ratio≥0.4")
        
        if not os.path.exists(self.preprocessed_green_path):
            print(f"️ Consensus Preprocessed_green folder not found: {self.preprocessed_green_path}")
            return set()
        
        # Get preprocessed files from consensus
        preprocessed_files = sorted([f for f in os.listdir(self.preprocessed_green_path) 
                                    if f.endswith('.tif') or f.endswith('.tiff')])
        
        if not preprocessed_files:
            print(f"️ No preprocessed files found in consensus")
            return set()
        
        ips_candidates = set()
        detection_scores = defaultdict(list)
        frames_processed = 0
        
        print(f" Found {len(preprocessed_files)} preprocessed files")
        print(f" Processing frames {detection_start_frame} to {detection_end_frame}")
        
        # Process detection frames using DIRECT SEGMENTATION OVERLAP
        for frame_idx in range(detection_start_frame, min(detection_end_frame + 1, 401)):
            preprocessed_frame_idx = frame_idx - 275  # Preprocessed starts at frame 275
            
            if preprocessed_frame_idx < 0 or preprocessed_frame_idx >= len(preprocessed_files):
                continue
            
            try:
                # Load preprocessed segmentation from consensus
                preprocessed_file_path = os.path.join(self.preprocessed_green_path, preprocessed_files[preprocessed_frame_idx])
                preprocessed_img = self.load_image_robust(preprocessed_file_path)
                
                if preprocessed_img is None:
                    continue
                
                frames_processed += 1
                
                # ORIGINAL APPROACH: Detect white cells with adaptive thresholding
                if len(preprocessed_img.shape) > 2:
                    white_threshold = 240  # High threshold for bright/white regions
                    white_mask = np.all(preprocessed_img >= white_threshold, axis=2).astype(np.uint8) * 255
                    
                    # Also check brightness in grayscale
                    gray_preprocessed = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
                    bright_mask = (gray_preprocessed >= white_threshold).astype(np.uint8) * 255
                    white_mask = cv2.bitwise_or(white_mask, bright_mask)
                else:
                    white_threshold = 240
                    white_mask = (preprocessed_img >= white_threshold).astype(np.uint8) * 255
                
                # Get consensus cells in this frame
                frame_consensus_data = self.consensus_df[self.consensus_df['Frame'] == frame_idx]
                
                if len(frame_consensus_data) == 0:
                    continue
                
                # Check each consensus cell for overlap with white regions
                for _, row in frame_consensus_data.iterrows():
                    cell_id = row['Cell ID']
                    cell_x = row['X']
                    cell_y = row['Y']
                    
                    # Create a small region around the cell position to check for white overlap
                    # Use a conservative radius to avoid false positives
                    radius = 15
                    y_start = max(0, int(cell_y) - radius)
                    y_end = min(white_mask.shape[0], int(cell_y) + radius)
                    x_start = max(0, int(cell_x) - radius)
                    x_end = min(white_mask.shape[1], int(cell_x) + radius)
                    
                    if y_start >= y_end or x_start >= x_end:
                        continue
                    
                    # Extract region around cell
                    cell_region = white_mask[y_start:y_end, x_start:x_end]
                    
                    # Calculate white pixel ratio in the region
                    total_pixels = cell_region.size
                    if total_pixels == 0:
                        continue
                    
                    white_pixels = np.sum(cell_region == 255)
                    white_ratio = white_pixels / total_pixels
                    
                    # STRICT threshold: require significant white presence
                    if white_ratio >= 0.3:  # 30% of region must be white
                        score = white_ratio * white_pixels
                        detection_scores[cell_id].append((frame_idx, score, white_ratio))
                        
            except Exception as e:
                if frame_idx == detection_start_frame:
                    print(f" Error processing frame {frame_idx}: {e}")
                continue
        
        # STRICT Validation: Require multiple detections with high confidence
        validated_ips_cells = set()
        
        for cell_id, scores in detection_scores.items():
            if len(scores) >= 2:  # REQUIRED: Must be detected in at least 2 frames
                avg_score = np.mean([s[1] for s in scores])
                avg_white_ratio = np.mean([s[2] for s in scores])
                
                # STRICT thresholds
                if avg_score >= 15.0 and avg_white_ratio >= 0.4:
                    validated_ips_cells.add(cell_id)
                    print(f" Validated iPS cell {cell_id}: {len(scores)} frames, score={avg_score:.1f}, ratio={avg_white_ratio:.2f}")
        
        self.segmented_cells_last_frame = validated_ips_cells
        
        print(f"\n COMPLETE PREPROCESSED DETECTION RESULTS:")
        print(f"   Frames processed: {frames_processed}")
        print(f"   Cell candidates detected: {len(detection_scores)}")
        print(f"   FINAL Validated iPS cells: {len(validated_ips_cells)}")
        print(f"    Used STRICT criteria: ≥2 frames, score≥15, ratio≥0.4")
        print(f"    These represent high-confidence iPS clusters")
        
        return validated_ips_cells
    
    def detect_failed_progenitors(self):
        """ Detect consensus GT cells that don't connect to preprocessed cells"""
        print(f"\n DETECTING FAILED PROGENITORS IN CONSENSUS DATA")
        
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
            print(f" Consensus Progenitor Analysis:")
            print(f"    Successful: {len(successful_progenitors)} ({success_rate:.1f}%)")
            print(f"    Failed: {len(failed_progenitors)}")
        
        return failed_progenitors
    
    def backward_trace_complete_lineages(self, seed_cells):
        """  Backward trace from seed cells with STRICTER validation"""
        print(f"\n STEP 3: BACKWARD TRACING (CONSENSUS) - STRICT VALIDATION")
        print(f" Starting from {len(seed_cells)} HIGH-CONFIDENCE iPS seed cells")
        
        if not self.has_lineage_data:
            print("️ No consensus lineage data - using spatial fallback")
            return self.spatial_tracking_fallback(seed_cells)
        
        backward_lineages = set()
        frame_0_candidates = set()
        
        print(" Tracing ancestors for each HIGH-CONFIDENCE seed cell...")
        
        for seed_cell in seed_cells:
            try:
                # STRICT: Only trace if seed cell has high confidence
                if seed_cell not in self.segmented_cells_last_frame:
                    continue
                    
                print(f"   Processing HIGH-CONFIDENCE seed cell {seed_cell}...")
                
                lineage = self.get_complete_backward_lineage(seed_cell)
                
                print(f"   Found {len(lineage)} cells in lineage of {seed_cell}")
                
                # Validate each cell in lineage
                for cell_id in lineage:
                    if cell_id in self.tracking_data:
                        # STRICT: Only include cells with good continuity
                        is_continuous, reason = self.validate_middle_frame_continuity(cell_id)
                        if is_continuous:
                            if self.tracking_data[cell_id]['start_frame'] == 0:
                                frame_0_candidates.add(cell_id)
                            else:
                                backward_lineages.add(cell_id)
                
            except Exception as e:
                print(f"️ Error tracing cell {seed_cell}: {e}")
        
        # EXTRA STRICT: Frame 0 validation
        validated_frame_0 = frame_0_candidates.intersection(self.blue_marker_cells)
        
        if validated_frame_0:
            print(f" Frame 0 validated: {len(validated_frame_0)} consensus cells")
            backward_lineages.update(validated_frame_0)
        
        rejected_frame_0 = frame_0_candidates - self.blue_marker_cells
        if rejected_frame_0:
            print(f" Frame 0 rejected: {len(rejected_frame_0)} cells (not consensus ground truth)")
        
        self.backward_traced_lineages = backward_lineages
        print(f" STRICT BACKWARD TRACING: {len(backward_lineages)} validated cells")
        
        return backward_lineages
    
    def forward_trace_from_ground_truth(self, ground_truth_cells):
        """ Forward trace from consensus ground truth cells"""
        print(f"\n STEP 4: FORWARD TRACING (CONSENSUS)")
        
        if not self.has_lineage_data:
            print("️ No consensus lineage data - using spatial fallback")
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
                print(f" Error tracing cell {gt_cell}: {e}")
                forward_lineages.add(gt_cell)
        
        self.forward_traced_lineages = forward_lineages
        print(f" Consensus forward traced: {len(forward_lineages)} cells")
        
        return forward_lineages
    
    def cross_validate_and_combine(self):
        """ Cross-validate and combine consensus results"""
        print(f"\n STEP 5: CROSS-VALIDATION (CONSENSUS)")
        
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
        
        print(f" Consensus Validation:")
        print(f"   Both methods: {len(both_methods)} cells")
        print(f"   Only backward: {len(only_backward)} cells")
        print(f"   Only forward: {len(only_forward)} cells")
        
        # Combine results - prioritize backward tracing 
        combined_lineages = backward_set.union(forward_set)
        
        # Add segmented cells (our iPS ground truth from preprocessed)
        for cell_id in self.segmented_cells_last_frame:
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
            elif cell_id in self.consensus_df['Cell ID'].values:
                validated_lineages.add(cell_id)
        
        self.final_ips_lineages = validated_lineages
        print(f" Final consensus validated: {len(validated_lineages)} cells")
        print(f" Includes {len(self.segmented_cells_last_frame)} direct iPS cells from preprocessed")
        
        return validated_lineages
    
    def spatial_tracking_fallback(self, seed_cells, max_distance=50):
        """ Spatial tracking fallback for consensus data"""
        print(f"\n CONSENSUS SPATIAL TRACKING FALLBACK")
        
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
        print(f" Consensus spatially tracked: {len(spatial_lineages)} cells")
        
        return spatial_lineages
    
    def get_spatial_lineage(self, seed_cell, max_distance=50):
        """Get lineage using spatial proximity in consensus data"""
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
        """Find nearby cells in frame using consensus data"""
        nearby = set()
        
        try:
            ref_center = self.get_cell_center_in_frame_from_consensus(reference_cell, frame)
            if ref_center is None:
                return nearby
            
            frame_cells = self.frame_to_cells.get(frame, set())
            for cell_id in frame_cells:
                if cell_id == reference_cell:
                    continue
                
                cell_center = self.get_cell_center_in_frame_from_consensus(cell_id, frame)
                if cell_center is None:
                    continue
                
                distance = np.sqrt((ref_center[0] - cell_center[0])**2 + 
                                 (ref_center[1] - cell_center[1])**2)
                
                if distance <= max_distance:
                    nearby.add(cell_id)
        
        except Exception:
            pass
        
        return nearby
    
    def get_cell_center_in_frame_from_consensus(self, cell_id, frame):
        """Get cell center in specific frame using consensus data"""
        try:
            frame_data = self.consensus_df[(self.consensus_df['Cell ID'] == cell_id) & 
                                          (self.consensus_df['Frame'] == frame)]
            
            if len(frame_data) == 0:
                return None
            
            # Get X, Y from consensus data
            x = frame_data['X'].iloc[0]
            y = frame_data['Y'].iloc[0]
            
            return (x, y)
        
        except Exception:
            return None
    
    def get_complete_backward_lineage(self, cell_id):
        """Get complete backward lineage from consensus data"""
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
        """Get complete forward lineage from consensus data"""
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
        """Get all ancestor cells from consensus data"""
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
        """Get all descendant cells from consensus data"""
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
    
    def create_comprehensive_labels(self):
        """️ Create comprehensive labels for consensus data"""
        print(f"\n️ STEP 6: CREATING CONSENSUS LABELS")
        
        # Get all consensus cells
        all_cells = set(self.tracking_data.keys())
        all_cells.update(self.consensus_df['Cell ID'].unique())
        
        cell_labels = {}
        validation_stats = defaultdict(int)
        
        for cell_id in all_cells:
            label = 'normal'
            
            if cell_id in self.final_ips_lineages:
                # Validate the cell
                if cell_id in self.blue_marker_cells:
                    label = 'iPS'
                    validation_stats['ground_truth'] += 1
                elif cell_id in self.segmented_cells_last_frame:
                    label = 'iPS'
                    validation_stats['preprocessed_detection'] += 1
                elif cell_id in self.tracking_data:
                    is_continuous, reason = self.validate_middle_frame_continuity(cell_id)
                    if is_continuous:
                        label = 'iPS'
                        validation_stats['continuous'] += 1
                    else:
                        label = 'normal'
                        validation_stats['gap_excluded'] += 1
                elif cell_id in self.consensus_df['Cell ID'].values:
                    label = 'iPS'
                    validation_stats['other'] += 1
            
            cell_labels[cell_id] = label
            if label == 'iPS':
                validation_stats['total_ips'] += 1
        
        ips_count = validation_stats['total_ips']
        total_count = len(cell_labels)
        
        print(f" CONSENSUS LABELING RESULTS:")
        print(f"   Total cells: {total_count}")
        print(f"   iPS cells: {ips_count} ({ips_count/total_count*100:.1f}%)")
        print(f"   Ground truth: {validation_stats['ground_truth']}")
        print(f"   Preprocessed detection: {validation_stats['preprocessed_detection']}")
        print(f"   Continuous: {validation_stats['continuous']}")
        print(f"   Gap excluded: {validation_stats['gap_excluded']}")
        
        return cell_labels
    
    def update_and_save_results(self, cell_labels):
        """ Save consensus results with metadata and create lineage files"""
        print(f"\n STEP 7: SAVING CONSENSUS RESULTS")
        
        # Update consensus_data.csv
        updated_df = self.consensus_df.copy()
        
        def get_label(cell_id):
            return 1 if cell_labels.get(cell_id, 'normal') == 'iPS' else 0
        
        updated_df['Label'] = updated_df['Cell ID'].apply(get_label)
        
        # Save updated consensus CSV
        labelled_csv_path = os.path.join(self.labelled_path, "consensus_retrospective_labels.csv")
        updated_df.to_csv(labelled_csv_path, index=False)
        print(f" Saved: {labelled_csv_path}")
        
        # Create comprehensive Excel lineage tree
        print(f"\n CREATING CONSENSUS LINEAGE TREE EXCEL...")
        excel_path = self.create_lineage_tree_excel(cell_labels)
        
        # Create detailed lineage CSV
        print(f"\n CREATING CONSENSUS DETAILED LINEAGE CSV...")
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
            'fixed_backward_tracing': {
                'method': 'Preprocessed green segmentation as iPS ground truth',
                'detection_frames': '275-400',
                'white_regions_detected': len(self.segmented_cells_last_frame),
                'approach': 'White regions → Cell positions → Ancestor tracing',
                'strict_criteria': '≥2 frames, score≥15, ratio≥0.4'
            },
            'lineage_files_created': {
                'excel_path': excel_path,
                'detailed_csv_path': detailed_csv_path
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(self.labelled_path, "consensus_retrospective_metadata.txt")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"Consensus Lineage Tracking - FOV {self.fov}\n")
            f.write("="*80 + "\n")
            f.write(f"Method: {metadata['method']}\n")
            f.write(f"Has lineage data: {metadata['has_lineage_data']}\n\n")
            f.write(f"BACKWARD TRACING WITH STRICT CRITERIA:\n")
            f.write(f"  Method: {metadata['fixed_backward_tracing']['method']}\n")
            f.write(f"  Detection frames: {metadata['fixed_backward_tracing']['detection_frames']}\n")
            f.write(f"  White regions found: {metadata['fixed_backward_tracing']['white_regions_detected']}\n")
            f.write(f"  Strict criteria: {metadata['fixed_backward_tracing']['strict_criteria']}\n")
            f.write(f"  Approach: {metadata['fixed_backward_tracing']['approach']}\n\n")
            f.write(f"TEMPORAL VALIDATION METHODOLOGY:\n")
            f.write(f"  Progressive Gap Tolerance: {metadata['validation_methodology']['progressive_gap_tolerance']}\n")
            f.write(f"  Missing Segmentation Tolerance: {metadata['validation_methodology']['missing_segmentation_tolerance']}\n")
            f.write(f"  Combined Pattern: {metadata['validation_methodology']['combined_pattern']}\n")
            f.write(f"  Final Tolerance: {metadata['validation_methodology']['final_tolerance']}\n\n")
            f.write(f"Ground truth blue markers: {len(self.blue_marker_cells)}\n")
            f.write(f"Preprocessed iPS detection: {len(self.segmented_cells_last_frame)}\n")
            f.write(f"Failed progenitors: {len(self.failed_progenitors)}\n\n")
            f.write(f"Backward traced: {metadata['backward_traced_lineages']}\n")
            f.write(f"Forward traced: {metadata['forward_traced_lineages']}\n")
            f.write(f"Spatial tracked: {metadata['spatial_tracked_lineages']}\n\n")
            f.write(f"Total cells: {metadata['total_cells']}\n")
            f.write(f"iPS cells: {metadata['ips_lineage_count']}\n")
            f.write(f"iPS percentage: {metadata['ips_percentage']:.1f}%\n\n")
            f.write(f"Validation stats: {metadata['validation_stats']}\n\n")
            f.write(f"CONSENSUS FILES CREATED:\n")
            f.write(f"   consensus_lineage_tree_retrospective.xlsx - Complete family tree\n")
            f.write(f"   consensus_detailed_lineage.csv - Detailed tracking information\n")
            f.write(f"   consensus_retrospective_labels.csv - Updated consensus labels\n")
            f.write(f"   consensus_retrospective_metadata.txt - This summary\n\n")
            f.write(f"CONSENSUS MODIFICATIONS:\n")
            f.write(f"   Uses consensus_data.csv as input instead of arranged.csv\n")
            f.write(f"   Uses consensus lineage data for tracking\n")
            f.write(f"   All results saved in consensus folder\n")
            f.write(f"   Backward tracing uses preprocessed segmentation\n")
            f.write(f"   STRICT: ≥2 frames, score≥15, ratio≥0.4 criteria\n")
            f.write(f"   White regions in preprocessed = iPS ground truth\n")
        
        print(f" Saved: {metadata_path}")
        
        return labelled_csv_path, metadata_path, metadata
    
    def create_lineage_tree_excel(self, cell_labels):
        """
         Create Excel lineage tree showing complete ancestry and descendancy for each cell
        COMPREHENSIVE VERSION with error handling and ASCENDING CELL ID SORTING for consensus data
        """
        print(f"\n Creating consensus Excel lineage tree with complete ancestry/descendancy...")
        
        # Check dependencies first
        try:
            import pandas as pd
        except ImportError:
            print(" ERROR: pandas not installed. Install with: pip install pandas")
            return None
        
        try:
            import openpyxl
        except ImportError:
            print(" ERROR: openpyxl not installed. Install with: pip install openpyxl")
            return None
        
        # Ensure labelled directory exists
        try:
            os.makedirs(self.labelled_path, exist_ok=True)
            print(f" Excel will be saved to: {self.labelled_path}")
        except Exception as e:
            print(f" ERROR: Cannot create directory {self.labelled_path}: {e}")
            return None
        
        # Validate input data
        if not self.tracking_data:
            print(" ERROR: No consensus tracking data available for Excel export")
            return None
        
        if not cell_labels:
            print(" ERROR: No cell labels available for Excel export")
            return None
        
        try:
            # Create comprehensive lineage tree data
            lineage_data = []
            print(f" Processing {len(self.tracking_data)} consensus cells for Excel export...")
            
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
                        'Generation_Level': 0,
                        'Total_Ancestors': 0,
                        'Total_Descendants': 0,
                        'Algorithms_Agreeing': 1  # Consensus always has 1
                    }
                    
                    # Calculate direct children - with error handling
                    try:
                        if cell_id in self.parent_to_children:
                            children = sorted([int(child) for child in self.parent_to_children[cell_id]])
                            cell_info['Direct_Children'] = ' → '.join(map(str, children))
                        else:
                            cell_info['Direct_Children'] = 'No_Children'
                    except Exception as e:
                        print(f"️ Warning: Error processing children for cell {cell_id}: {e}")
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
                        
                    except Exception as e:
                        print(f"️ Warning: Error processing ancestry for cell {cell_id}: {e}")
                        cell_info['Generation_Level'] = 0
                        cell_info['Total_Ancestors'] = 0
                    
                    # Calculate complete descendancy tree - with error handling
                    try:
                        descendants = list(self.get_all_descendants(cell_id))
                        cell_info['Total_Descendants'] = len(descendants)
                        
                    except Exception as e:
                        print(f" Warning: Error processing descendants for cell {cell_id}: {e}")
                        cell_info['Total_Descendants'] = 0
                    
                    lineage_data.append(cell_info)
                    processed_count += 1
                    
                    # Progress indicator
                    if processed_count % 100 == 0:
                        print(f"    Processed {processed_count}/{len(self.tracking_data)} cells...")
                        
                except Exception as e:
                    print(f"️ Warning: Error processing cell {cell_id}: {e}")
                    # Add basic cell info even if processing fails
                    lineage_data.append({
                        'Cell_ID': int(cell_id),
                        'Label': 'Error',
                        'Start_Frame': 0,
                        'End_Frame': 0,
                        'Lifespan': 0,
                        'Direct_Parent': 'Error',
                        'Direct_Children': 'Error',
                        'Generation_Level': 0,
                        'Total_Ancestors': 0,
                        'Total_Descendants': 0,
                        'Algorithms_Agreeing': 1
                    })
                    continue
            
            print(f" Successfully processed {processed_count} consensus cells")
            
            # Validate we have data
            if not lineage_data:
                print(" ERROR: No lineage data created")
                return None
            
            # Create DataFrame with error handling
            try:
                lineage_df = pd.DataFrame(lineage_data)
                print(f" Created DataFrame with {len(lineage_df)} rows and {len(lineage_df.columns)} columns")
            except Exception as e:
                print(f" ERROR: Failed to create DataFrame: {e}")
                return None
            
            # Sort by Cell_ID in ASCENDING order for better readability
            try:
                lineage_df = lineage_df.sort_values(['Cell_ID'])
                print(" DataFrame sorted by Cell_ID in ascending order")
            except Exception as e:
                print(f"️ Warning: Could not sort DataFrame: {e}")
            
            # Create Excel file path
            excel_path = os.path.join(self.labelled_path, "consensus_lineage_tree_retrospective.xlsx")
            print(f" Saving Excel file to: {excel_path}")
            
            # Save as Excel file with multiple sheets - with comprehensive error handling
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    print(" Creating Excel sheets...")
                    
                    # Main lineage tree sheet
                    try:
                        lineage_df.to_excel(writer, sheet_name='Complete_Lineage_Tree', index=False)
                        print("    Complete_Lineage_Tree sheet created")
                    except Exception as e:
                        print(f"    Error creating Complete_Lineage_Tree: {e}")
                    
                    # iPS lineages only sheet - ALSO SORTED BY CELL_ID
                    try:
                        ips_lineage_df = lineage_df[lineage_df['Label'] == 'iPS'].copy().sort_values(['Cell_ID'])
                        ips_lineage_df.to_excel(writer, sheet_name='iPS_Lineages_Only', index=False)
                        print(f"    iPS_Lineages_Only sheet created ({len(ips_lineage_df)} iPS cells)")
                    except Exception as e:
                        print(f"    Error creating iPS_Lineages_Only: {e}")
                    
                    # Root cells sheet (generation 0) - SORTED BY CELL_ID
                    try:
                        root_cells_df = lineage_df[lineage_df['Generation_Level'] == 0].copy().sort_values(['Cell_ID'])
                        root_cells_df.to_excel(writer, sheet_name='Root_Cells_Gen0', index=False)
                        print(f"    Root_Cells_Gen0 sheet created ({len(root_cells_df)} root cells)")
                    except Exception as e:
                        print(f"    Error creating Root_Cells_Gen0: {e}")
                    
                    # Terminal cells sheet (no descendants) - SORTED BY CELL_ID
                    try:
                        terminal_cells_df = lineage_df[lineage_df['Total_Descendants'] == 0].copy().sort_values(['Cell_ID'])
                        terminal_cells_df.to_excel(writer, sheet_name='Terminal_Cells', index=False)
                        print(f"    Terminal_Cells sheet created ({len(terminal_cells_df)} terminal cells)")
                    except Exception as e:
                        print(f"    Error creating Terminal_Cells: {e}")
                    
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
                                'Total Consensus Cells Tracked',
                                'iPS Cells Identified',
                                'Normal Cells',
                                'iPS Percentage',
                                'Root Cells (Generation 0)',
                                'Terminal Cells (No Descendants)',
                                'Maximum Generation Level',
                                'Average Family Size',
                                'Largest Family Tree Size',
                                'Consensus Tracking Method Used',
                                'Excel Creation Date',
                                'FOV Number',
                                'Validation Method Applied',
                                'Cell ID Sorting',
                                'Data Source',
                                'Input File Used',
                                'Backward Tracing Method',
                                'iPS Detection Source',
                                'Strict Detection Criteria'
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
                                getattr(self, 'tracking_mode', 'consensus_tracking'),
                                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                                str(getattr(self, 'fov', 'unknown')),
                                'Three-method temporal validation (Progressive, Segmentation, Combined)',
                                'Cell IDs sorted in ascending order',
                                'Consensus Results',
                                'consensus_data.csv',
                                'Preprocessed green segmentation as ground truth',
                                'White regions in preprocessed images',
                                '≥2 frames, score≥15, ratio≥0.4'
                            ]
                        }
                        
                        summary_df = pd.DataFrame(summary_data)
                        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                        print("    Summary_Statistics sheet created")
                    except Exception as e:
                        print(f"    Error creating Summary_Statistics: {e}")
                
                # Verify file was created
                if os.path.exists(excel_path):
                    file_size = os.path.getsize(excel_path)
                    print(f" Consensus Excel lineage tree saved successfully!")
                    print(f" File location: {excel_path}")
                    print(f" File size: {file_size:,} bytes")
                    print(f" Contains {len(lineage_df)} consensus cells with complete lineage information")
                    print(f" Applied three-method temporal validation for accuracy")
                    print(f" Cell IDs sorted in ascending order for easy navigation")
                    print(f" Based on consensus data with  backward tracing")
                    print(f" STRICT CRITERIA: ≥2 frames, score≥15, ratio≥0.4")
                    
                    # Additional statistics
                    if len(lineage_df) > 0:
                        ips_count = len(lineage_df[lineage_df['Label'] == 'iPS'])
                        generations = lineage_df['Generation_Level'].nunique()
                        root_families = len(lineage_df[lineage_df['Generation_Level'] == 0])
                        max_gen = lineage_df['Generation_Level'].max()
                        
                        print(f" Consensus iPS lineages: {ips_count} cells across {generations} generations")
                        print(f" Family trees: {root_families} root families, max {max_gen} generations deep")
                        print(f" Uses preprocessed segmentation as iPS ground truth")
                    
                    return excel_path
                else:
                    print(f" ERROR: Excel file was not created at {excel_path}")
                    return None
                    
            except PermissionError as e:
                print(f" ERROR: Permission denied when saving Excel file: {e}")
                print(f" Try running as administrator or check file permissions for: {excel_path}")
                return None
            except Exception as e:
                print(f" ERROR: Failed to save Excel file: {e}")
                print(f" Check disk space and write permissions for: {self.labelled_path}")
                return None
                
        except Exception as e:
            print(f" ERROR: Unexpected error in Excel creation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_detailed_lineage_csv(self, cell_labels):
        """
         Create detailed lineage CSV similar to count.py output
        COMPREHENSIVE VERSION with correct "Is iPS" mapping and missing cell detection for consensus data
        """
        print(f"\n Creating consensus detailed lineage CSV...")
        
        try:
            # First, load the updated consensus CSV to get the correct Label values
            updated_csv_path = os.path.join(self.labelled_path, "consensus_retrospective_labels.csv")
            
            # Create a mapping from Cell_ID to Label for quick lookup
            cell_id_to_label = {}
            if os.path.exists(updated_csv_path):
                try:
                    updated_df = pd.read_csv(updated_csv_path)
                    for _, row in updated_df.iterrows():
                        cell_id_to_label[row['Cell ID']] = row['Label']
                    print(f" Loaded consensus label mapping from updated CSV: {len(cell_id_to_label)} mappings")
                except Exception as e:
                    print(f" Warning: Could not load updated consensus CSV for label mapping: {e}")
                    print("Will use cell_labels dictionary instead")
            
            # Prepare detailed lineage data
            detailed_lineage_data = []
            
            # Process each cell in consensus tracking data
            for cell_id in sorted(self.tracking_data.keys()):
                try:
                    track_info = self.tracking_data[cell_id]
                    start_frame = track_info['start_frame']
                    end_frame = track_info['end_frame']
                    parent_id = track_info['parent_id']
                    
                    # Determine if cell is iPS based on Label column from consensus CSV
                    is_ips_value = "No"  # Default
                    
                    # First try to get from updated consensus CSV (most accurate)
                    if cell_id in cell_id_to_label:
                        label_value = cell_id_to_label[cell_id]
                        is_ips_value = "Yes" if label_value == 1 else "No"
                    else:
                        # Fallback to cell_labels dictionary
                        is_ips_value = "Yes" if cell_labels.get(cell_id, 'normal') == 'iPS' else "No"
                    
                    # Check if cell has children
                    has_children = cell_id in self.parent_to_children and len(self.parent_to_children[cell_id]) > 0
                    
                    # Get children IDs
                    children_ids = []
                    if cell_id in self.parent_to_children:
                        children_ids = sorted([int(child) for child in self.parent_to_children[cell_id]])
                    
                    # Get validation information
                    is_continuous, validation_reason = self.validate_middle_frame_continuity(cell_id)
                    
                    # Create detailed entry with CLEANED COLUMNS (removed specified columns)
                    detailed_entry = {
                        "Cell ID": int(cell_id),
                        "Start Frame": int(start_frame),
                        "End Frame": int(end_frame),
                        "Parent ID": int(parent_id) if parent_id != 0 else 0,
                        "Is iPS": is_ips_value,  # Now correctly maps to Label column
                        "Has Children": "Yes" if has_children else "No",
                        "Children IDs": ",".join(map(str, children_ids)) if children_ids else "None",
                        "Generation Level": 0,
                        "Total Ancestors": 0,
                        "Total Descendants": len(children_ids) if has_children else 0,
                        "Track Length": end_frame - start_frame + 1,
                        "Division Frame": "N/A",
                        "In Preprocessed Detection": "Yes" if cell_id in self.segmented_cells_last_frame else "No"
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
                        
                        # Set division frame if this cell has children
                        if has_children and cell_id in self.division_frames:
                            detailed_entry["Division Frame"] = str(end_frame)
                        
                    except Exception as e:
                        print(f"️ Warning: Error calculating lineage metrics for consensus cell {cell_id}: {e}")
                    
                    detailed_lineage_data.append(detailed_entry)
                    
                except Exception as e:
                    print(f"️ Warning: Error processing consensus cell {cell_id} for detailed CSV: {e}")
                    continue
            
            # Sort the detailed lineage data by Cell ID in ascending order
            detailed_lineage_data.sort(key=lambda x: x["Cell ID"])
            
            # Save detailed lineage CSV
            detailed_csv_path = os.path.join(self.labelled_path, "consensus_detailed_lineage.csv")
            
            try:
                with open(detailed_csv_path, 'w', newline='', encoding='utf-8') as f:
                    if detailed_lineage_data:
                        fieldnames = list(detailed_lineage_data[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(detailed_lineage_data)
                
                print(f" Consensus detailed lineage CSV saved successfully!")
                print(f" File location: {detailed_csv_path}")
                print(f" Contains {len(detailed_lineage_data)} consensus cell records")
                print(f" Cell IDs sorted in ascending order")
                print(f" 'Is iPS' column correctly maps to consensus Label column (1='Yes', 0='No')")
                print(f" Based on consensus data with backward tracing")
                print(f" STRICT CRITERIA: ≥2 frames, score≥15, ratio≥0.4")
                
                # Summary statistics
                if detailed_lineage_data:
                    ips_cells = [entry for entry in detailed_lineage_data if entry["Is iPS"] == "Yes"]
                    preprocessed_cells = [entry for entry in detailed_lineage_data if entry["In Preprocessed Detection"] == "Yes"]
                    
                    print(f" Consensus Summary:")
                    print(f"   iPS cells: {len(ips_cells)}")
                    print(f"   Direct preprocessed detection: {len(preprocessed_cells)}")
                
                return detailed_csv_path
                
            except Exception as e:
                print(f" ERROR: Failed to save consensus detailed lineage CSV: {e}")
                return None
                
        except Exception as e:
            print(f" ERROR: Unexpected error in consensus detailed CSV creation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
    
    def run_complete_consensus_tracking(self, detection_start_frame=275, detection_end_frame=400):
        """ Run complete consensus tracking pipeline with backward tracing"""
        print("="*100)
        print(f" CONSENSUS LINEAGE TRACKING - FOV {self.fov}")
        print("="*100)
        print("CONSENSUS DATA MODIFICATIONS:")
        print(f"   Input: consensus_data.csv (instead of arranged.csv)")
        print(f"   Lineage Data: consensus/res_track.txt")
        print(f"   Output: consensus folder")
        print(" BACKWARD TRACING WITH STRICT CRITERIA:")
        print(f"   Uses preprocessed green segmentation as iPS ground truth")
        print(f"   White regions in preprocessed = successfully induced iPS")
        print(f"   Direct segmentation overlap matching")
        print(f"   STRICT: ≥2 frames, score≥15, ratio≥0.4")
        print(f"   Traces ancestors of HIGH-CONFIDENCE matched cells backward")
        print("TEMPORAL VALIDATION METHODOLOGY:")
        print("  Method 1: Progressive Gap Tolerance (frame-based)")
        print("  Method 2: Missing Segmentation Tolerance (missing-based)")
        print("  Method 3: Combined Pattern (pattern-based)")
        print("  Final Tolerance = MIN of all three methods")
        print("="*100)
        
        try:
            # Step 1: Find blue markers from consensus data
            ground_truth_cells = self.find_blue_markers_ground_truth()
            
            # Step 2: Find segmented cells using preprocessed green segmentation
            segmented_cells = self.find_segmented_cells_last_frame_fixed(detection_start_frame, detection_end_frame)
            
            # Step 2.5: Detect failed progenitors
            failed_progenitors = self.detect_failed_progenitors()
            
            if not segmented_cells and not ground_truth_cells:
                print("️ No cells detected in consensus data")
                cell_labels = {cell_id: 'normal' for cell_id in self.consensus_df['Cell ID'].unique()}
                metadata = {
                    'fov': str(self.fov),
                    'method': 'consensus_no_detection',
                    'has_lineage_data': self.has_lineage_data,
                    'total_cells': len(cell_labels),
                    'ips_lineage_count': 0,
                    'ips_percentage': 0.0
                }
                return False, metadata
            
            # Step 3: Backward tracing from preprocessed segmentation
            backward_lineages = self.backward_trace_complete_lineages(segmented_cells)
            
            # Step 4: Forward tracing
            forward_lineages = self.forward_trace_from_ground_truth(ground_truth_cells)
            
            # Step 5: Cross-validate
            final_lineages = self.cross_validate_and_combine()
            
            # Step 6: Create labels
            cell_labels = self.create_comprehensive_labels()
            
            # Step 7: Save results (includes Excel and CSV creation)
            csv_path, metadata_path, metadata = self.update_and_save_results(cell_labels)
            
            print(f"\n CONSENSUS TRACKING COMPLETED!")
            print(f" Summary:")
            print(f"   Ground truth: {len(ground_truth_cells)}")
            print(f"    Preprocessed detection: {len(segmented_cells)}")
            print(f"   Final iPS: {metadata['ips_lineage_count']} ({metadata['ips_percentage']:.1f}%)")
            print(f"   Method: {metadata['method']}")
            print(f"   Validation: Three-method temporal validation applied")
            print(f" Results saved in: {self.labelled_path}")
            print(f" Excel lineage tree: consensus_lineage_tree_retrospective.xlsx")
            print(f" Detailed CSV: consensus_detailed_lineage.csv")
            print(f" CONSENSUS FEATURES: Uses consensus data")
            print(f" Backward tracing uses preprocessed green segmentation")
            print(f" STRICT CRITERIA: ≥2 frames, score≥15, ratio≥0.4")
            
            return True, metadata
            
        except Exception as e:
            print(f" Error in consensus tracking: {e}")
            import traceback
            traceback.print_exc()
            return False, None


def label_single_fov_consensus_tracking(fov, reference_model="KIT-GE", detection_start_frame=275, detection_end_frame=400):
    """Run consensus tracking for single FOV with backward tracing"""
    try:
        labeler = ConsensusIPSLabeler(fov)
        success, metadata = labeler.run_complete_consensus_tracking(detection_start_frame, detection_end_frame)
        return success, metadata
    except Exception as e:
        print(f" Error with FOV {fov}: {e}")
        return False, None


def label_all_fovs_consensus_tracking(reference_model="KIT-GE", detection_start_frame=275, detection_end_frame=400):
    """Run consensus tracking for all FOVs with  backward tracing"""
    consensus_base_dir = "consensus"
    
    if not os.path.exists(consensus_base_dir):
        print(f" Consensus directory not found: {consensus_base_dir}")
        return
    
    # Find consensus FOVs
    fov_dirs = []
    for d in os.listdir(consensus_base_dir):
        consensus_fov_path = os.path.join(consensus_base_dir, d)
        if os.path.isdir(consensus_fov_path):
            # Check if consensus_data.csv exists
            consensus_data_path = os.path.join(consensus_fov_path, "consensus_data.csv")
            if os.path.exists(consensus_data_path):
                fov_dirs.append(d)
    
    fov_dirs = sorted(fov_dirs)
    print(f" Found {len(fov_dirs)} consensus FOVs")
    
    successful = 0
    
    for fov in fov_dirs:
        print(f"\n{'='*80}")
        print(f" Processing Consensus FOV {fov}")
        print(f"{'='*80}")
        
        success, metadata = label_single_fov_consensus_tracking(fov, reference_model, detection_start_frame, detection_end_frame)
        
        if success:
            successful += 1
            if metadata:
                ips_count = metadata.get('ips_lineage_count', 0)
                ips_pct = metadata.get('ips_percentage', 0)
                print(f" Consensus FOV {fov}: {ips_count} iPS cells ({ips_pct:.1f}%)")
        else:
            print(f" Consensus FOV {fov} failed")
    
    print(f"\n{'='*100}")
    print(" ALL CONSENSUS FOVs SUMMARY")
    print(f"{'='*100}")
    print(f" Successfully processed: {successful}/{len(fov_dirs)} FOVs")
    print("Validation Methodology: Three-method temporal validation on consensus data")
    print(" Excel lineage trees and detailed CSV files created for each FOV")
    print(" CONSENSUS FEATURES: consensus_data.csv input, consensus lineage data")
    print(" Backward tracing uses preprocessed green segmentation as ground truth")
    print(" White regions in preprocessed images = successfully induced iPS cells")
    print(" STRICT CRITERIA: ≥2 frames, score≥15, ratio≥0.4 to reduce false positives")


if __name__ == "__main__":
    print(" Testing consensus tracking with backward tracing and STRICT criteria!")
    print(" Now includes consensus_lineage_tree_retrospective.xlsx and consensus_detailed_lineage.csv!")
    print(" CONSENSUS MODIFICATIONS: Uses consensus data")
    print("  Backward tracing uses preprocessed green segmentation as iPS ground truth")
    print(" STRICT: ≥2 frames, score≥15, ratio≥0.4 to reduce false positives")
    label_single_fov_consensus_tracking("2")
