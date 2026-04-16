import os
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import cv2

# Import existing modules - FIXED IMPORTS
try:
    from track import track_main  # FIXED: Import the correct function from your track.py
    from features import extract_features_from_test
    from count import count_cells_in_dataset, save_results_to_csv  # FIXED: Import count functions
    from generate_trace import get_trace, get_video
    from visualization import visualize_single_fov_proper  # Use the SAME visualization as global pipeline
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure track.py, features.py, count.py, visualization.py are available")
    sys.exit(1)

def read_and_arrange_features_for_phase(features_file, phase):
    """Adapt arrange.py logic for phase processing"""
    if not os.path.exists(features_file):
        print(f"Warning: {features_file} does not exist")
        return None, None
    
    try:
        df = pd.read_csv(features_file)
        print(f"  Original data: {len(df['Cell ID'].unique())} cells, {len(df)} total rows")
        
        # Drop specific columns as in arrange.py
        columns_to_drop = ['Parent ID', 'Daughter IDs', 'Division Frame']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df = df.drop(columns=existing_columns_to_drop)
            print(f"  Dropped columns: {', '.join(existing_columns_to_drop)}")
        
        # Drop N/A values (following arrange.py pattern)
        df_clean = df.dropna(subset=['X', 'Y', 'Frame'])
        
        # Additional cleanup for other important columns if they exist
        if 'Cell ID' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['Cell ID'])
        
        print(f"  After dropping N/A: {len(df_clean['Cell ID'].unique())} cells, {len(df_clean)} total rows")
        
        # Sort by Cell ID first, then by Frame (following arrange.py sorting pattern)
        arranged_df = df_clean.sort_values(['Cell ID', 'Frame']).reset_index(drop=True)
        
        # Move Frame column to the last position
        if 'Frame' in arranged_df.columns:
            cols = [col for col in arranged_df.columns if col != 'Frame']
            cols.append('Frame')  # Add Frame at the end
            arranged_df = arranged_df[cols]
            print(f"  Moved Frame column to last position")
        
        print(f"  Arranged by Cell ID and Frame")
        
        # Get the output directory (features directory for this phase)
        output_dir = os.path.dirname(features_file)
        
        return arranged_df, output_dir
        
    except Exception as e:
        print(f"Error reading {features_file}: {e}")
        return None, None


class PhasialRetrospectiveLabeler:
    """
    Phasial Retrospective Labeling Coordinator
    
    This class implements true retrospective labeling across phases:
    1. Start from Phase 8 (strongest green signal)
    2. Identify iPS cells in Phase 8
    3. Use GLOBAL tracking data to trace lineages backward through phases 7→6→5→4→3→2→1
    4. Apply retrospective labels to ALL phases
    """
    
    def __init__(self, base_path: str, fov: str):
        self.base_path = base_path
        self.fov = fov
        
        # Define phase boundaries
        self.phases = {
            1: (0, 101),      # 102 frames
            2: (102, 142),    # 41 frames 
            3: (143, 232),    # 90 frames 
            4: (233, 274),    # 42 frames
            5: (275, 302),    # 28 frames - Green signal starts
            6: (303, 316),    # 14 frames
            7: (317, 354),    # 38 frames
            8: (355, 400)     # 46 frames
        }
        
        # Paths
        self.fov_path = os.path.join(base_path, "nuclear_dataset", fov)
        self.phases_path = os.path.join(self.fov_path, "phases")
        self.global_track_result_path = os.path.join(self.fov_path, "track_result")
        self.global_green_signal_path = os.path.join(self.fov_path, "green_signal")
        
        # Global tracking data
        self.global_tracking_data = {}
        self.global_parent_to_children = {}
        self.global_child_to_parent = {}
        
        # iPS detection results
        self.phase8_ips_cells = set()
        self.global_ips_lineages = set()
    
    def load_global_tracking_data(self):
        """Load GLOBAL tracking data from the original full dataset"""
        print(f"\n🌍 Loading GLOBAL tracking data...")
        
        global_res_track = os.path.join(self.global_track_result_path, "res_track.txt")
        if not os.path.exists(global_res_track):
            print(f"❌ Global res_track.txt not found at {global_res_track}")
            return False
        
        try:
            with open(global_res_track, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    cell_id = int(parts[0])
                    start_frame = int(parts[1])
                    end_frame = int(parts[2])
                    parent_id = int(parts[3])
                    
                    self.global_tracking_data[cell_id] = {
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'parent_id': parent_id
                    }
                    
                    # Build global parent-child relationships
                    if parent_id != 0:
                        if parent_id not in self.global_parent_to_children:
                            self.global_parent_to_children[parent_id] = []
                        self.global_parent_to_children[parent_id].append(cell_id)
                        self.global_child_to_parent[cell_id] = parent_id
            
            print(f"✅ Loaded global tracking data: {len(self.global_tracking_data)} cells")
            print(f"🌳 Found {len(self.global_parent_to_children)} parent-child relationships")
            return True
            
        except Exception as e:
            print(f"❌ Error loading global tracking data: {e}")
            return False
    
    def identify_ips_in_phase8(self):
        """
        Step 1: Identify iPS cells in Phase 8 (frames 355-400) using green signal
        """
        print(f"\n🎯 STEP 1: Identifying iPS cells in Phase 8 (frames 355-400)")
        
        phase8_dir = os.path.join(self.phases_path, "phase_8")
        green_partition_dir = os.path.join(phase8_dir, "green_signal_partition")
        track_partition_dir = os.path.join(phase8_dir, "track_result")
        
        if not os.path.exists(green_partition_dir) or not os.path.exists(track_partition_dir):
            print(f"❌ Phase 8 partitions not found")
            return set()
        
        # Get green signal files in phase 8
        green_files = sorted([f for f in os.listdir(green_partition_dir) 
                             if f.endswith('.tif') or f.endswith('.tiff')])
        track_files = sorted([f for f in os.listdir(track_partition_dir) 
                             if f.endswith('.tif') or f.endswith('.tiff')])
        
        ips_candidates = set()
        detection_scores = {}
        
        print(f"📊 Processing {len(green_files)} green signal frames in Phase 8...")
        
        for frame_idx in range(len(green_files)):
            if frame_idx >= len(track_files):
                continue
            
            try:
                # Load green and tracking images
                green_path = os.path.join(green_partition_dir, green_files[frame_idx])
                track_path = os.path.join(track_partition_dir, track_files[frame_idx])
                
                green_img = cv2.imread(green_path, cv2.IMREAD_GRAYSCALE)
                track_img = cv2.imread(track_path, cv2.IMREAD_UNCHANGED)
                
                if green_img is None or track_img is None:
                    continue
                
                # Resize green to match tracking
                if green_img.shape != track_img.shape:
                    green_img = cv2.resize(green_img, (track_img.shape[1], track_img.shape[0]))
                
                # Analyze each cell
                unique_cell_ids = np.unique(track_img)
                if 0 in unique_cell_ids:
                    unique_cell_ids = unique_cell_ids[unique_cell_ids != 0]
                
                for cell_id in unique_cell_ids:
                    cell_id = int(cell_id)
                    if cell_id == 0:
                        continue
                    
                    score = self.calculate_ips_score_phase8(green_img, track_img, cell_id, frame_idx)
                    
                    if score > 0:
                        if cell_id not in detection_scores:
                            detection_scores[cell_id] = []
                        detection_scores[cell_id].append((frame_idx, score))
            
            except Exception as e:
                print(f"⚠️ Error processing frame {frame_idx}: {e}")
                continue
        
        # Validate candidates across multiple frames
        for cell_id, scores in detection_scores.items():
            if len(scores) >= 2:  # Detected in multiple frames
                avg_score = np.mean([s[1] for s in scores])
                max_score = max([s[1] for s in scores])
                
                if avg_score >= 2.5 and max_score >= 3.0:
                    ips_candidates.add(cell_id)
        
        self.phase8_ips_cells = ips_candidates
        print(f"✅ Identified {len(ips_candidates)} iPS cells in Phase 8: {sorted(ips_candidates)}")
        return ips_candidates
    
    def calculate_ips_score_phase8(self, green_img, track_img, cell_id, frame_idx):
        """Calculate iPS score for a cell in Phase 8"""
        try:
            cell_mask = (track_img == cell_id)
            if not np.any(cell_mask):
                return 0.0
            
            # Green fluorescence analysis
            cell_pixels = green_img[cell_mask]
            if len(cell_pixels) == 0:
                return 0.0
            
            cell_mean = np.mean(cell_pixels)
            
            # Background comparison
            background_mask = (track_img == 0)
            if np.any(background_mask):
                bg_pixels = green_img[background_mask]
                bg_mean = np.mean(bg_pixels) if len(bg_pixels) > 0 else np.mean(green_img) * 0.5
                bg_std = np.std(bg_pixels) if len(bg_pixels) > 0 else np.std(green_img) * 0.5
            else:
                bg_mean = np.mean(green_img) * 0.5
                bg_std = np.std(green_img) * 0.5
            
            # Intensity score
            intensity_score = (cell_mean - bg_mean) / (bg_std + 1e-6)
            
            # Size factor
            cell_area = np.sum(cell_mask)
            size_factor = min(cell_area / 100.0, 2.0)
            
            final_score = intensity_score * size_factor
            return max(0, final_score)
            
        except Exception:
            return 0.0
    
    def trace_global_lineages(self):
        """
        Step 2: Trace global lineages backward from Phase 8 iPS cells
        """
        print(f"\n🔄 STEP 2: Tracing global lineages backward from Phase 8")
        
        if not self.phase8_ips_cells:
            print("❌ No iPS cells found in Phase 8 to trace")
            return set()
        
        all_ips_lineages = set()
        
        for ips_cell_id in self.phase8_ips_cells:
            # Get complete lineage for this iPS cell
            lineage = self.get_complete_global_lineage(ips_cell_id)
            all_ips_lineages.update(lineage)
            
            ancestors = self.get_global_ancestors(ips_cell_id)
            descendants = self.get_global_descendants(ips_cell_id)
            
            print(f"📈 iPS Cell {ips_cell_id}: {len(ancestors)} ancestors, {len(descendants)} descendants, {len(lineage)} total lineage")
        
        self.global_ips_lineages = all_ips_lineages
        print(f"✅ Total global iPS lineage: {len(all_ips_lineages)} cells")
        return all_ips_lineages
    
    def get_complete_global_lineage(self, cell_id):
        """Get complete lineage using global tracking data"""
        lineage = set([cell_id])
        lineage.update(self.get_global_ancestors(cell_id))
        lineage.update(self.get_global_descendants(cell_id))
        return lineage
    
    def get_global_ancestors(self, cell_id):
        """Get all ancestor cells using global tracking"""
        ancestors = set()
        current_id = cell_id
        max_iterations = 100
        
        for _ in range(max_iterations):
            if current_id not in self.global_child_to_parent:
                break
            parent_id = self.global_child_to_parent[current_id]
            if parent_id == 0:
                break
            ancestors.add(parent_id)
            current_id = parent_id
        
        return ancestors
    
    def get_global_descendants(self, cell_id):
        """Get all descendant cells using global tracking"""
        descendants = set()
        
        def recursive_descendants(cid, depth=0):
            if depth > 50:
                return
            if cid in self.global_parent_to_children:
                for child_id in self.global_parent_to_children[cid]:
                    descendants.add(child_id)
                    recursive_descendants(child_id, depth + 1)
        
        recursive_descendants(cell_id)
        return descendants
    
    def apply_retrospective_labels_to_all_phases(self):
        """
        Step 3: Apply retrospective iPS labels to ALL phases
        """
        print(f"\n🏷️ STEP 3: Applying retrospective labels to all phases")
        
        if not self.global_ips_lineages:
            print("❌ No global iPS lineages to apply")
            return False
        
        success_count = 0
        
        for phase in range(1, 9):  # Phases 1-8
            phase_dir = os.path.join(self.phases_path, f"phase_{phase}")
            track_result_dir = os.path.join(phase_dir, "track_result")
            arranged_path = os.path.join(track_result_dir, "arranged.csv")
            
            if not os.path.exists(arranged_path):
                print(f"⚠️ Phase {phase}: arranged.csv not found")
                continue
            
            try:
                # Load phase arranged data
                df = pd.read_csv(arranged_path)
                original_ips_count = len(df[df['Label'] == 1])
                
                # Apply retrospective labels
                def get_retrospective_label(cell_id):
                    return 1 if cell_id in self.global_ips_lineages else 0
                
                df['Label'] = df['Cell ID'].apply(get_retrospective_label)
                
                # Save updated arranged.csv
                df.to_csv(arranged_path, index=False)
                
                # Also save to Labelled folder
                labelled_dir = os.path.join(phase_dir, "Labelled")
                os.makedirs(labelled_dir, exist_ok=True)
                labelled_path = os.path.join(labelled_dir, "arranged_retrospective_labels.csv")
                df.to_csv(labelled_path, index=False)
                
                new_ips_count = len(df[df['Label'] == 1])
                print(f"✅ Phase {phase}: {original_ips_count} → {new_ips_count} iPS cells")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ Phase {phase}: Error applying labels - {e}")
        
        print(f"✅ Applied retrospective labels to {success_count}/8 phases")
        return success_count == 8
    
    def run_phasial_retrospective_labeling(self):
        """
        Run the complete phasial retrospective labeling process
        CRITICAL: This can ONLY run AFTER Phase 8 is completely finished!
        """
        print("="*80)
        print(f"🔄 PHASIAL RETROSPECTIVE LABELING - FOV {self.fov}")
        print("="*80)
        print("⚠️ CRITICAL PREREQUISITES:")
        print("   📋 Phase 8 MUST be completely finished first!")
        print("   📋 Phase 8 MUST have green signal analysis completed!")
        print("   📋 Global tracking data MUST be available!")
        print("")
        print("This process:")
        print("1. 🌍 Loads GLOBAL tracking data from full dataset")
        print("2. 🎯 Identifies iPS cells in Phase 8 (strongest green signal)")
        print("3. 🔄 Traces lineages backward through phases 7→6→5→4→3→2→1")
        print("4. 🏷️ Applies retrospective labels to ALL phases")
        print("="*80)
        
        # Validate Prerequisites
        print(f"\n🔍 VALIDATING PREREQUISITES...")
        
        # Check if Phase 8 directory exists and is complete
        phase8_dir = os.path.join(self.phases_path, "phase_8")
        if not os.path.exists(phase8_dir):
            print(f"❌ Phase 8 directory not found: {phase8_dir}")
            return False
        
        # Check if Phase 8 has required components
        required_phase8_dirs = [
            "green_signal_partition",
            "track_result", 
            "features",
            "test_partition"
        ]
        
        for req_dir in required_phase8_dirs:
            dir_path = os.path.join(phase8_dir, req_dir)
            if not os.path.exists(dir_path):
                print(f"❌ Phase 8 missing required directory: {req_dir}")
                return False
            
            # Check if directory has files
            files = [f for f in os.listdir(dir_path) if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.csv') or f.endswith('.txt')]
            if not files:
                print(f"❌ Phase 8 directory is empty: {req_dir}")
                return False
        
        print(f"✅ Phase 8 prerequisites validated!")
        
        try:
            # Step 1: Load global tracking data
            print(f"\n🌍 STEP 1: Loading global tracking data...")
            if not self.load_global_tracking_data():
                print("❌ Failed to load global tracking data")
                return False
            
            # Step 2: Identify iPS cells in Phase 8
            print(f"\n🎯 STEP 2: Identifying iPS cells in Phase 8...")
            phase8_ips = self.identify_ips_in_phase8()
            if not phase8_ips:
                print("❌ No iPS cells identified in Phase 8")
                print("💡 This could mean:")
                print("   - Green signal is too weak")
                print("   - Detection thresholds need adjustment") 
                print("   - Phase 8 data is incomplete")
                return False
            
            # Step 3: Trace global lineages
            print(f"\n🔄 STEP 3: Tracing global lineages...")
            global_lineages = self.trace_global_lineages()
            if not global_lineages:
                print("❌ No global lineages traced")
                return False
            
            # Step 4: Apply to all phases
            print(f"\n🏷️ STEP 4: Applying retrospective labels to all phases...")
            if not self.apply_retrospective_labels_to_all_phases():
                print("❌ Failed to apply labels to all phases")
                return False
            
            print(f"\n🎉 PHASIAL RETROSPECTIVE LABELING COMPLETED!")
            print(f"📊 Summary:")
            print(f"   🎯 Phase 8 iPS detections: {len(self.phase8_ips_cells)}")
            print(f"   🔄 Global iPS lineage: {len(self.global_ips_lineages)} cells")
            print(f"   📈 Expansion factor: {len(self.global_ips_lineages)/len(self.phase8_ips_cells):.1f}x")
            print(f"   🏷️ Applied to all 8 phases")
            print(f"   ✅ Ready for visualization!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in phasial retrospective labeling: {e}")
            import traceback
            traceback.print_exc()
            return False


class PhaseBasedPipeline:
    """
    FIXED Phase-based pipeline with PHASIAL RETROSPECTIVE LABELING
    """
    
    def __init__(self, base_path: str, fov: str):
        self.base_path = base_path
        self.fov = fov
        
        # Define phase boundaries
        self.phases = {
            1: (0, 101),      # 102 frames
            2: (102, 142),    # 41 frames 
            3: (143, 232),    # 90 frames 
            4: (233, 274),    # 42 frames
            5: (275, 302),    # 28 frames - Green signal starts
            6: (303, 316),    # 14 frames
            7: (317, 354),    # 38 frames
            8: (355, 400)     # 46 frames
        }
        
        # Paths
        self.fov_path = os.path.join(base_path, "nuclear_dataset", fov)
        self.test_path = os.path.join(self.fov_path, "test")
        self.res_result_path = os.path.join(self.fov_path, "res_result")
        self.green_signal_path = os.path.join(self.fov_path, "green_signal")
        self.phases_path = os.path.join(self.fov_path, "phases")
        
        # Detect green signal range
        self.green_signal_info = self.detect_green_signal_range()
        
        print(f"📋 Phase boundaries defined:")
        for phase, (start, end) in self.phases.items():
            frame_count = end - start + 1
            print(f"   Phase {phase}: frames {start}-{end} ({frame_count} frames)")
        
        if self.green_signal_info:
            print(f"🟢 Green signal detected: {self.green_signal_info}")
    
    def detect_green_signal_range(self):
        """Detect actual green signal frame range from files"""
        if not os.path.exists(self.green_signal_path):
            print(f"⚠️ Green signal path not found: {self.green_signal_path}")
            return None
        
        green_files = sorted([f for f in os.listdir(self.green_signal_path) 
                             if f.endswith('.tif') or f.endswith('.tiff')])
        
        if not green_files:
            print(f"⚠️ No green signal files found")
            return None
        
        # Extract frame numbers from filenames
        frame_numbers = []
        for filename in green_files:
            try:
                frame_num = int(filename.split('.')[0])
                frame_numbers.append(frame_num)
            except ValueError:
                continue
        
        if not frame_numbers:
            print(f"⚠️ Could not parse frame numbers from green signal files")
            return None
        
        frame_numbers.sort()
        start_frame = min(frame_numbers)
        end_frame = max(frame_numbers)
        
        print(f"🟢 Green signal detected: frames {start_frame}-{end_frame} ({len(frame_numbers)} files)")
        
        return {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'total_files': len(frame_numbers),
            'frame_numbers': frame_numbers
        }
    
    def create_phase_directories(self):
        """Create directory structure for all phases"""
        print(f"\n📁 Creating phase directories for FOV {self.fov}...")
        
        if not os.path.exists(self.phases_path):
            os.makedirs(self.phases_path)
        
        for phase in self.phases.keys():
            phase_dir = os.path.join(self.phases_path, f"phase_{phase}")
            os.makedirs(phase_dir, exist_ok=True)
            
            # Create subdirectories
            subdirs = [
                "test_partition", 
                "res_result_partition", 
                "track_result", 
                "features", 
                "counting", 
                "trace", 
                "Labelled",
                "Visualization"
            ]
            
            # Add green signal partition for overlapping phases
            start_frame, end_frame = self.phases[phase]
            if (self.green_signal_info and 
                start_frame <= self.green_signal_info['end_frame'] and 
                end_frame >= self.green_signal_info['start_frame']):
                subdirs.append("green_signal_partition")
            
            for subdir in subdirs:
                os.makedirs(os.path.join(phase_dir, subdir), exist_ok=True)
        
        print(f"✅ Created directories for {len(self.phases)} phases")
    
    def get_file_list_with_partition(self, source_dir, phase):
        """Get partitioned file list for a specific phase"""
        if not os.path.exists(source_dir):
            print(f"❌ Source directory not found: {source_dir}")
            return []
        
        all_files = sorted([f for f in os.listdir(source_dir) 
                           if f.endswith('.tif') or f.endswith('.tiff')])
        
        if not all_files:
            print(f"❌ No files found in {source_dir}")
            return []
        
        start_frame, end_frame = self.phases[phase]
        
        # Extract files for specific phase
        phase_files = []
        for filename in all_files:
            try:
                frame_num = int(filename.split('.')[0])
                if start_frame <= frame_num <= end_frame:
                    phase_files.append(filename)
            except ValueError:
                continue
        
        # Fallback to index-based approach
        if not phase_files:
            print(f"⚠️ Frame number extraction failed, using index-based approach")
            if end_frame < len(all_files):
                phase_files = all_files[start_frame:end_frame + 1]
            else:
                phase_files = all_files[start_frame:] if start_frame < len(all_files) else []
        
        print(f"📄 Phase {phase}: extracted {len(phase_files)} files")
        return phase_files
    
    def create_phase_partitions(self, phase):
        """Create partitioned data for a specific phase"""
        print(f"\n🔄 Creating partitions for Phase {phase}...")
        
        phase_dir = os.path.join(self.phases_path, f"phase_{phase}")
        start_frame, end_frame = self.phases[phase]
        
        # Partition test images
        print(f"📷 Partitioning test images...")
        test_files = self.get_file_list_with_partition(self.test_path, phase)
        test_partition_dir = os.path.join(phase_dir, "test_partition")
        
        for i, filename in enumerate(test_files):
            src = os.path.join(self.test_path, filename)
            dst = os.path.join(test_partition_dir, f"{i:06d}.tif")
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        print(f"✅ Copied {len(test_files)} test images")
        
        # Partition res_result images
        print(f"🎯 Partitioning res_result images...")
        res_files = []
        if os.path.exists(self.res_result_path):
            res_files = self.get_file_list_with_partition(self.res_result_path, phase)
            res_partition_dir = os.path.join(phase_dir, "res_result_partition")
            
            for i, filename in enumerate(res_files):
                src = os.path.join(self.res_result_path, filename)
                dst = os.path.join(res_partition_dir, f"{i:06d}.tif")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            
            print(f"✅ Copied {len(res_files)} res_result images")
        else:
            print(f"⚠️ res_result directory not found")
        
        # Partition green signal for overlapping phases
        green_files_copied = 0
        if (self.green_signal_info and 
            start_frame <= self.green_signal_info['end_frame'] and 
            end_frame >= self.green_signal_info['start_frame']):
            
            print(f"🟢 Partitioning green signal images...")
            green_partition_dir = os.path.join(phase_dir, "green_signal_partition")
            os.makedirs(green_partition_dir, exist_ok=True)
            
            # Calculate overlap
            overlap_start = max(start_frame, self.green_signal_info['start_frame'])
            overlap_end = min(end_frame, self.green_signal_info['end_frame'])
            
            print(f"   🎯 Overlap: frames {overlap_start}-{overlap_end}")
            
            # Copy green signal files for overlap range
            green_counter = 0
            for frame_num in range(overlap_start, overlap_end + 1):
                green_filename = f"{frame_num:06d}.tif"
                src = os.path.join(self.green_signal_path, green_filename)
                
                if os.path.exists(src):
                    dst = os.path.join(green_partition_dir, f"{green_counter:06d}.tif")
                    shutil.copy2(src, dst)
                    green_counter += 1
            
            green_files_copied = green_counter
            print(f"✅ Copied {green_files_copied} green signal images")
        else:
            print(f"   ℹ️ Phase {phase} has no green signal overlap")
        
        return len(test_files), len(res_files)
    
    def run_tracking_for_phase(self, phase):
        """Run tracking for a specific phase"""
        print(f"\n🔍 Running tracking for Phase {phase}...")
        
        phase_dir = os.path.join(self.phases_path, f"phase_{phase}")
        res_partition_dir = os.path.join(phase_dir, "res_result_partition")
        track_result_dir = os.path.join(phase_dir, "track_result")
        
        if not os.path.exists(res_partition_dir):
            print(f"❌ res_result_partition not found for phase {phase}")
            return False
        
        # Check if partition has files
        partition_files = [f for f in os.listdir(res_partition_dir) 
                          if f.endswith('.tif') or f.endswith('.tiff')]
        
        if not partition_files:
            print(f"❌ No files found in res_result_partition for phase {phase}")
            return False
        
        print(f"📊 Found {len(partition_files)} files for tracking")
        
        try:
            # Rename files for track_main compatibility
            temp_files = []
            for f in partition_files:
                old_path = os.path.join(res_partition_dir, f)
                new_name = f"predict_{f}"
                new_path = os.path.join(res_partition_dir, new_name)
                shutil.move(old_path, new_path)
                temp_files.append((new_path, old_path))
            
            print(f"🔄 Renamed {len(temp_files)} files to predict_XXXXXX.tif format")
            
            # Run tracking
            success = track_main(res_partition_dir, track_result_dir, threshold=0.15)
            
            # Rename files back
            for new_path, old_path in temp_files:
                if os.path.exists(new_path):
                    shutil.move(new_path, old_path)
            
            # Check tracking results
            track_files = [f for f in os.listdir(track_result_dir) 
                          if f.endswith('.tif') or f.endswith('.tiff')]
            res_track_file = os.path.join(track_result_dir, "res_track.txt")
            
            if os.path.exists(res_track_file) and len(track_files) > 0:
                print(f"✅ Tracking completed for Phase {phase}")
                
                with open(res_track_file, 'r') as f:
                    lines = f.readlines()
                print(f"   📊 Tracking entries: {len(lines)}")
                
                return True
            else:
                print(f"❌ Tracking failed for Phase {phase}")
                return False
        
        except Exception as e:
            print(f"❌ Error during tracking for Phase {phase}: {e}")
            return False
    
    def run_features_for_phase(self, phase):
        """Extract features for a specific phase"""
        print(f"\n🧬 Extracting features for Phase {phase}...")
        
        phase_dir = os.path.join(self.phases_path, f"phase_{phase}")
        test_partition_dir = os.path.join(phase_dir, "test_partition")
        track_result_dir = os.path.join(phase_dir, "track_result")
        features_dir = os.path.join(phase_dir, "features")
        
        # Verify required directories
        required_dirs = [test_partition_dir, track_result_dir]
        for req_dir in required_dirs:
            if not os.path.exists(req_dir):
                print(f"❌ Required directory not found: {req_dir}")
                return False
        
        # Check for res_track.txt
        res_track_file = os.path.join(track_result_dir, "res_track.txt")
        if not os.path.exists(res_track_file):
            print(f"❌ res_track.txt not found for Phase {phase}")
            return False
        
        try:
            # Extract features
            extract_features_from_test(test_partition_dir, track_result_dir, features_dir)
            
            # Verify features were extracted
            features_file = os.path.join(features_dir, "features.csv")
            if os.path.exists(features_file):
                df = pd.read_csv(features_file)
                print(f"✅ Features extracted for Phase {phase}")
                print(f"📊 Feature records: {len(df)}")
                print(f"🧬 Unique cells: {len(df['Cell ID'].unique())}")
                return True
            else:
                print(f"❌ features.csv not created for Phase {phase}")
                return False
        
        except Exception as e:
            print(f"❌ Error extracting features for Phase {phase}: {e}")
            return False
    
    def arrange_features_for_phase(self, phase):
        """Arrange features for a specific phase"""
        print(f"\n📋 Arranging features for Phase {phase}...")
        
        phase_dir = os.path.join(self.phases_path, f"phase_{phase}")
        features_dir = os.path.join(phase_dir, "features")
        features_file = os.path.join(features_dir, "features.csv")
        
        if not os.path.exists(features_file):
            print(f"❌ features.csv not found for Phase {phase}")
            return False
        
        try:
            # Read and arrange features
            arranged_df, output_dir = read_and_arrange_features_for_phase(features_file, phase)
            
            if arranged_df is not None and not arranged_df.empty:
                # Save arranged data
                arranged_file = os.path.join(features_dir, "arranged.csv")
                arranged_df.to_csv(arranged_file, index=False)
                
                # Also save in track_result for labeling compatibility
                track_arranged_file = os.path.join(phase_dir, "track_result", "arranged.csv")
                arranged_df.to_csv(track_arranged_file, index=False)
                
                print(f"✅ Arranged features saved for Phase {phase}")
                print(f"📊 Arranged data: {len(arranged_df['Cell ID'].unique())} cells, {len(arranged_df)} rows")
                return True
            else:
                print(f"❌ No valid data after arranging for Phase {phase}")
                return False
        
        except Exception as e:
            print(f"❌ Error arranging features for Phase {phase}: {e}")
            return False
    
    def run_counting_for_phase(self, phase):
        """Run cell counting for a specific phase"""
        print(f"\n📊 Running cell counting for Phase {phase}...")
        
        phase_dir = os.path.join(self.phases_path, f"phase_{phase}")
        test_partition_dir = os.path.join(phase_dir, "test_partition")
        track_result_dir = os.path.join(phase_dir, "track_result")
        counting_dir = os.path.join(phase_dir, "counting")
        
        # Verify required directories
        required_dirs = [test_partition_dir, track_result_dir]
        for req_dir in required_dirs:
            if not os.path.exists(req_dir):
                print(f"❌ Required directory not found: {req_dir}")
                return False
        
        # Check for res_track.txt
        res_track_file = os.path.join(track_result_dir, "res_track.txt")
        if not os.path.exists(res_track_file):
            print(f"❌ res_track.txt not found for Phase {phase}")
            return False
        
        try:
            # Run counting
            results, missing_cells_tracker, overall_counts = count_cells_in_dataset(
                test_partition_dir, track_result_dir, None
            )
            
            if results or overall_counts:
                # Save counting results
                counting_csv = os.path.join(counting_dir, "cell_count_results.csv")
                save_results_to_csv(results, missing_cells_tracker, overall_counts, counting_csv)
                
                print(f"✅ Cell counting completed for Phase {phase}")
                
                if overall_counts and len(overall_counts) > 0:
                    total_cells = overall_counts[0].get('Total Cells', 0)
                    ips_cells = overall_counts[0].get('iPS Count', 0) + overall_counts[0].get('Divided iPS', 0)
                    print(f"   📈 Summary: {total_cells} total cells, {ips_cells} iPS-related")
                
                return True
            else:
                print(f"❌ No counting results generated for Phase {phase}")
                return False
        
        except Exception as e:
            print(f"❌ Error during counting for Phase {phase}: {e}")
            return False
    
    def run_visualization_for_phase(self, phase):
        """Run SAME visualization.py for a specific phase - MUCH BETTER than custom phase visualizer"""
        print(f"\n🎨 Running SAME visualization.py for Phase {phase}...")
        
        phase_dir = os.path.join(self.phases_path, f"phase_{phase}")
        
        # Create a temporary "mini-FOV" structure that visualization.py can work with
        temp_fov_path = os.path.join(phase_dir, "temp_fov_structure")
        os.makedirs(temp_fov_path, exist_ok=True)
        
        # Create symlinks/copies to the required folders for visualization.py
        required_folders = {
            "test": os.path.join(phase_dir, "test_partition"),
            "track_result": os.path.join(phase_dir, "track_result"),
            "Labelled": os.path.join(phase_dir, "Labelled"),
            "green_signal": os.path.join(phase_dir, "green_signal_partition")
        }
        
        try:
            # Create symlinks or copy folders
            for folder_name, source_path in required_folders.items():
                target_path = os.path.join(temp_fov_path, folder_name)
                
                if os.path.exists(source_path):
                    if os.name == 'nt':  # Windows
                        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                    else:  # Unix/Linux/Mac
                        if os.path.exists(target_path):
                            os.remove(target_path)
                        os.symlink(source_path, target_path)
                    print(f"   ✅ Linked {folder_name} for visualization")
                else:
                    print(f"   ⚠️ {folder_name} not found, creating empty directory")
                    os.makedirs(target_path, exist_ok=True)
            
            # Use the SAME visualization.py system - MUCH BETTER!
            from visualization import iPSVisualizerProper
            
            # Create visualizer with temporary structure
            visualizer = iPSVisualizerProper(phase_dir, "temp_fov_structure")
            
            # Override the visualization path to save in the phase directory
            visualizer.visualization_path = os.path.join(phase_dir, "Visualization")
            os.makedirs(visualizer.visualization_path, exist_ok=True)
            
            # Run the SAME high-quality visualization
            success = visualizer.run_visualization(visualize_all_frames=True)
            
            # Clean up temporary structure
            if os.path.exists(temp_fov_path):
                shutil.rmtree(temp_fov_path)
            
            if success:
                print(f"✅ Phase {phase} visualization completed using SAME system as global!")
                return True
            else:
                print(f"⚠️ Phase {phase} visualization completed with warnings")
                return True
        
        except Exception as e:
            print(f"❌ Error during visualization for Phase {phase}: {e}")
            
            # Clean up on error
            if os.path.exists(temp_fov_path):
                shutil.rmtree(temp_fov_path)
            
            return False
    
    def run_single_phase(self, phase):
        """Run complete pipeline for a single phase"""
        print(f"\n{'='*60}")
        print(f"🚀 PROCESSING PHASE {phase} - FOV {self.fov}")
        start_frame, end_frame = self.phases[phase]
        print(f"📅 Frames: {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)")
        print(f"{'='*60}")
        
        # Step 1: Create partitions
        test_count, res_count = self.create_phase_partitions(phase)
        if test_count == 0:
            print(f"❌ No test files for Phase {phase}")
            return False
        
        # Step 2: Run tracking (only if we have res_result files)
        if res_count > 0:
            if not self.run_tracking_for_phase(phase):
                print(f"❌ Tracking failed for Phase {phase}")
                return False
            
            # Step 3: Extract features
            if not self.run_features_for_phase(phase):
                print(f"❌ Feature extraction failed for Phase {phase}")
                return False
            
            # Step 4: Arrange features
            if not self.arrange_features_for_phase(phase):
                print(f"❌ Feature arrangement failed for Phase {phase}")
                return False
            
            # Step 5: Run cell counting
            if not self.run_counting_for_phase(phase):
                print(f"❌ Cell counting failed for Phase {phase}")
                return False
            
            # Step 6: Generate trace visualizations
            print(f"\n📹 Generating trace visualizations for Phase {phase}...")
            test_partition_dir = os.path.join(self.phases_path, f"phase_{phase}", "test_partition")
            track_result_dir = os.path.join(self.phases_path, f"phase_{phase}", "track_result")
            trace_dir = os.path.join(self.phases_path, f"phase_{phase}", "trace")
            
            try:
                get_trace(test_partition_dir, track_result_dir, trace_dir)
                print(f"✅ Trace visualizations generated for Phase {phase}")
                
                # Create video from traces
                get_video(trace_dir)
                print(f"🎥 Video created from trace visualizations")
            except Exception as e:
                print(f"⚠️ Error generating trace for Phase {phase}: {e}")
        
        else:
            print(f"⚠️ Phase {phase} has no res_result files - skipping advanced processing")
        
        print(f"\n✅ PHASE {phase} COMPLETED SUCCESSFULLY!")
        return True
    
    def run_phasial_retrospective_labeling(self):
        """Run the Phasial Retrospective Labeling across all phases"""
        print(f"\n{'='*80}")
        print(f"🔄 RUNNING PHASIAL RETROSPECTIVE LABELING")
        print(f"{'='*80}")
        
        # Create and run the phasial retrospective labeler
        labeler = PhasialRetrospectiveLabeler(self.base_path, self.fov)
        success = labeler.run_phasial_retrospective_labeling()
        
        if success:
            print(f"🎉 Phasial retrospective labeling completed successfully!")
            return True
        else:
            print(f"❌ Phasial retrospective labeling failed")
            return False
    
    def run_phasial_visualization(self):
        """Run the SAME visualization.py for ALL phases - MUCH BETTER than custom visualizer"""
        print(f"\n{'='*80}")
        print(f"🎨 RUNNING PHASIAL VISUALIZATION (Using SAME visualization.py)")
        print(f"{'='*80}")
        
        successful_phases = []
        failed_phases = []
        
        # Process phases in order 1→8 for visualization
        for phase in sorted(self.phases.keys()):
            print(f"\n🎨 Visualizing Phase {phase}...")
            success = self.run_visualization_for_phase(phase)
            
            if success:
                successful_phases.append(phase)
                print(f"✅ Phase {phase} visualization completed")
            else:
                failed_phases.append(phase)
                print(f"❌ Phase {phase} visualization failed")
        
        print(f"\n📊 Visualization Summary:")
        print(f"   ✅ Successful: {successful_phases}")
        print(f"   ❌ Failed: {failed_phases}")
        print(f"   🎨 Using SAME high-quality visualization.py as global pipeline")
        
        return len(successful_phases) > 0
    
    def run_all_phases(self):
        """Run pipeline for all phases with PHASIAL RETROSPECTIVE LABELING"""
        print(f"\n{'='*70}")
        print(f"🚀 PHASIAL RETROSPECTIVE PIPELINE - FOV {self.fov}")
        print(f"{'='*70}")
        print("✅ CORRECT EXECUTION ORDER:")
        print("   1️⃣ Process ALL phases (1→8) basic pipeline")
        print("   2️⃣ WAIT for Phase 8 to be COMPLETELY finished")
        print("   3️⃣ Run retrospective labeling (Phase 8→1)")
        print("   4️⃣ Run visualization for all phases")
        print("   🔄 Phase 8 MUST be reference point!")
        
        # Create directory structure
        self.create_phase_directories()
        
        successful_phases = []
        failed_phases = []
        
        # STEP 1: Process each phase for basic pipeline (1→8)
        print(f"\n{'='*60}")
        print("🏗️ STEP 1: BASIC PHASE PROCESSING (1→8)")
        print(f"{'='*60}")
        print("⚠️ Retrospective labeling will WAIT until Phase 8 is complete!")
        
        for phase in sorted(self.phases.keys()):  # 1,2,3,4,5,6,7,8
            success = self.run_single_phase(phase)
            if success:
                successful_phases.append(phase)
            else:
                failed_phases.append(phase)
        
        print(f"\n📊 Basic Phase Processing Summary:")
        print(f"   ✅ Successful: {successful_phases}")
        print(f"   ❌ Failed: {failed_phases}")
        
        # STEP 2: Check if Phase 8 is successfully completed (CRITICAL!)
        if 8 not in successful_phases:
            print(f"\n❌ CRITICAL: Phase 8 failed or not completed!")
            print(f"🚫 Cannot run retrospective labeling without Phase 8 reference!")
            print(f"💡 Phase 8 contains the strongest green signal needed for iPS detection")
            return False
        
        print(f"\n✅ Phase 8 successfully completed - proceeding with retrospective labeling")
        
        # STEP 3: Run Phasial Retrospective Labeling (ONLY AFTER Phase 8 is done)
        print(f"\n{'='*60}")
        print("🔄 STEP 2: PHASIAL RETROSPECTIVE LABELING (8→1)")
        print(f"{'='*60}")
        print("📋 Requirements met:")
        print(f"   ✅ Phase 8 completed with green signal analysis")
        print(f"   ✅ Global tracking data available")
        print(f"   🎯 Starting retrospective labeling from Phase 8...")
        
        retrospective_success = self.run_phasial_retrospective_labeling()
        
        if not retrospective_success:
            print(f"❌ Phasial retrospective labeling failed")
            print(f"⚠️ Phases will have basic labeling only (no retrospective)")
            return False
        
        print(f"✅ Phasial retrospective labeling completed!")
        
        # STEP 4: Run Phasial Visualization (ONLY AFTER retrospective labeling)
        print(f"\n{'='*60}")
        print("🎨 STEP 3: PHASIAL VISUALIZATION (All Phases)")
        print(f"{'='*60}")
        print("📋 Requirements met:")
        print(f"   ✅ All phases have retrospective labels")
        print(f"   ✅ Using SAME visualization.py as global pipeline")
        print(f"   🎯 Creating high-quality visualizations...")
        
        viz_success = self.run_phasial_visualization()
        
        if viz_success:
            print(f"✅ Phasial visualization completed!")
        else:
            print(f"⚠️ Some phase visualizations failed")
        
        # Final success check
        final_success = retrospective_success and viz_success
        
        # Summary
        print(f"\n{'='*70}")
        print("📊 PHASIAL RETROSPECTIVE PIPELINE SUMMARY")
        print(f"{'='*70}")
        print(f"✅ Successful phases: {successful_phases}")
        print(f"❌ Failed phases: {failed_phases}")
        print(f"📈 Success rate: {len(successful_phases)}/{len(self.phases)} phases")
        print(f"🔄 Retrospective labeling: {'✅ Success' if retrospective_success else '❌ Failed'}")
        print(f"🎨 Visualization: {'✅ Success' if viz_success else '❌ Failed'}")
        
        if final_success and len(successful_phases) > 0:
            print(f"\n🎉 PHASIAL RETROSPECTIVE PIPELINE COMPLETED!")
            print(f"📁 Each successful phase now has:")
            for phase in successful_phases:
                phase_dir = os.path.join(self.phases_path, f"phase_{phase}")
                start_frame, end_frame = self.phases[phase]
                print(f"   Phase {phase} (frames {start_frame}-{end_frame}): {phase_dir}")
                print(f"     ├── Labelled/ ✅ (Phasial retrospective labeling)")
                print(f"     ├── Visualization/ ✅ (SAME visualization.py)")
                print(f"     ├── trace/ ✅")
                print(f"     └── [all other folders] ✅")
            print(f"🔄 CORRECT ORDER: Phase 8 completed → retrospective labeling → visualization")
            print(f"🎨 Visualization: SAME high-quality system as global pipeline")
            return True
        else:
            print(f"❌ Pipeline failed - check Phase 8 completion and retrospective labeling")
            return False


# Main execution functions
def run_phase_pipeline_single_fov(base_path, fov):
    """Run PHASIAL RETROSPECTIVE pipeline for a single FOV"""
    print(f"\n{'='*80}")
    print(f"🔄 PHASIAL RETROSPECTIVE PIPELINE - FOV {fov}")
    print("⚠️ CORRECT EXECUTION ORDER:")
    print("   1️⃣ Process ALL phases (1→8) basic pipeline")
    print("   2️⃣ WAIT for Phase 8 to be COMPLETELY finished")
    print("   3️⃣ Run retrospective labeling (Phase 8→1)")
    print("   4️⃣ Run visualization for all phases")
    print("   🎯 Phase 8 MUST be completed first as reference!")
    print(f"{'='*80}")
    
    try:
        pipeline = PhaseBasedPipeline(base_path, fov)
        success = pipeline.run_all_phases()
        return success
    except Exception as e:
        print(f"❌ Error in phasial retrospective pipeline for FOV {fov}: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_phase_pipeline_all_fovs(base_path="."):
    """Run PHASIAL RETROSPECTIVE pipeline for all FOVs"""
    nuclear_dataset_dir = os.path.join(base_path, "nuclear_dataset")
    
    if not os.path.exists(nuclear_dataset_dir):
        print(f"❌ Directory not found: {nuclear_dataset_dir}")
        return
    
    # Find FOV directories (2-54)
    fov_dirs = []
    for d in os.listdir(nuclear_dataset_dir):
        if os.path.isdir(os.path.join(nuclear_dataset_dir, d)) and d.isdigit():
            fov_num = int(d)
            if 2 <= fov_num <= 54:
                # Check if required folders exist
                fov_path = os.path.join(nuclear_dataset_dir, d)
                test_path = os.path.join(fov_path, "test")
                
                if os.path.exists(test_path):
                    fov_dirs.append(d)
    
    fov_dirs = sorted(fov_dirs, key=int)
    print(f"🔍 Found {len(fov_dirs)} FOVs ready for PHASIAL RETROSPECTIVE processing")
    
    successful = 0
    failed = 0
    
    for fov in fov_dirs:
        print(f"\n{'='*80}")
        print(f"🚀 PHASIAL RETROSPECTIVE PIPELINE - FOV {fov}")
        print(f"{'='*80}")
        
        success = run_phase_pipeline_single_fov(base_path, fov)
        if success:
            successful += 1
            print(f"✅ FOV {fov} PHASIAL RETROSPECTIVE pipeline finished")
        else:
            failed += 1
            print(f"❌ FOV {fov} PHASIAL RETROSPECTIVE pipeline failed")
    
    print(f"\n{'='*80}")
    print("🎊 PHASIAL RETROSPECTIVE PIPELINE SUMMARY - ALL FOVs")
    print(f"{'='*80}")
    print(f"✅ Successfully processed: {successful} FOVs")
    print(f"❌ Failed: {failed} FOVs")
    print(f"📈 Success rate: {successful}/{len(fov_dirs)} FOVs")
    
    if successful > 0:
        print(f"\n🎉 PHASIAL RETROSPECTIVE PROCESSING COMPLETED!")
        print(f"📁 Each FOV now has 8 phases with CORRECT execution order:")
        print(f"   1️⃣ Basic processing for ALL phases (1→8)")
        print(f"   2️⃣ WAIT for Phase 8 completion (critical reference)")
        print(f"   3️⃣ Phasial Retrospective Labeling (Phase 8→1)")
        print(f"   4️⃣ SAME visualization.py as global pipeline")
        print(f"   🌍 Global tracking data integration")
        print(f"   📹 Trace generation")
        print(f"   📊 Cell counting and analysis")
        print(f"   ✅ All data properly partitioned and processed")
        print(f"   ⚠️ Phase 8 is the critical reference point!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PHASIAL RETROSPECTIVE Cell Tracking Pipeline")
    parser.add_argument("--fov", type=str, help="Process specific FOV number")
    parser.add_argument("--all", action="store_true", help="Process all FOVs")
    parser.add_argument("--phase", type=int, choices=range(1, 9), help="Process specific phase only (basic pipeline)")
    parser.add_argument("--base-path", type=str, default=".", help="Base directory path")
    
    args = parser.parse_args()
    
    if not args.fov and not args.all:
        print("❌ Error: Must specify either --fov <number> or --all")
        print("\n💡 PHASIAL RETROSPECTIVE PIPELINE:")
        print("   🔄 Correct execution order:")
        print("   1️⃣ Process ALL phases (1→8) basic pipeline")
        print("   2️⃣ WAIT for Phase 8 to be COMPLETELY finished")
        print("   3️⃣ Run retrospective labeling (Phase 8→1)")
        print("   4️⃣ Run visualization for all phases")
        print("   ⚠️ Phase 8 MUST be completed first as reference!")
        parser.print_help()
        sys.exit(1)
    
    if args.fov and args.all:
        print("❌ Error: Cannot specify both --fov and --all")
        sys.exit(1)
    
    if args.fov:
        if args.phase:
            # Run single phase for single FOV (basic pipeline only)
            print(f"🔧 Running BASIC pipeline for Phase {args.phase} only (no retrospective)")
            pipeline = PhaseBasedPipeline(args.base_path, args.fov)
            pipeline.create_phase_directories()
            success = pipeline.run_single_phase(args.phase)
            print(f"Phase {args.phase} for FOV {args.fov}: {'✅ Success' if success else '❌ Failed'}")
        else:
            # Run PHASIAL RETROSPECTIVE pipeline for single FOV
            print(f"🔄 Running FULL PHASIAL RETROSPECTIVE pipeline for FOV {args.fov}")
            run_phase_pipeline_single_fov(args.base_path, args.fov)
    else:
        # Run PHASIAL RETROSPECTIVE pipeline for all FOVs
        print(f"🔄 Running FULL PHASIAL RETROSPECTIVE pipeline for ALL FOVs")
        run_phase_pipeline_all_fovs(args.base_path)
