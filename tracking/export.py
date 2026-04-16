"""Utilities to export tracking results to ctc metrics readable format
 (tracking masks + txt file with lineage)"""
import os

import numpy as np
import pandas as pd
#from tifffile import imsave
from tifffile import imread, imwrite
'''class ExportResults:
    """Exports tracking results in a ctc tracking metrics readable format."""
    def __init__(self):
        self.img_file_name = 'mask'
        self.img_file_ending = '.tif'
        self.track_file_name = 'res_track.txt'
        self.time_steps = None

    def __call__(self, tracks, export_dir, img_shape, time_steps):
        """
        Exports tracks to a given export directory.
        Args:
            tracks: a dictionary containing the trajectories
            export_dir: string path to the directory where results will be written to
            img_shape: a tuple providing the img shape of the original data
            time_steps: a list of time steps the tracking was applied to

        Returns:

        """
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        tracks = remove_short_tracks(tracks)
        tracks = fill_empty_tracking_images(tracks, time_steps)
        self.time_steps = time_steps
        self.create_track_file(tracks, export_dir)
        self.create_segm_masks(tracks, export_dir, img_shape)'''
'''class ExportResults:
    def __call__(self, tracks, export_dir, img_shape, time_steps):
        # Enhanced track cleaning
        tracks = self._clean_tracks(tracks)
        tracks = self._resolve_conflicts(tracks)
        self._export_results(tracks, export_dir, img_shape, time_steps)

    def _clean_tracks(self, tracks):
        """Advanced track filtering"""
        valid_tracks = {}
        
        # Calculate statistics
        track_lengths = [len(t.masks) for t in tracks.values()]
        median_length = np.median(track_lengths)
        
        for track_id, track in tracks.items():
            # Dynamic length threshold
            min_length = max(3, median_length * 0.3)
            
            if len(track.masks) >= min_length:
                # Additional quality checks
                positions = [np.median(m, axis=1) for m in track.masks.values()]
                movement = np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=0))
                
                if np.all(movement < img_shape[0] * 0.2):  # Filter stuck tracks
                    valid_tracks[track_id] = track
                    
        return valid_tracks

    def _resolve_conflicts(self, tracks):
        """Handle overlapping tracks"""
        # Implement conflict resolution logic
        return tracks

    def create_track_file(self, all_tracks, export_dir):
        """
        Creates a res_track.txt file readable by TRA measure.
        Args:
            all_tracks: a dictionary containing the trajectories
            export_dir: string path to the directory where results will be written to

        Returns:

        """
        track_info = {'track_id': [], 't_start': [], 't_end': [], 'predecessor_id': []}
        for track in all_tracks.values():

            track_info['track_id'].append(track.track_id)
            frame_ids = sorted(list(track.masks.keys()))
            track_info['t_start'].append(frame_ids[0])
            track_info['t_end'].append(frame_ids[-1])
            track_info['predecessor_id'].append(track.pred_track_id)
        df = pd.DataFrame.from_dict(track_info)
        df.to_csv(os.path.join(export_dir, self.track_file_name),
                  columns=["track_id", "t_start", "t_end", 'predecessor_id'],
                  sep=' ', index=False, header=False)


    def create_segm_masks(self, all_tracks, export_dir, img_shape):
        tracks_in_frame = {time: [] for time in self.time_steps}
        for track_data in all_tracks.values():
            time_steps = sorted(list(track_data.masks.keys()))
            for time in time_steps:
                tracks_in_frame[time].append(track_data.track_id)
            # Force 6-digit numbering to match input files
            z_fill = 6
            for time, track_ids in tracks_in_frame.items():
                all_tracking_masks = np.zeros(img_shape, dtype=np.uint16)
                for t_id in track_ids:
                    track = all_tracks[t_id]
                    mask = track.masks[time]
                    if not isinstance(mask, tuple):
                        mask = tuple(*mask)
                    all_tracking_masks[mask] = t_id
                # Consistent predict_XXXXXX.tif naming
                file_name = f"predict_{str(time).zfill(z_fill)}.tif"
                imwrite(os.path.join(export_dir, file_name), all_tracking_masks.astype('uint16'))'''

class ExportResults:
    """Exports tracking results in a ctc tracking metrics readable format."""
    def __init__(self):
        self.img_file_name = 'mask'
        self.img_file_ending = '.tif'
        self.track_file_name = 'res_track.txt'
        self.time_steps = None

    def __call__(self, tracks, export_dir, img_shape, time_steps):
        """
        Exports tracks to a given export directory.
        Args:
            tracks: a dictionary containing the trajectories
            export_dir: string path to the directory where results will be written to
            img_shape: a tuple providing the img shape of the original data
            time_steps: a list of time steps the tracking was applied to
        """
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        # Store img_shape as instance variable
        self.img_shape = img_shape  
        self.time_steps = time_steps
        
        # Clean and process tracks
        tracks = self._clean_tracks(tracks)
        tracks = fill_empty_tracking_images(tracks, time_steps)
        
        # Export results
        self.create_track_file(tracks, export_dir)
        self.create_segm_masks(tracks, export_dir, img_shape)

    def _clean_tracks(self, tracks):
        """Advanced track filtering with dynamic length threshold"""
        valid_tracks = {}
        
        # Calculate statistics
        track_lengths = [len(t.masks) for t in tracks.values()]
        median_length = np.median(track_lengths) if track_lengths else 0
        
        for track_id, track in tracks.items():
            # Dynamic length threshold
            min_length = max(3, median_length * 0.3)
            
            if len(track.masks) >= min_length:
                # Additional quality checks
                positions = [np.median(np.stack(m), axis=1) for m in track.masks.values()]
                if len(positions) > 1:  # Only check movement if multiple positions exist
                    movement = np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=0))
                    if np.all(movement < self.img_shape[0] * 0.2):  # Filter stuck tracks
                        valid_tracks[track_id] = track
                else:
                    valid_tracks[track_id] = track
                    
        return valid_tracks

    def create_track_file(self, all_tracks, export_dir):
        """Creates res_track.txt file with lineage information"""
        track_info = {'track_id': [], 't_start': [], 't_end': [], 'predecessor_id': []}
        for track in all_tracks.values():
            track_info['track_id'].append(track.track_id)
            frame_ids = sorted(list(track.masks.keys()))
            track_info['t_start'].append(frame_ids[0])
            track_info['t_end'].append(frame_ids[-1])
            track_info['predecessor_id'].append(track.pred_track_id)
        
        df = pd.DataFrame.from_dict(track_info)
        df.to_csv(os.path.join(export_dir, self.track_file_name),
                columns=["track_id", "t_start", "t_end", 'predecessor_id'],
                sep=' ', index=False, header=False)

    def create_segm_masks(self, all_tracks, export_dir, img_shape):
        """Creates tracking masks for each time point"""
        tracks_in_frame = {time: [] for time in self.time_steps}

        for track_data in all_tracks.values():
            time_steps = sorted(list(track_data.masks.keys()))
            for time in time_steps:
                tracks_in_frame[time].append(track_data.track_id)

        # Use consistent 6-digit numbering
        z_fill = 6  
        
        for time, track_ids in tracks_in_frame.items():
            tracked_img = np.zeros(img_shape, dtype=np.uint16)
            for t_id in track_ids:
                mask = all_tracks[t_id].masks[time]
                if isinstance(mask, tuple):
                    coords = np.ravel_multi_index(mask, img_shape)
                    tracked_img.flat[coords] = t_id
                else:
                    tracked_img[mask] = t_id
            
            file_name = f"predict_{str(time).zfill(z_fill)}.tif"
            imwrite(os.path.join(export_dir, file_name), tracked_img)

            
def remove_short_tracks(all_tracks):
    """
    Removes single time tracks without predecessor+successor
    Args:
        all_tracks:  a dictionary containing the trajectories

    Returns:
        a dictionary containing the edited trajectories
    """
    predecessor = [track.pred_track_id for track in all_tracks.values()]
    temp = all_tracks.copy()
    for track_id, track in all_tracks.items():
        frame_ids = sorted(list(track.masks.keys()))
        if (len(frame_ids) == 1) and (track.pred_track_id == 0) and (track_id not in predecessor):
            temp.pop(track_id)
    return temp


def fill_empty_tracking_images(all_tracks, time_steps):
    """
    Fills missing tracking frames with the temporally closest, filled tracking frame
    Args:
        all_tracks:  a dictionary containing the trajectories
        time_steps: a list of time steps the tracking was run on

    Returns:
        a dictionary containing the edited trajectories

    """
    tracks_in_frame = {}
    for track_data in all_tracks.values():
        track_timesteps = sorted(list(track_data.masks.keys()))
        for time in track_timesteps:
            if time not in tracks_in_frame:
                tracks_in_frame[time] = []
            tracks_in_frame[time].append(track_data.track_id)
    if sorted(time_steps) != sorted(list(tracks_in_frame.keys())):
        empty_timesteps = sorted(np.array(time_steps)[~np.isin(time_steps, list(tracks_in_frame.keys()))])
        filled_timesteps = np.array(sorted(list(tracks_in_frame.keys())))
        for empty_frame in empty_timesteps:
            nearest_filled_frame = filled_timesteps[np.argmin(abs(filled_timesteps-empty_frame))]
            track_ids = tracks_in_frame[nearest_filled_frame]
            for track_id in track_ids:
                all_tracks[track_id].masks[empty_frame] = all_tracks[track_id].masks[nearest_filled_frame]
            tracks_in_frame[empty_frame] = track_ids
            filled_timesteps = np.array(sorted(list(tracks_in_frame.keys())))
    return all_tracks
