import os
import pandas as pd

def get_fov_dirs():
    """Get all field of view directories (2 to 54)."""
    nuclear_dataset_dir = "nuclear_dataset"
    if not os.path.exists(nuclear_dataset_dir):
        print(f"Error: {nuclear_dataset_dir} directory not found")
        return []
    
    fov_dirs = []
    for d in os.listdir(nuclear_dataset_dir):
        if os.path.isdir(os.path.join(nuclear_dataset_dir, d)) and d.isdigit():
            fov_num = int(d)
            if 2 <= fov_num <= 54:
                fov_dirs.append(d)
    
    return sorted(fov_dirs, key=int)

def read_and_arrange_features(fov):
    """Read features.csv for a specific FOV, arrange by Cell ID, and drop N/A values."""
    features_path = os.path.join("nuclear_dataset", fov, "track_result", "features.csv")
    if not os.path.exists(features_path):
        print(f"Warning: {features_path} does not exist")
        return None, None
    
    try:
        df = pd.read_csv(features_path)
        print(f"  Original data: {len(df['Cell ID'].unique())} cells, {len(df)} total rows")
        
        # Drop specific columns as requested
        columns_to_drop = ['Parent ID', 'Daughter IDs', 'Division Frame']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df = df.drop(columns=existing_columns_to_drop)
            print(f"  Dropped columns: {', '.join(existing_columns_to_drop)}")
        
        # Drop N/A values (following consensus.py pattern)
        df_clean = df.dropna(subset=['X', 'Y', 'Frame'])
        
        # Additional cleanup for other important columns if they exist
        if 'Cell ID' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['Cell ID'])
        
        print(f"  After dropping N/A: {len(df_clean['Cell ID'].unique())} cells, {len(df_clean)} total rows")
        
        # Sort by Cell ID first, then by Frame (following consensus.py sorting pattern)
        arranged_df = df_clean.sort_values(['Cell ID', 'Frame']).reset_index(drop=True)
        
        # Move Frame column to the last position
        if 'Frame' in arranged_df.columns:
            cols = [col for col in arranged_df.columns if col != 'Frame']
            cols.append('Frame')  # Add Frame at the end
            arranged_df = arranged_df[cols]
            print(f"  Moved Frame column to last position")
        
        print(f"  Arranged by Cell ID and Frame")
        
        # Get the track_result directory path for saving
        track_result_dir = os.path.join("nuclear_dataset", fov, "track_result")
        
        return arranged_df, track_result_dir
        
    except Exception as e:
        print(f"Error reading {features_path}: {e}")
        return None, None

def save_arranged_data(arranged_df, track_result_dir, fov):
    """Save the arranged data to arranged.csv in the track_result folder."""
    if arranged_df is None or arranged_df.empty:
        print(f"  No data to save for FOV {fov}")
        return
    
    output_path = os.path.join(track_result_dir, "arranged.csv")
    
    try:
        arranged_df.to_csv(output_path, index=False)
        print(f"  Successfully saved arranged.csv to {output_path}")
    except Exception as e:
        print(f"  Error saving to {output_path}: {e}")

def main():
    """Main processing function."""
    print("Starting data arrangement for MU-US algorithm")
    print("Reading features.csv from each FOV, dropping N/A values, arranging by Cell ID")
    print("Saving arranged.csv in each track_result folder\n")
    
    # Get all FOV directories
    fov_dirs = get_fov_dirs()
    if not fov_dirs:
        print("No valid FOV directories found (expecting 2-54)")
        return
    
    print(f"Found {len(fov_dirs)} FOV directories: {', '.join(fov_dirs)}\n")
    
    # Process each FOV individually
    total_processed = 0
    total_cells_all_fovs = 0
    total_rows_all_fovs = 0
    
    for fov in fov_dirs:
        print(f"Processing FOV {fov}...")
        
        # Read, clean, and arrange data for this FOV
        arranged_df, track_result_dir = read_and_arrange_features(fov)
        
        if arranged_df is not None and not arranged_df.empty:
            # Save arranged.csv in the track_result folder
            save_arranged_data(arranged_df, track_result_dir, fov)
            
            # Update totals
            unique_cells = len(arranged_df['Cell ID'].unique())
            total_cells_all_fovs += unique_cells
            total_rows_all_fovs += len(arranged_df)
            total_processed += 1
            
            # Show sample data for verification
            print(f"  Sample arranged data (Cell ID, Frame):")
            sample_data = arranged_df[['Cell ID', 'Frame']].head(5)
            for _, row in sample_data.iterrows():
                print(f"    Cell {row['Cell ID']}, Frame {row['Frame']}")
                
        else:
            print(f"  Skipping FOV {fov}: no valid data after cleaning")
        
        print()  # Empty line for readability
    
    print("=" * 50)
    print("SUMMARY:")
    print(f"  FOVs processed successfully: {total_processed}/{len(fov_dirs)}")
    print(f"  Total cells across all FOVs: {total_cells_all_fovs}")
    print(f"  Total rows across all FOVs: {total_rows_all_fovs}")
    print(f"  arranged.csv files created in each track_result folder")
    print("=" * 50)
    
    if total_processed > 0:
        print(f"\nData arrangement completed successfully!")
        print(f"Each arranged.csv is saved in its respective track_result folder")
    else:
        print("\nNo data was processed successfully")

if __name__ == "__main__":
    main()
