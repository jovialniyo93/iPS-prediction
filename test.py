import torch
#from net import Unet
#from net import FCN8s
#from net import DeepSeg
from net import TransDeepSeg
#from net import StarDistUNet
#from net import Cellpose
#from net import AttU_Net
#from net import NestedUNet
from utils import *
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from torch import nn
import shutil
import os
import cv2
from track import main
from generate_trace import get_trace, get_video
from count import count_cells_in_dataset, save_results_to_csv
from features import extract_features_from_test
from matplotlib import pyplot as plt

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = clahe.apply(img)
    return img

# Image enhancement function
def enhance(img):
    img = np.clip(img * 1.2, 0, 255)
    img = img.astype(np.uint8)
    return img

# Test function - makes predictions and saves results
def test(test_path, result_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    #model=Unet(1,1)
    #model = FCN8s(1, 1)
    #model = DeepSeg(1, 1)
    model = TransDeepSeg(1, 1)
    #model = StarDistUNet(1, 1)
    #model = Cellpose(1, 1)
    #model = AttU_Net(1, 1)
    #model = NestedUNet(1, 1)
    model.eval()
    model = torch.nn.DataParallel(model).cuda()
    model = model.to(device)
    
    # Load model state
    print(f"Loading model from checkpoints/CP_epoch100.pth")
    try:
        checkpoint = torch.load('checkpoints/CP_epoch100.pth')
        new_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("module."):
                new_state_dict[key[7:]] = value  # Remove `module.` prefix
            else:
                new_state_dict[f"module.{key}"] = value  # Add `module.` prefix
        model.load_state_dict(new_state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    test_data = TestDataset(test_path, transform=x_transforms)
    dataloader = DataLoader(test_data, batch_size=1)

    # Perform prediction
    with torch.no_grad():
        for index, x in enumerate(dataloader):
            x = x.to(device)
            y = model(x)
            y = y.cpu()
            y = torch.squeeze(y)
            img_y = torch.sigmoid(y).numpy()
            img_y = (img_y * 255).astype(np.uint8)
            # Save the predicted image
            output_path = os.path.join(result_path, f"predict_{index:06d}.tif")
            cv2.imwrite(output_path, img_y, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            print(f"Saved prediction {index+1}/{len(dataloader)}: {output_path}")

    print(f"{test_path} prediction finished!")
    return True

# Process and enhance the image
def process_img():
    img_root = "data/test/"
    n = len(os.listdir(img_root))
    for i in range(n):
        img_path = os.path.join(img_root, str(i).zfill(6)+".tif")
        img = cv2.imread(img_path, -1)
        img = np.uint8(np.clip((0.02 * img + 60), 0, 255))
        cv2.imwrite(img_path, img)

# Process images horizontally (combine)
def processImg2():
    directory = "data/test"
    img_list = os.listdir(directory)
    imgs = []
    for img_name in img_list:
        img_path = os.path.join(directory, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        imgs.append(img)
    whole = imgs[0]

    for i in range(1, len(imgs)):
        whole = np.hstack((whole, imgs[i]))
    whole = clahe(whole)
    for i, img_name in enumerate(img_list):
        img_path = os.path.join(directory, img_name)
        img = whole[:, 770 * i:770 * (i + 1)]
        cv2.imwrite(img_path, img)

# Add blur to image
def add_blur(img):
    new_img = cv2.GaussianBlur(img, (21, 21), 0)
    new_img = cv2.GaussianBlur(new_img, (5, 5), 0)
    return new_img

# Process prediction results: threshold and connected components
def process_predictResult(source_path, result_path):
    if not os.path.isdir(result_path):
        print('Creating RES directory...')
        os.mkdir(result_path)

    names = os.listdir(source_path)
    names = [name for name in names if '.tif' in name]
    names.sort()

    for name in names:
        predict_result = cv2.imread(os.path.join(source_path, name), -1)
        if predict_result is None:
            print(f"Error: Could not read {name} from {source_path}")
            continue
            
        # Apply thresholding
        ret, predict_result = cv2.threshold(predict_result, 127, 255, cv2.THRESH_BINARY)

        # Perform connected components analysis
        ret, markers = cv2.connectedComponents(predict_result)

        # Save results explicitly in TIFF format
        output_path = os.path.join(result_path, f"{os.path.splitext(name)[0]}.tif")
        cv2.imwrite(output_path, markers, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        print(f"Processed {name} -> {output_path}")

# Filter results by minimum area
def useAreaFilter(img, area_size):
    img_8bit = cv2.convertScaleAbs(img)
    contours, hierarchy = cv2.findContours(img_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_new = np.stack((img_8bit, img_8bit, img_8bit), axis=2)
    for cont in contours:
        area = cv2.contourArea(cont)
        if area < area_size:
            img_new = cv2.fillConvexPoly(img_new, cont, (0, 0, 0))
    img = img_new[:, :, 0]
    return img

# Delete files in the specified directory
def delete_file(path):
    if not os.path.isdir(path):
        print(path, " does not exist!")
        os.mkdir(path)
        return
    file_list = os.listdir(path)
    for file in file_list:
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    print(path, " has been cleaned!")

# Create a folder if it doesn't exist
def createFolder(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        print(path, " has been created.")
    else:
        print(path, " already exists.")

# Main function that integrates everything
def process_dataset(folder):
    """Process a single dataset folder with all steps: prediction, tracking, tracing, counting, and feature extraction"""
    print(f"\n{'='*80}")
    print(f"Processing dataset: {folder}")
    print(f"{'='*80}")
    
    # Setup paths
    test_path = os.path.join(folder, "test")
    test_result_path = os.path.join(folder, "test_result")
    res_path = os.path.join(folder, "res")
    res_result_path = os.path.join(folder, "res_result")
    track_result_path = os.path.join(folder, "track_result")
    trace_path = os.path.join(folder, "trace")
    features_path = os.path.join(folder, "features")

    # Create necessary folders
    for path in [test_result_path, res_path, res_result_path, track_result_path, trace_path, features_path]:
        createFolder(path)

    # Step 1: Perform prediction (segmentation)
    print("\nStep 1: Running neural network prediction...")
    prediction_success = test(test_path, test_result_path)
    if not prediction_success:
        print("Error in prediction step. Stopping processing for this dataset.")
        return False

    # Step 2: Process prediction results (connected components)
    print("\nStep 2: Processing prediction results...")
    try:
        process_predictResult(test_result_path, res_path)
    except Exception as e:
        print(f"Error processing prediction results: {e}")
        return False

    # Step 3: Filter results by area
    print("\nStep 3: Applying area filter...")
    try:
        result_files = sorted([f for f in os.listdir(res_path) if f.endswith('.tif')])
        for picture in result_files:
            input_path = os.path.join(res_path, picture)
            output_path = os.path.join(res_result_path, picture)
            
            image = cv2.imread(input_path, -1)
            if image is None:
                print(f"Warning: Could not read {input_path}")
                continue
                
            image = useAreaFilter(image, 100)
            cv2.imwrite(output_path, image)
            print(f"Filtered {picture} by area")
    except Exception as e:
        print(f"Error applying area filter: {e}")
        return False

    # Step 4: Perform tracking
    print("\nStep 4: Running cell tracking...")
    try:
        main(res_result_path,track_result_path)
    except Exception as e:
        print(f"Error during tracking: {e}")
        return False

    # Step 5: Generate traces
    print("\nStep 5: Generating traces...")
    try:
        get_trace(test_path, track_result_path, trace_path)
    except Exception as e:
        print(f"Error generating traces: {e}")
        return False

    # Step 6: Create video from traced images
    print("\nStep 6: Creating trace video...")
    try:
        get_video(trace_path)
    except Exception as e:
        print(f"Error creating video: {e}")
        print("Continuing to next step anyway...")

    # Step 7: Count cells and save results
    print("\nStep 7: Counting cells and tracking missing cells...")
    try:
        results, missing_cells_tracker, overall_counts = count_cells_in_dataset(
            test_path, track_result_path, trace_path
        )
        
        count_output_path = os.path.join(folder, "cell_count_results.csv")
        save_results_to_csv(
            results, missing_cells_tracker, overall_counts, count_output_path
        )
        print(f"Cell counting results saved to {count_output_path}")
    except Exception as e:
        print(f"Error counting cells: {e}")
        print("Continuing to next step anyway...")

    # Step 8: Extract features
    print("\nStep 8: Extracting cell features...")
    try:
        extract_features_from_test(test_path, track_result_path, features_path)
        print(f"Features extracted and saved to {features_path}")
    except Exception as e:
        print(f"Error extracting features: {e}")
        print("Feature extraction failed, but previous steps completed.")

    print(f"\nProcessing completed for {folder}")
    return True

if __name__ == "__main__":
    try:
        # Look for datasets in the nuclear_dataset directory
        dataset_base = "nuclear_dataset"
        if not os.path.exists(dataset_base):
            print(f"Error: {dataset_base} directory does not exist.")
            exit(1)
            
        test_folders = sorted([os.path.join(dataset_base, folder) for folder in os.listdir(dataset_base) if os.path.isdir(os.path.join(dataset_base, folder))])
        
        if not test_folders:
            print(f"No dataset folders found in {dataset_base}")
            exit(1)
            
        print(f"Found {len(test_folders)} dataset folders to process:")
        for folder in test_folders:
            print(f" - {folder}")
        
        # Process each dataset folder
        successful = 0
        failed = 0
        
        for folder in test_folders:
            success = process_dataset(folder)
            if success:
                successful += 1
            else:
                failed += 1
                
        print(f"\nProcessing summary:")
        print(f"Successfully processed: {successful} datasets")
        print(f"Failed to process: {failed} datasets")
        
        if failed > 0:
            print("Please check the error messages above for details on failures.")
        
    except Exception as e:
        print(f"Unexpected error in main execution: {e}")
