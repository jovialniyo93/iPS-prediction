import cv2
from torch.utils.data import Dataset
from pathlib import Path
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy import ndimage as ndi

class BasicSegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, transforms=None, mask_suffix: str = '',if_train_aug=False,train_aug_iter=1):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        self.transforms=transforms
        
        # Get all image files first
        all_files = [os.path.splitext(file)[0] for file in sorted(os.listdir(images_dir)) if not file.startswith('.')]
        
        # Filter to only include files that have both valid image and mask
        self.ids = []
        skipped_count = 0
        
        for file_id in all_files:
            # Check if both image and mask exist and are readable
            img_files = list(self.images_dir.glob(file_id + '.*'))
            mask_files = list(self.masks_dir.glob(file_id + self.mask_suffix + '.*'))
            
            if len(img_files) != 1 or len(mask_files) != 1:
                skipped_count += 1
                continue
                
            # Test if files are readable
            img_path = img_files[0].as_posix()
            mask_path = mask_files[0].as_posix()
            
            # Test image
            test_img = cv2.imread(img_path, 0)
            if test_img is None:
                print(f"Warning: Cannot read image {img_path}, skipping...")
                skipped_count += 1
                continue
                
            # Test mask
            test_mask = cv2.imread(mask_path, 0)
            if test_mask is None:
                print(f"Warning: Cannot read mask {mask_path}, skipping...")
                skipped_count += 1
                continue
                
            # If both are readable, add to valid IDs
            self.ids.append(file_id)
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} files due to missing or corrupted images/masks")
            
        if not self.ids:
            raise RuntimeError(f'No valid file pairs found in {images_dir}, make sure you put your images there')
        
        print(f"Found {len(self.ids)} valid image-mask pairs")
        
        # Apply augmentation multiplier if needed
        if if_train_aug:
            tmp = []
            for i in range(train_aug_iter+1):
                tmp += self.ids
            self.ids = tmp

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, pil_mask, transforms):
        tensor_img, tensor_mask = transforms(pil_img, pil_mask)
        return tensor_img, tensor_mask

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        # These should never fail now due to pre-filtering, but keep as safety
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        # Read mask with error handling
        mask_path = mask_file[0].as_posix()
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise RuntimeError(f"Cannot read mask file: {mask_path}")
        mask = mask > 0
        mask = mask.astype('float32')
        
        # Read image with error handling
        img_path = img_file[0].as_posix()
        img = cv2.imread(img_path, 0)
        if img is None:
            raise RuntimeError(f"Cannot read image file: {img_path}")
        img = img.astype('float32')
        img = (255 * ((img - img.min()) / (np.ptp(img)+1e-6))).astype(np.uint8)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        tensor_img, tensor_mask = self.preprocess(img, mask, self.transforms)

        return {
            'image': tensor_img,
            'mask': tensor_mask
        }

class BasicTrackerDataset(Dataset):
    def __init__(self, data_dir: str, transforms=None, mask_suffix: str = '_cell_area_masked',label_suffix: str = '_cell_pos_labels',window_size=70,if_train_aug=False,train_aug_iter=1,if_test=False):
        self.mask_suffix = mask_suffix
        self.transforms=transforms
        self.ids={}
        cell_id=0
        print('INFO: Read subfolders of video sequences and prepare training dataset ...')
        print('INFO: Wait until finished, it takes relatively long time depending on dataset size and complexity ...')
        for subfolder in tqdm(sorted(os.listdir(data_dir))):
            image_list=sorted(os.listdir(os.path.join(data_dir,subfolder,'images')))
            for idx in range(len(image_list)-1):
                img_curr_name = image_list[idx+1]
                
                # Add error handling for image reading
                try:
                    image_curr = cv2.imread(os.path.join(data_dir, subfolder, 'images', img_curr_name))
                    if image_curr is None:
                        print(f"Warning: Cannot read {img_curr_name}, skipping...")
                        continue
                    image_curr = image_curr[:, :, 0]
                    
                    mask_curr = cv2.imread(os.path.join(data_dir, subfolder, 'masks', img_curr_name.replace('.tif', mask_suffix + '.tif')))
                    if mask_curr is None:
                        print(f"Warning: Cannot read mask for {img_curr_name}, skipping...")
                        continue
                    mask_curr = mask_curr[:,:, 0]

                    img_prev_name=image_list[idx]
                    image_prev = cv2.imread(os.path.join(data_dir, subfolder, 'images', img_prev_name))
                    if image_prev is None:
                        print(f"Warning: Cannot read {img_prev_name}, skipping...")
                        continue
                    image_prev = image_prev[:, :, 0]
                    
                    mask_prev=cv2.imread(os.path.join(data_dir,subfolder,'masks',img_prev_name.replace('.tif',mask_suffix+'.tif')))
                    if mask_prev is None:
                        print(f"Warning: Cannot read mask for {img_prev_name}, skipping...")
                        continue
                    mask_prev = mask_prev[:, :, 0]
                    
                except Exception as e:
                    print(f"Error reading files for {img_curr_name}: {e}")
                    continue
                
                try:
                    lines=open(os.path.join(data_dir,subfolder,'labels',img_prev_name.replace('.tif',label_suffix+'.txt'))).readlines()
                    lines.pop(0)
                    labels_prev,centroids_prev=[],[]
                    for line in lines:
                        line_info=line.replace('\n','').split('\t')
                        labels_prev.append(line_info[0])
                        centroids_prev.append([int(line_info[1]),int(line_info[2])])

                    lines = open(os.path.join(data_dir, subfolder, 'labels',img_curr_name.replace('.tif', label_suffix + '.txt'))).readlines()
                    lines.pop(0)
                    labels_curr, centroids_curr = [], []
                    for line in lines:
                        line_info = line.replace('\n', '').split('\t')
                        labels_curr.append(line_info[0])
                        centroids_curr.append([int(line_info[1]), int(line_info[2])])
                except Exception as e:
                    print(f"Error reading label files: {e}")
                    continue

                markers_prev, num_labels_prev = ndi.label(mask_prev)
                markers_curr, num_labels_curr = ndi.label(mask_curr)
                for i in range(1,num_labels_prev):
                    if np.sum(markers_prev==i)<20:
                        continue
                    centroid = np.array(ndi.measurements.center_of_mass(mask_prev, markers_prev, i))
                    distances = np.array([np.sqrt(np.sum((centroid - np.array([point[1], point[0]])) ** 2)) for point in centroids_prev])
                    if len(np.where(distances < 5)[0])==0:
                        continue
                    target_cell_prev_idx = np.where(distances < 5)[0][0]
                    label_prev=labels_prev[target_cell_prev_idx]
                    if label_prev in labels_curr:
                        target_cell_curr_idx = [labels_curr.index(label_prev)]
                    elif label_prev+'_1' in labels_curr and label_prev+'_2' in labels_curr:
                        target_cell_curr_idx=[labels_curr.index(label_prev+'_1'),labels_curr.index(label_prev+'_2')]
                    else:
                        continue

                    crop_prev = markers_prev.copy()
                    crop_prev[crop_prev != i] = 0
                    crop_prev[crop_prev>0] = 1
                    tmp = np.where(crop_prev > 0)
                    crop_prev = crop_prev.astype('float32') * image_prev
                    crop_curr = (mask_curr > 0).astype('float32') * image_curr
                    window_size = np.max([(np.max(tmp[0]) - np.min(tmp[0])) * 6, (np.max(tmp[1]) - np.min(tmp[1])) * 6])
                    crop_prev = Image.fromarray(crop_prev)
                    crop_prev = crop_prev.crop((int(centroid[1] - window_size / 2), int(centroid[0] - window_size / 2),
                                                int(centroid[1] + window_size / 2), int(centroid[0] + window_size / 2)))
                    crop_prev = np.uint8(np.asarray(crop_prev))

                    crop_curr = Image.fromarray(crop_curr)
                    crop_curr = crop_curr.crop((int(centroid[1] - window_size / 2), int(centroid[0] - window_size / 2),
                                                int(centroid[1] + window_size / 2), int(centroid[0] + window_size / 2)))
                    crop_curr = np.uint8(np.asarray(crop_curr))

                    crop_out=np.zeros(markers_curr.shape)
                    for idx in target_cell_curr_idx:
                        traget_marker_curr_val=markers_curr[centroids_curr[idx][1],centroids_curr[idx][0]]
                        tmp = markers_curr.copy()
                        tmp[tmp != traget_marker_curr_val] = 0
                        tmp[tmp > 0] = 1
                        crop_out+=tmp
                    crop_out[crop_out>0]=1
                    crop_out = Image.fromarray(crop_out)
                    crop_out = crop_out.crop((int(centroid[1] - window_size / 2), int(centroid[0] - window_size / 2),
                                                int(centroid[1] + window_size / 2), int(centroid[0] + window_size / 2)))
                    crop_out = np.uint8(np.asarray(crop_out).astype('float32') * 255)

                    if np.sum(crop_out):
                        num_copies=1
                        if len(target_cell_curr_idx)>1 and if_test==False:
                           num_copies=50
                        if if_train_aug:
                            for _ in range(num_copies*train_aug_iter):
                                self.ids[cell_id]=[crop_prev,crop_curr,crop_out]
                                cell_id += 1
                        else:
                            for _ in range(num_copies):
                                self.ids[cell_id]=[crop_prev,crop_curr,crop_out]
                                cell_id += 1


    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img_prev, pil_curr,pil_mask,transforms):
        tensor_img_prev,tensor_img_curr,tensor_mask=transforms(pil_img_prev, pil_curr,pil_mask)
        return tensor_img_prev,tensor_img_curr,tensor_mask

    def __getitem__(self, idx):
        img_prev,img_curr,mask = self.ids[idx]
        tensor_img_prev,tensor_img_curr,tensor_mask = self.preprocess(img_prev,img_curr,mask,self.transforms)

        return {
            'image_prev': tensor_img_prev,
            'image_curr': tensor_img_curr,
            'mask': tensor_mask
        }