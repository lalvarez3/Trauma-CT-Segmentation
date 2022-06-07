import glob
import SimpleITK as sitk
import os
import sys
import numpy as np
import shutil
from tqdm import tqdm


task_name = 'Task504_LiverSpleenTrauma' #change here for different task name
MODE = 'train'

def copy_and_rename(old_location,old_file_name,new_location,new_filename,delete_original = False):

    shutil.copy(os.path.join(old_location,old_file_name),new_location)
    os.rename(os.path.join(new_location,old_file_name),os.path.join(new_location,new_filename))
    if delete_original:
        os.remove(os.path.join(old_location,old_file_name))

def make_if_dont_exist(folder_path,overwrite=False):
    """
    creates a folder if it does not exists
    input: 
    folder_path : relative path of the folder which needs to be created
    over_write :(default: False) if True overwrite the existing folder 
    """
    if os.path.exists(folder_path):
        
        if not overwrite:
            print(f'{folder_path} exists.')
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)

    else:
      os.makedirs(folder_path)
      print(f"{folder_path} created!")


def adapt_overlay(overlay_path, mha_path, label):
        # Load the mha
        mha_data = sitk.ReadImage(mha_path)
        mha_org = mha_data.GetOrigin()[-1]
        # Load the mha image
        mha_img = sitk.GetArrayFromImage(mha_data)

        if mha_img.shape == label.shape: return label
        
        original_z_size = mha_img.shape[0]

        # Load the overlay
        overlay_data = sitk.ReadImage(overlay_path)
        overlay_org = overlay_data.GetOrigin()[-1]

        overlay_init = np.abs(1/mha_data.GetSpacing()[-1]*(mha_org-overlay_org) )

        lower_bound = int(overlay_init)
        upper_bound = label.shape[0]
        zeros_up = lower_bound
        zeros_down = original_z_size - (upper_bound + lower_bound)
        new = list()

        if zeros_up > 0:
            new.append(np.zeros((zeros_up, label.shape[1], label.shape[2]), dtype=label.dtype))

        new.append(label)

        if zeros_down > 0:
            new.append(np.zeros((zeros_down, label.shape[1], label.shape[2]), dtype=label.dtype))

        label = np.concatenate(new, axis=0)

        return label

def one_channel_overlay(img):
    mha_img = sitk.GetArrayFromImage(img)

    if len(mha_img.shape) == 3:
        z,h,w = mha_img.shape
        c = 1
    else:
        c, z, h, w = mha_img.shape

    new_labels = np.zeros((z, h, w), np.int8)

    # Depending on the number of channels, we have to adapt the overlay (GC has 1 working, the other has 6 channels)
    # We want 0 and 2 for channels and 
    if c == 1:
        # we need to transform Liver (6) to 1
        liver_channel = np.where((mha_img != 6), 0, mha_img)
        liver_channel = np.where((liver_channel == 6), 1, liver_channel)
        spleen_channel = np.where((mha_img != 1), 0, mha_img)
        spleen_channel = np.where((spleen_channel == 1), 2, spleen_channel)
        new_labels = liver_channel + spleen_channel
    else:
        # each channel is a different label, we want 0(liver = 1) and 2(liver injure = 2) and the background to be 0 
        channels = list()
        for i, channel in enumerate([0,1,2,3,4,5]):  # TODO: desharcodead
               c = mha_img[channel, :, :, :]
               c = np.where((c == 1), i+1, c)
               channels.append(c)

        # we got channels splitted with 1 each. if we add them we should get what we want
        chans = np.stack(channels, axis=0)
        new_labels = np.max(chans, axis=0)

    return new_labels
                        

def save_csv(output_path, data):
    import csv
    keys = data[0].keys()
    a_file = open(output_path, "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    a_file.close()


train_images = sorted(
    glob.glob(
        os.path.join("/mnt/chansey", "lauraalvarez", "data", "liver_spleen_nnunet", MODE, "data", "*.mha")
    )
)
train_labels = sorted(
    glob.glob(
        os.path.join(
            "/mnt/chansey/", "lauraalvarez", "data", "liver_spleen_nnunet", MODE, "mask", "*.mha"
        )
    )
)

BASE_PATH = os.path.join( "/mnt/chansey", "lauraalvarez", "nnunet", "nnUNet_raw_data_base", "nnUNet_raw_data")

task_folder_name = os.path.join(BASE_PATH, task_name)

if MODE == 'train':
    train_image_dir = os.path.join(task_folder_name,'imagesTr')
    train_label_dir = os.path.join(task_folder_name,'labelsTr')

else:
    train_image_dir = os.path.join(task_folder_name,'imagesTs')
    train_label_dir = os.path.join(task_folder_name,'labelsTs')


make_if_dont_exist(task_folder_name,overwrite = False)
make_if_dont_exist(train_image_dir)
make_if_dont_exist(train_label_dir)
# make_if_dont_exist(test_dir,overwrite= False)


equivalence_l = list()


for i in tqdm(range(0, len(train_images))):
    
        save_filename = "TRMLIV_%03i_0000.nii.gz" % i
        equiv = {"mha": os.path.basename(train_images[i]).split(".")[0], "nii": save_filename}
        equivalence_l.append(equiv)
        print("Converting {}".format(train_images[i]))
        try:
            img = sitk.ReadImage(train_images[i])
            img = sitk.DICOMOrient(img, 'RAS')        
            print("Saving to  {}".format(os.path.join(train_image_dir, save_filename)))
            sitk.WriteImage(img, os.path.join(train_image_dir, save_filename))
             = sitk.ReadImage(os.path.join(train_imageimg_dir, save_filename))
            filename = os.path.basename(train_images[i])
            labelpath =  os.path.join(
                    "/mnt/chansey", "lauraalvarez", "data", "liver_spleen_nnunet", MODE, "mask", filename
                )
            print(f"Converting mask for {labelpath}")
            save_filename = "TRMLIV_%03i.nii.gz" % i
            try:
                img_mask = sitk.ReadImage(labelpath)
            except Exception as e:
                print(e)
                print("Error reading {}".format(labelpath))
                continue
            # Adapt the channel
            img_array = one_channel_overlay(img_mask)
            # Transform the label to the same size as the image
            img_array = adapt_overlay(labelpath, train_images[i], img_array)
            img_array = sitk.GetImageFromArray(img_array)

            img_array.SetSpacing(img.GetSpacing())
            img_array.SetOrigin(img.GetOrigin())
            # img_array.SetDirection(img.GetDirection())

            keys = img.GetMetaDataKeys()
            for key in keys:
                img_array.SetMetaData(key, img.GetMetaData(key))
            
            print("Saving to  {}".format(os.path.join(train_label_dir, save_filename)))
            sitk.WriteImage(img_array, os.path.join(train_label_dir, save_filename))
            print("Reading image again...")
        
            img_array = sitk.ReadImage(os.path.join(train_label_dir, save_filename))
            print("Orienting image...")
            img_array = sitk.DICOMOrient(img_array, 'RAS')
            print("Saving image again...")
            sitk.WriteImage(img_array, os.path.join(train_label_dir, save_filename))

            assert (img_array.GetArrayFromImage().shape == img.GetArrayFromImage().shape)
            # Save the csv file for each iteration in case of error
            save_csv(os.path.join(task_folder_name, f"equivalence_{MODE}.csv"), equivalence_l)
        except:
            print("Error reading {}".format(train_images[i]))
            continue
    

print(f"Finished converting {MODE} to NII")