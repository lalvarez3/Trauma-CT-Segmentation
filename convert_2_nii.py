import glob
import SimpleITK as sitk
import os
import numpy as np
import shutil


task_name = 'Task501_LiverTrauma' #change here for different task name

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
        original_z_size = mha_img.shape[0]

        # Load the overlay
        overlay_data = sitk.ReadImage(overlay_path)
        overlay_org = overlay_data.GetOrigin()[-1]

        overlay_init = np.abs(1/mha_data.GetSpacing()[-1]*(mha_org-overlay_org) )

        lower_bound = int(overlay_init)
        upper_bound = label.shape[-1]
        zeros_up = lower_bound
        zeros_down = original_z_size - (upper_bound + lower_bound)
        new = list()

        if zeros_up > 0:
            new.append(np.zeros((label.shape[0], label.shape[1], zeros_up), dtype=label.dtype))

        new.append(label)

        if zeros_down > 0:
            new.append(np.zeros((label.shape[0], label.shape[1], zeros_down), dtype=label.dtype))

        label = np.concatenate(new, axis=2)

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
        new_labels = liver_channel
    else:
        # each channel is a different label, we want 0(liver = 1) and 2(liver injure = 2) and the background to be 0 
        channels = list()
        for i, channel in enumerate([0, 2]):  # TODO: desharcodead
               channels.append(mha_img[channel, :, :, :])
        # we got channels splitted with 1 each. if we add them we should get what we want
        new_labels =  new_labels + channels[0] + channels[1]

    return new_labels
                        

def save_csv(output_path, data):
    import csv
    keys = data[0].keys()
    a_file = open(output_path, "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    a_file.close()

scans_nii_path = os.path.join(
    "U:", "\lauraalvarez", "data", "liver", "train", "scans_nii"
)

overlays_nii_path = os.path.join(
    "U:", "\lauraalvarez", "data", "liver", "train", "overlays_nii"
)

scans_path = os.path.join(
    "U:", "\lauraalvarez", "data", "liver", "train", "scans"
)

overlays_path = os.path.join(
    "U:", "\lauraalvarez", "data", "liver", "train", "overlays"
)

train_images = sorted(
    glob.glob(
        os.path.join("U:", "\lauraalvarez", "data", "liver", "train", "scans", "*.mha")
    )
)
train_labels = sorted(
    glob.glob(
        os.path.join(
            "U:", "\lauraalvarez", "data", "liver", "train", "overlays", "*.mha"
        )
    )
)

BASE_PATH = os.path.join( "U:", "\lauraalvarez", "/nnunet/nnUNet_raw_data_base/nnUNet_raw_data")

task_folder_name = os.path.join(BASE_PATH, task_name)
train_image_dir = os.path.join(task_folder_name,'imagesTr')
train_label_dir = os.path.join(task_folder_name,'labelsTr')
test_dir = os.path.join(task_folder_name,'imagesTs')

make_if_dont_exist(task_folder_name,overwrite = False)
make_if_dont_exist(train_image_dir)
make_if_dont_exist(train_label_dir)
make_if_dont_exist(test_dir,overwrite= False)


equivalence_l = list()


for i in range(0, len(train_images)):
    save_filename = f"TRMLIV_00{i}_0000.nii.gz"
    equiv = {"mha": os.path.basename(train_images[i-1]).split(".")[0], "nii": save_filename}
    equivalence_l.append(equiv)
    print("Converting {}".format(train_images[i]))
    img = sitk.ReadImage(train_images[i])
    # img_array = sitk.GetArrayFromImage(img)
    # arr8 = img_array.astype(np.int8)
    # img8 = sitk.GetImageFromArray(arr8)
    print("Saving to  {}".format(os.path.join(train_image_dir, save_filename)))
    sitk.WriteImage(img, os.path.join(train_image_dir, save_filename))
    print(f"Converting mask for {train_labels[i]}")
    img = sitk.ReadImage(train_labels[i])
    # Adapt the channel
    img_array = one_channel_overlay(img)
    # Transform the label to the same size as the image
    img_array = adapt_overlay(train_labels[i], train_images[i], img_array)
    img_array = sitk.GetImageFromArray(img_array)

    keys = img.GetMetaDataKeys()
    for key in keys:
        img_array.SetMetaData(key, img.GetMetaData(key))
    
    print("Saving to  {}".format(os.path.join(train_label_dir, save_filename)))
    sitk.WriteImage(img, os.path.join(train_label_dir, save_filename))

    # Save the csv file for each iteration in case of error
    save_csv(os.path.join(task_folder_name, "equivalence.csv"), equivalence_l)

print("Finished converting Train to NII")