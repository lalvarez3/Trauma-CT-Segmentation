import glob
import SimpleITK as sitk
import os
import numpy as np
import shutil
from tqdm import tqdm
import nibabel as nib

from monai.transforms import (
    Compose,
    Orientation,
    AddChannel
)

def check_orientation(ct_image):
    """
    Check the NIfTI orientation, and flip to  'RPS' if needed.
    :param ct_image: NIfTI file
    :param ct_arr: array file
    :return: array after flipping
    """
    x, y, z = nib.aff2axcodes(ct_image.affine)
    ct_arr = ct_image.get_fdata()
    if x != 'R':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=0)
    if y != 'P':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=1)
    if z != 'S':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=2)
    new_nifti = nib.Nifti1Image(ct_arr.astype(np.float), ct_image.affine)
    return new_nifti

task_name = 'Task503_LiverSpleenTrauma' #change here for different task name

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
                        

def save_csv(output_path, data):
    import csv
    keys = data[0].keys()
    a_file = open(output_path, "w")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    a_file.close()

BASE_PATH = os.path.join( "/mnt/chansey", "lauraalvarez", "nnunet", "nnUNet_raw_data_base", "nnUNet_raw_data")
task_folder_name = os.path.join(BASE_PATH, task_name)


trainTr = os.path.join(task_folder_name,'imagesTr')

labelTr = os.path.join(task_folder_name,'labelsTr')

trainTs =os.path.join(task_folder_name,'imagesTs')

labelTs = os.path.join(task_folder_name,'labelsTs')

train_images = sorted(
    glob.glob(
        os.path.join(trainTr, "*.nii.gz")
    )
)
train_labels = sorted(
    glob.glob(
        os.path.join(
           labelTr, "*.nii.gz"
        )
    )
)

test_images = sorted(
    glob.glob(
        os.path.join(trainTs, "*.nii.gz")
    )
)
test_labels = sorted(
    glob.glob(
        os.path.join(
           labelTs, "*.nii.gz"
        )
    )
)

# transform = Compose(
#     [
#     AddChannel(), Orientation(axcodes="RAS"),
#     ]
# )


for i in tqdm(range(0, len(train_images))):
    
        print("reading origin and spacing from {}".format(train_images[i]))
        try:
            img = sitk.ReadImage(train_images[i])
            print("original information")
            print(img.GetOrigin())
            origs = img.GetOrigin()
            print(img.GetSpacing())
            spacing = img.GetSpacing()

            # img = sitk.DICOMOrient(img, 'RAS')

            # print(img.GetDirection())

            # sitk.WriteImage(img, train_images[i])

            print(f"Adapt spacing and origin for mask {train_labels[i]}")
            try:
                label_img = sitk.ReadImage(train_labels[i])
                print()
            except:
                print("Error reading {}".format(train_labels[i]))
                continue

            print("old_label_information")
            print(label_img.GetOrigin())
            print(label_img.GetSpacing())


            
            label_img.SetSpacing(img.GetSpacing())
            label_img.SetOrigin(img.GetOrigin())
            
            # label_img = sitk.DICOMOrient(label_img, 'RAS')

            print("new_label_information")
            print(label_img.GetOrigin())
            print(label_img.GetSpacing())
            print(label_img.GetDirection())
            
            print("Saving to  {}".format((train_labels[i])))
            sitk.WriteImage(label_img,(train_labels[i]))

        except:
            print("Error reading {}".format(train_images[i]))
            continue



for i in tqdm(range(0, len(test_images))):

    print("reading origin and spacing from {}".format(test_images[i]))
    # try:
    img = sitk.ReadImage(test_images[i])
    print("original information")
    print(img.GetOrigin())
    origs = img.GetOrigin()
    print(img.GetSpacing())
    spacing = img.GetSpacing()

    # img = sitk.DICOMOrient(img, 'RAS')

    # print(img.GetDirection())

    # sitk.WriteImage(img, test_images[i])
        

    print(f"Adapt spacing and origin for mask {test_labels[i]}")
        # try:
    label_img = sitk.ReadImage(test_labels[i])
    print()
        # except:
        #     print("Error reading {}".format(test_labels[i]))
        #     continue

    print("old_label_information")
    print(label_img.GetOrigin())
    print(label_img.GetSpacing())
    
    label_img.SetSpacing(img.GetSpacing())
    label_img.SetOrigin(img.GetOrigin())

    # label_img = sitk.DICOMOrient(label_img, 'RAS')

    print("new_label_information")
    print(label_img.GetOrigin())
    print(label_img.GetSpacing())
    print(label_img.GetDirection())
    
    print("Saving to  {}".format((test_labels[i])))
    sitk.WriteImage(label_img,(test_labels[i]))
    # except:
    #     print("Error reading {}".format(test_images[i]))
    #     continue
    

print(f"Finished converting adapting the space and origin to NII")