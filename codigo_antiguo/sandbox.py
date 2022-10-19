import SimpleITK as sitk
import os
import numpy as np

import  skimage.measure as measure
# from monai.transforms.utils import (
#     remove_small_objects,
# )
HOME = "/mnt/chansey"

prediction_p = os.path.join(
    HOME,
    "lauraalvarez",
    "nnunet",
    "nnUNet_raw_data",
    "Task510_LiverTraumaDGX",
    "TLIV_007_out.nii.gz",
)
label_p = os.path.join(
    HOME,
    "lauraalvarez",
    "nnunet",
    "nnUNet_raw_data",
    "Task510_LiverTraumaDGX",
    "TLIV_007.nii.gz",
)
# label_p = os.path.join(
#     HOME,
#     "lauraalvarez",
#     "nnunet",
#     "nnUNet_raw_data",
#     "Task510_LiverTraumaDGX/train-removed/",
#     "TLIV_003.nii.gz",
# )
image_p = os.path.join(
    HOME,
    "lauraalvarez",
    "nnunet",
    "nnUNet_raw_data",
    "Task510_LiverTraumaDGX",
    "TLIV_007_0000.nii.gz",
)

other_label = os.path.join(
    HOME,
    "lauraalvarez",
    "nnunet",
    "nnUNet_raw_data",
    "Task512_LiverSpleenTrauma/imagesTs/"
    "TSpLi_008.nii.gz",
)

prev = os.path.join(
    HOME,
    "lauraalvarez",
    "data",
    "_overlays_from_alessa",
    "overlay_june9_v2/overlay/"
    "L110162.mha",
)

folder = os.path.join(
    HOME,
    "lauraalvarez",
    "nnunet",
    "nnUNet_raw_data",
    "Task510_LiverTraumaDGX")


def adapt_overlay(overlay_path, mha_data, label):
    # Load the mha
    mha_org = mha_data.GetOrigin()[-1]
    # Load the mha image
    mha_img = sitk.GetArrayFromImage(mha_data)

    if mha_img.shape == label.shape:
        return label

    original_z_size = mha_img.shape[0]

    # Load the overlay
    overlay_data = sitk.ReadImage(overlay_path)
    overlay_org = overlay_data.GetOrigin()[-1]

    overlay_init = np.abs(1 / mha_data.GetSpacing()[-1] * (mha_org - overlay_org))

    lower_bound = int(overlay_init)
    upper_bound = label.shape[0]
    zeros_up = lower_bound
    zeros_down = original_z_size - (upper_bound + lower_bound)
    new = list()

    if zeros_up > 0:
        new.append(
            np.zeros((zeros_up, label.shape[1], label.shape[2]), dtype=label.dtype)
        )

    new.append(label)

    if zeros_down > 0:
        new.append(
            np.zeros((zeros_down, label.shape[1], label.shape[2]), dtype=label.dtype)
        )

    label = np.concatenate(new, axis=0)

    return label

def channel_first(img_mask, img):
    img = sitk.GetArrayFromImage(img)
    if img.shape[0] != img_mask.shape[0]:
        if img.shape[0] == img_mask.shape[-1] and img.shape[-1] == img_mask.shape[0]:
            img_mask = img_mask.transpose((2, 1, 0))
    return img_mask

def match_channels(img_mask, img):
    sizes = img.GetSize()
    shapes = img_mask.shape

    ind_1 = sizes.index(shapes[0])
    ind_2 = sizes.index(shapes[1])
    ind_3 = sizes.index(shapes[2])

    img_mask = img_mask.transpose((ind_1, ind_2, ind_3))
    img_mask = img_mask.transpose((2,1,0))

    return img_mask


def one_channel_overlay(img, organ):
    mha_img = sitk.GetArrayFromImage(img)
    mha_img = mha_img.astype(np.int8)

    if len(mha_img.shape) == 3:
        z, h, w = mha_img.shape
        c = 1
    else:
        c, z, h, w = mha_img.shape

    new_labels = np.zeros((z, h, w), np.int8)

    # Depending on the number of channels, we have to adapt the overlay (GC has 1 working, the other has 6 channels)
    # We want 0 and 2 for channels and
    if c == 1:
        if organ == "liver":
            # we need to transform Liver (6) to 1
            liver_channel = np.where((mha_img != 6), 0, mha_img)
            liver_channel = np.where((liver_channel == 6), 1, liver_channel)
            new_labels = liver_channel
        elif organ== "spleen":
            spleen_channel = np.where((mha_img != 1), 0, mha_img)
            spleen_channel = np.where((spleen_channel == 1), 1, spleen_channel)
            new_labels = spleen_channel
        else:
            liver_channel = np.where((mha_img != 6), 0, mha_img)
            liver_channel = np.where((liver_channel == 6), 1, liver_channel)
            spleen_channel = np.where((mha_img != 1), 0, mha_img)
            spleen_channel = np.where((spleen_channel == 1), 1, spleen_channel)
            new_labels = liver_channel + spleen_channel
    else:
        # each channel is a different label, we want 0(liver = 1) and 2(liver injure = 2) and the background to be 0
        # 0:liver 1:spleen 2 :liver_injure 3:Spleen_injure 4:VLI 5:VSI
        if organ == "liver":
            labels = [0,2]
        elif organ == "spleen":
            labels = [1,3]
        else:
            labels = [0,1,2,3]
        channels = list()
        for i, channel in enumerate(labels):
            c = mha_img[channel, :, :, :]
            c = np.where((c == 1), i + 1, c)
            channels.append(c)

        # we got channels splitted with 1 each. if we add them we should get what we want
        chans = np.stack(channels, axis=0)
        new_labels = np.max(chans, axis=0)

    return new_labels

def fix_wrong_labels():
    """Esto se usa para cuando una label está mal generada. En particular la hemos usado para recrear
    la label de la imagen TSpLi_008.nii.gz que estaba mal generada. Para ello, se ha usado la imagen
    L110162.mha que es la que se ha usado para generar la label de la imagen TSpLi_008.nii.gz y de
    la TLIV_007_0000.nii.gz

    """
    # pred = sitk.ReadImage(prediction_p)

    # label = sitk.ReadImage(label_p)

    image = sitk.ReadImage(image_p)

    prev_o = sitk.ReadImage(prev)
    img_array = one_channel_overlay(prev_o, "liver")
    # img_array = match_channels(img_array, image)
    # img_array = adapt_overlay(prev, prev_o, img_array)
    img_array = img_array.transpose(1,2,0)
    img_array = np.flip(img_array, 0)
    img_array = np.flip(img_array, 2)
    img_array = sitk.GetImageFromArray(img_array)

    save_filename = "TLIV_007_1.nii.gz"
    print("Saving to  {}".format(os.path.join(folder,save_filename)))
    sitk.WriteImage(img_array, os.path.join(folder, save_filename))
    print("Reading image again...")

    img_array = sitk.ReadImage(os.path.join(folder, save_filename))
    print("Orienting image...")
    img_array = sitk.DICOMOrient(img_array, "RAS")
    # img_array.SetDirection(image.GetDirection())
    print(f"spacing: {img_array.GetSpacing()}")
    img_array.SetSpacing(image.GetSpacing())
    img_array.SetOrigin(image.GetOrigin())
    # print(f"img shape {prev_o.GetSize()}")
    # print(f"img spacing {prev_o.GetSpacing()}")
    # print(f"img origin {prev_o.GetOrigin()}")
    print(f"shape: {img_array.GetSize()}")
    print(f"spacing: {img_array.GetSpacing()}")
    print(f"origin: {img_array.GetOrigin()}")
    print("Saving image again...")
    sitk.WriteImage(img_array, os.path.join(folder, save_filename))

    # print(
    #     f"Size -> pred: {pred.GetSize()}, image: {image.GetSize()}, label: {label.GetSize()}"
    # )


# label = sitk.ReadImage(label_p)


# from scipy import ndimage

# def find_clusters(array):
#     """Find clusters in a 3D array"""

#     clustered = np.empty_like(array)
#     unique_vals = np.unique(array)
#     cluster_count = 0
#     for val in unique_vals:
#         labelling = measure.label(array == val)
#         label_count = np.max(labelling)
#         for k in range(1, label_count + 1):
#             clustered[labelling == k] = cluster_count
#             cluster_count += 1
#     return clustered, cluster_count - 1

# def get_connected_components(img, selected_label):
#     """Get connected components from any image.
#     For injuries is better to use it in a single channel or one_hot image

#     Args:
#         img (np.ndarray): labeled image

#     Returns:
#         np.ndarray: labeled image
#     """
#     result = {}
#     img_ = img.copy()
#     foreground = np.where((img_ != selected_label), 0, img_)
#     cluster, n_clusters = find_clusters(foreground)
#     labeled = ndimage.find_objects(cluster)
#     for n in range(n_clusters):
#         result[n] = ndimage.sum(img[labeled[n]])
#     return result

# # ndimage.sum(img[labeled[1]])
# injury_sizes = get_connected_components(np.expand_dims(sitk.GetArrayFromImage(label), 0),2)

# print(f"Number of injuries: {len(injury_sizes)}")
# print(f"Size of injuries: {injury_sizes}")
fix_wrong_labels()
print("finish")
