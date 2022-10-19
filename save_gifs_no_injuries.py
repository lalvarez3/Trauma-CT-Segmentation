import os
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ToTensord,
    Resized,
    AsChannelLastd,
    AsChannelFirstd,
    AsDiscrete,
    CropForeground,
    SpatialCropd,
    AsDiscreted,
    ScaleIntensityRanged,
    EnsureType,
    KeepLargestConnectedComponent,
    KeepLargestConnectedComponentd,
    LabelToContour,
    FillHolesd
)
import glob
from monai.transforms.transform import MapTransform
from monai.transforms.inverse import InvertibleTransform
from monai.data import decollate_batch
import SimpleITK as sitk
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from monai.visualize import matshow3d, blend_images
import torch
from monai.metrics import DiceMetric
import csv
import cv2
# import cc3d
# import morphsnakes as ms
import cv2
import imageio
from collections import Counter
from skimage.morphology import disk, dilation, binary_dilation, ball

import cv2
import imageio
from collections import Counter
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient, expand_labels,)
from skimage.morphology import disk, dilation, binary_dilation, ball, cube, closing
from scipy import ndimage
from monai.data import MetaTensor
import  skimage.measure as measure

class RefineOutput(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        old_mask_organ = np.where((d["label"] != 1), 0, d["label"])
        kernel = np.ones((2, 2), np.uint8)
        old_mask_organ =  np.expand_dims(np.stack([cv2.dilate(old_mask_organ[0,slice,:,:],kernel,iterations = 1) for slice in range(old_mask_organ.shape[1])]),0)
        old_mask_injury = np.where((d["label"] != 2), 0, d["label"]) 
        new_mask_injury = np.zeros_like(old_mask_injury)
        new_img = d["image"][:, :, :, :].copy()
        idx_label_organ = np.where(old_mask_organ.flatten() == 1)[0] #ids of spleen
        min_intensity = np.min(new_img[old_mask_injury!=0]) 
        idx_img = np.where((new_img.flatten() > min_intensity))[0]
        idx_img_2 = np.where((new_img.flatten() < min_intensity +30))[0]
        idx_img = np.intersect1d(idx_img, idx_img_2)
        idx_to_change = np.intersect1d(idx_img, idx_label_organ)
        np.put(new_mask_injury, idx_to_change, 1)
        old_mask_injury += new_mask_injury
        old_mask_injury = np.where((old_mask_injury == 3), 2, old_mask_injury) 
        old_mask_injury = np.where((old_mask_injury == 1), 2, old_mask_injury) 

        # closed_slices = list()
        # for slice in range(new_mask.shape[-1]):
        #     result = cv2.morphologyEx(
        #         new_mask[0, :, :, slice], cv2.MORPH_CLOSE, kernel, iterations=2
        #     )
        #     result = cv2.medianBlur(result, 3)
        #     closed_slices.append(result)

        # new_mask = np.stack(closed_slices)

        final_mask = old_mask_injury + old_mask_organ

        d["label"] = final_mask


        return d


class DilationLabel(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        img = d["label"]
        old_mask_injury = np.where((img != 2), 0, img)
        if ORGAN == "Spleen":
            injury_size = np.sum(old_mask_injury)/2
            if injury_size < 4500:
                # radius = int((injury_size/1000)*2)
                old_mask_organ = np.where((img != 1), 0, img)
                final_mask_injury = dilation(old_mask_injury[0,:,:,:], footprint=ball(radius=6))
                final_mask_injury = np.expand_dims(final_mask_injury, 0).astype(np.int8)
                final_mask = old_mask_organ + final_mask_injury
                final_mask = np.where((final_mask == 3), 2, final_mask)
                d["label"] = final_mask
            else:
                old_mask_organ = np.where((img != 1), 0, img)
                final_mask_injury = dilation(old_mask_injury[0,:,:,:], footprint=ball(radius=2))
                final_mask_injury = np.expand_dims(final_mask_injury, 0).astype(np.int8)
                final_mask = old_mask_organ + final_mask_injury
                final_mask = np.where((final_mask == 3), 2, final_mask)
                d["label"] = final_mask
        if ORGAN == "Liver":
            old_mask_organ = np.where((img != 1), 0, img)
            mask = disk(2)
            new_mask_injury = list()
            for slice in range(old_mask_injury.shape[1]):
                result = dilation(old_mask_injury[0,slice,:,:], footprint=mask)
                new_mask_injury.append(result)
            final_mask_injury = np.stack(new_mask_injury)
            final_mask_injury = np.expand_dims(final_mask_injury, 0).astype(np.int8)
            final_mask = old_mask_organ + final_mask_injury
            final_mask = np.where((final_mask == 3), 2, final_mask)
            d["label"] = final_mask

        return d


def fill_contours_fixed(arr):
    slices = []
    for _ in range(arr.shape[0]):
        slices.append(
        np.maximum.accumulate(arr, 1) &\
            np.maximum.accumulate(arr[:, :, ::-1], 1)[:, :, ::-1] &\
            np.maximum.accumulate(arr[:, ::-1, :], 0)[:,::-1, :] &\
            np.maximum.accumulate(arr[::-1, :, :], 0)[::-1, :, :] &\
            np.maximum.accumulate(arr, 0))
    return np.stack(slices, 0)



class ActiveContour(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        label = d["label"][:,:,:,:]
        img = d["image"][:,:,:,:]
        old_mask_organ = np.where((label != 1), 0, label)
        old_mask_injury = np.where((label != 2), 0, label)
        # old_mask_injury = np.expand_dims(old_mask_injury,0)
        temp_d = {"image": d["image"], "label": old_mask_injury}
        cropped_contour = CropForegroundd(keys=["image", "label"], 
                                            source_key="label",
                                            margin=10)(temp_d)
        cropped_img = cropped_contour["image"][0,:,:,:]
        cropped_label_injury = cropped_contour["label"][0,:,:,:]
        # gimg = ms.inverse_gaussian_gradient(cropped_img, alpha=1000, sigma=5.48)
        # contour_injury = LabelToContour()(cropped_label_injury)
        # label_ac = ms.morphological_geodesic_active_contour(gimg, iterations=10,
        #                                      init_level_set=cropped_label_injury,
        #                                      smoothing=1, threshold=0.31,
        #                                      balloon=1)
        label_ac = ms.morphological_chan_vese(cropped_img, 25, init_level_set=cropped_label_injury, lambda2=2)
        label_ac = np.where((label_ac == 1), 2, label_ac)
        cropped_contour["label"] = np.expand_dims(label_ac, 0)
        inv_cropped = CropForegroundd(keys=["image", "label"], source_key="label",
                                            margin=30).inverse(cropped_contour)
        label_ac = inv_cropped["label"]
        final_mask_injury = label_ac.astype(np.int8)
        final_mask = old_mask_organ + final_mask_injury
        final_mask = np.where((final_mask == 3), 2, final_mask)
        d["label"] = final_mask

        return d

class injury_postprocessing(MapTransform):


    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
        settings: dict = {"iterations": 2, "smoothing": 2, "balloon": 0, "threshold": 'auto', 'sigma':2, 'alpha': 7000},
        organ: str = "Liver",
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.settings = settings
        self.organ = organ



    def get_connected_components(self, init_label, selected_label, min_size=4000):
        result = {}
        removed = {}
        init_label_ = init_label.copy()
        foreground = np.where((init_label_ != selected_label), 0, init_label_)
        labelling, label_count = measure.label(foreground == selected_label, return_num=True)
        init_clusters = np.unique(labelling, return_counts=True)
        counter = 0
        for n in range(1, label_count+1):
            cluster_size = ndimage.sum(labelling ==n)
            if cluster_size < min_size:
                labelling[labelling == n] = 0
                removed[n] = cluster_size
                counter +=1
            else:
                result[n] = cluster_size
        for n in range(1, label_count+1):
            if n in result.keys():
                labelling[labelling == n] = 2
        if counter == label_count:
            print('empty')
        else:
            print('not empty')
            # labelling = init_label
        
        return labelling, init_clusters, result, removed

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        if self.organ == 'Liver': fixed_settings = {'min_injury_size': 7000, 'small_injury_size': 25000, 'cube_size': 2, 'expanding_size' : 2}
        elif self.organ == 'Spleen': fixed_settings = {'min_injury_size': 500, 'small_injury_size': 5000, 'cube_size': 8, 'expanding_size' : 3}
        else: print("Select correct organ, options: Liver, Spleen")

        original_label = d["label"].squeeze() 
        if len(np.unique(original_label)) <= 2:
            return d
        old_old_mask_organ = np.where((original_label != 1), 0, original_label)
        cropped_injury = CropForegroundd(keys=["image", "label"],  source_key="label",  margin=5)(d)
        init_label = cropped_injury["label"].squeeze() 
        old_old_mask_injury = np.where((init_label != 2), 0, init_label) # save only the injury
        old_mask_injury, init_clusters, result, removed = self.get_connected_components(init_label=old_old_mask_injury.astype(np.int8), selected_label=2, min_size=fixed_settings['min_injury_size']) #liver 7000
        scanner = cropped_injury["image"].squeeze()
        gimage = inverse_gaussian_gradient(scanner, sigma=self.settings['sigma'], alpha=self.settings['alpha']) #init sigma 3 a=100

        ls = old_mask_injury
        injury_size = np.sum(old_mask_injury)/2
        size = 0

        if injury_size < fixed_settings['small_injury_size']: #spleen 5000
            dilation_bool= True
            footprint = cube(fixed_settings['cube_size']) #spleen 8
            size = fixed_settings['expanding_size'] # 3 spleen
        else: 
            dilation_bool= True
            footprint = ball(1) # 1 para spleen 
            size = 1

        if size != 0: ls = expand_labels(ls, size) #footprint=cube(2)) expand_labels(ls, 2)
        ls = closing(ls)
        ls = morphological_geodesic_active_contour(gimage, num_iter=self.settings['iterations'],  init_level_set=ls, smoothing=self.settings['smoothing'], balloon=self.settings['balloon'],  threshold=self.settings['threshold']) #morphological_chan_vese
        if dilation_bool: ls = dilation(ls, footprint) 

        cropped_injury["label"] = MetaTensor(torch.tensor(np.expand_dims(ls,0)), meta=cropped_injury["label"].meta, applied_operations = cropped_injury["label"].applied_operations )
        inv_cropped = CropForegroundd(keys=["image", "label"], source_key="label",  margin=30).inverse(cropped_injury)
        final_mask_injury = torch.where(((inv_cropped["label"][0]) == 1), 2, 0)
        final_mask = old_old_mask_organ + final_mask_injury
        final_mask = np.where((final_mask == 3), 2, final_mask)

        d["label"] =  np.expand_dims(final_mask,0)

        return d

OUT_FOLDER = "out_no_injuries"
GIF_FOLDER = "gifs_no_injuries_2"
ORGAN = 'Spleen'#"Liver"
TASK =  'Task511_SpleenTraumaCV' #'Task510_LiverTraumaDGX'

def save_csv(output_path, task_name, data):
    import csv

    base_path = os.path.join(
        "U:\\",
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        OUT_FOLDER,
        GIF_FOLDER,
        output_path)

    keys = data[0].keys()
    a_file = open(base_path, "w+")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    a_file.close()


def _save_gif(volume, filename, task_name="Task504_LiverTrauma"):
    volume = volume.astype(np.float64) / np.max(volume)  # normalize the data to 0 - 1
    volume = volume * 255  # Now scale by 255
    volume = volume.astype(np.uint8)
    base_path = os.path.join(
        "U:\\",
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        OUT_FOLDER,
        GIF_FOLDER)
    path_to_gif = os.path.join(base_path, f"{filename}.mp4")
    if not os.path.exists(base_path):
        print("Creating gifs directory")
        os.mkdir(base_path)
    imageio.mimsave(path_to_gif, volume, fps=5)
    return path_to_gif


def create_figs(image_path, prediction_path, label_path, task_name=TASK): 
    #Task511_SpleenTraumaCV Task510_LiverTraumaDGX
    images = glob.glob(
        os.path.join(
            "U:\\",
            "lauraalvarez",
            "nnunet",
            "nnUNet_raw_data",
            "Task513_NIExtendedTrauma",
            "imagesTs",
            "*.nii.gz",
        )
    )
    predicitions = glob.glob(
        os.path.join(
            "U:\\",
            "lauraalvarez",
            "nnunet",
            "nnUNet_raw_data",
            task_name,
            OUT_FOLDER,
            "*.nii.gz",
        )
    )

    data_dicts_test = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images, predicitions)
    ]

# NOTE: POr alguna razon aqui uno de los scanneres me sale metric 0 cuando con el mismo codigo
# en el run_metrics me sale 0.67, habra que debugear, ignorando for now.
    csv_list = []
    for data in data_dicts_test[:]:
        print(f"Infering for \n\t image:{data['image']}, \n\t label: {data['label']}")
        normal_plot = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AsChannelFirstd(keys=["image", "label"]),
                AddChanneld(keys=["label", "image"]),
                CropForegroundd(keys=["image", "label"], source_key="image"),
#                 DilationLabel(keys=["label"]),
                # KeepLargestConnectedComponentd(keys=["label"], applied_labels=[1,2], is_onehot=False, independent=True),
                # ActiveContour(keys=["label", "image"]),
                # FillHolesd(keys=["label"]),
                
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                KeepLargestConnectedComponentd(
                    keys=["label"],
                    applied_labels=[1, 2],
                    is_onehot=False,
                    independent=False,
                    connectivity=None,
                ),
                injury_postprocessing(keys=["image", "label"],  organ=ORGAN, settings = {"iterations": 2, "smoothing": 2, "balloon": 0, "threshold": 'auto', 'sigma':2, 'alpha': 7000}),
                
            ]
        )
        basename = os.path.basename(data["image"])
        injuries = normal_plot(data)
        predicted_labels = np.unique(injuries["label"])

        dict_data = {
            "image": basename,
            "predicted_labels": predicted_labels,
        }
        csv_list.append(dict_data)
        save_gif = True
        if save_gif == True:
            post_plotting = Compose([EnsureType(), AsDiscrete(argmax=False)])
            injuries["label"] = post_plotting(injuries["label"])
            inj = dict(injuries)
            inj = Resized(keys=["image", "label"], spatial_size=(512, 512, 512))(
                inj
            )
            # injures["label"] = np.expand_dims(injures["label"],0)
            # injures["image"] = np.expand_dims(injures["image"],0)

            blended_label_in = blend_images(inj["image"], inj["label"], 0.5)
            blended_final = blended_label_in.permute(1, 2, 0, 3)

            volume = torch.hstack(
                (
                    inj["image"].permute(1, 2, 0, 3).repeat(1, 1, 3, 1),
                    blended_final,
                )
            )
            volume = volume.permute(0, 1, 3, 2)

            volume_path = _save_gif(volume.numpy(), f"{basename}", task_name)
            # _save_gif(blended_true_label.numpy().transpose(0, 1, 3, 2), f"{basename}_True", task_name)
            # _save_gif(blended_final.numpy().transpose(0, 1, 3, 2), f"{basename}_Pred", task_name)

            print(f"Saved {volume_path}")
        save_csv("summary.csv",task_name, csv_list)


def main():
    create_figs("", "", "")  # TODO Desharcodear


if __name__ == "__main__":
    main()
