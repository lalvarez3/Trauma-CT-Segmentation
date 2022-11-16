""" Utilities for the project. Includes postprocessing functions and other for save_gifs.py and run_metrics.py"""

import glob
import imageio
import csv
import os
from collections import Counter, OrderedDict
from typing import (Any, Dict, Hashable, List, Mapping, Optional, Sequence,
                    Tuple, Union)

import numpy as np
import SimpleITK as sitk
import skimage.measure as measure
import torch
from batchgenerators.utilities.file_and_folder_operations import join, subfiles
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data import MetaTensor
from monai.transforms import (AddChanneld, AsChannelFirstd, Compose,
                              CropForegroundd, KeepLargestConnectedComponentd,
                              LoadImaged)
from monai.transforms.transform import MapTransform, Transform
from monai.utils import TransformBackends
from monai.utils.type_conversion import convert_to_dst_type
from nnunet.evaluation.evaluator import Evaluator, aggregate_scores
from scipy import ndimage
from skimage.morphology import (ball, binary_dilation, closing, cube, dilation,
                                disk)
from skimage.segmentation import (expand_labels, inverse_gaussian_gradient,
                                  morphological_geodesic_active_contour)


class injury_postprocessing(MapTransform):
    
    """ 
    Postprocessing of injuries for the predicted segmentation.

    """

    def __init__(
        self,
        keys: KeysCollection,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
        settings: dict = {
            "iterations": 2,
            "smoothing": 2,
            "balloon": 0,
            "threshold": "auto",
            "sigma": 2,
            "alpha": 7000,
        },
        organ: str = "Liver",
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.settings = settings
        self.organ = organ

    def get_connected_components(self, init_label, selected_label, min_size=4000):
        """
        Get the connected components of the selected label and remove the components that are smaller than the min_size.

        Args:
            init_label (np.ndarray): The initial label.
            selected_label (int): The selected label to extract the components.
            min_size (int): The minimum size of the connected components.
        """
        result = {} # Dictionary to store the connected components
        removed = {} # Dictionary to store the removed connected components
        init_label_ = init_label.copy() # Copy the initial label
        foreground = np.where((init_label_ != selected_label), 0, init_label_) # Get the foreground (selected label)
        labelling, label_count = measure.label( # Get the connected components
            foreground == selected_label, return_num=True
        )
        init_clusters = np.unique(labelling, return_counts=True) # Get the connected components and their sizes
        counter = 0
        for n in range(1, label_count + 1): # Loop over the connected components
            cluster_size = ndimage.sum(labelling == n) # Get the size of the connected component
            if cluster_size < min_size: # If the size is smaller than the min_size
                labelling[labelling == n] = 0 # Remove the connected component
                removed[n] = cluster_size
                counter += 1
            else: # If the size is larger than the min_size
                result[n] = cluster_size # Store the connected component
        for n in range(1, label_count + 1): # Loop over the connected components
            if n in result.keys(): # If the connected component is not removed
                labelling[labelling == n] = selected_label # Set the connected component to the selected label
        if counter == label_count: # If all the connected components are removed
            labelling = init_label # Set the initial label

        return labelling, init_clusters, result, removed

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        if self.organ == "Liver":
            fixed_settings = {
                "min_injury_size": 7000, # Minimum size of the injury
                "small_injury_size": 25000, # Maximum size of the small injury (for morphological operations)
                "cube_size": 2, # Size of the cube for the dilation
                "expanding_size": 2, # Size of the expanding for the dilation
            }
        elif self.organ == "Spleen":
            fixed_settings = {
                "min_injury_size": 500, # Minimum size of the injury
                "small_injury_size": 5000, # Maximum size of the small injury (for morphological operations)
                "cube_size": 8, # Size of the cube for the dilation
                "expanding_size": 3,  # Size of the expanding for the dilation
            }
        else:
            print("Select correct organ, options: Liver, Spleen")

        original_label = d["label"].squeeze() # Get the original label
        old_old_mask_organ = np.where((original_label != 1), 0, original_label) # Get the organ
        cropped_injury = CropForegroundd(keys=["image", "label"], source_key="label")(d) # Crop the injury
        init_label = cropped_injury[
            "label"
        ].squeeze() 
        old_old_mask_injury = np.where(
            (init_label != 2), 0, init_label
        )  # save only the injury
        old_mask_injury, init_clusters, result, removed = self.get_connected_components( # Get the connected components of the injury
            init_label=old_old_mask_injury.astype(np.int8),
            selected_label=2,
            min_size=fixed_settings["min_injury_size"],
        ) 
        scanner = cropped_injury[
            "image"
        ].squeeze()
        gimage = inverse_gaussian_gradient(
            scanner, sigma=self.settings["sigma"], alpha=self.settings["alpha"]
        )  # Get the inverse gaussian gradient

        ls = old_mask_injury 
        injury_size = np.sum(old_mask_injury) / 2 # Get the size of the injury
        size = 0

        if injury_size < fixed_settings["small_injury_size"]:  # If the size of the injury is smaller than the small_injury_size
            dilation_bool = True # Set the dilation_bool to True
            footprint = cube(fixed_settings["cube_size"]) # Set the footprint for the dilation
            size = fixed_settings["expanding_size"]  # Set the expanding size for the dilation
        else: # If the size of the injury is larger than the small_injury_size
            dilation_bool = True # Set the dilation_bool to True
            footprint = ball(1)  # Set the footprint for the dilation
            size = 1 # Set the expanding size for the dilation

        if size != 0: # If the expanding size is not 0
            ls = expand_labels(ls, size)  # Expand the injury
        ls = closing(ls) # Close the injury
        ls = morphological_geodesic_active_contour( # Get the active contour
            gimage, 
            num_iter=self.settings["iterations"],
            init_level_set=ls,
            smoothing=self.settings["smoothing"],
            balloon=self.settings["balloon"],
            threshold=self.settings["threshold"],
        )  
        if dilation_bool: # If the dilation_bool is True
            ls = dilation(ls, footprint) # Dilate the active contour

        cropped_injury["label"] = MetaTensor( # Set the label to the cropped injury
            torch.tensor(np.expand_dims(ls, 0)),
            meta=cropped_injury["label"].meta,
            applied_operations=cropped_injury["label"].applied_operations,
        )
        inv_cropped = CropForegroundd( # Inverse the cropping
            keys=["image", "label"], source_key="label"
        ).inverse(
            cropped_injury
        )
        final_mask_injury = torch.where(((inv_cropped["label"][0]) == 1), 2, 0) # Get the final mask of the injury
        final_mask = old_old_mask_organ + final_mask_injury # Get the final mask of the organ and the injury
        final_mask = np.where((final_mask == 3), 2, final_mask) # Adjust the label numbers of the classes
        d["label"] = np.expand_dims(final_mask, 0) # Set the label to the data

        return d


def save_csv(output_path, task_name, data, out_folder, gif_folder):

    base_path = os.path.join(
        "U:\\",
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        out_folder,
        gif_folder,
        output_path)

    keys = data[0].keys()
    a_file = open(base_path, "w+")
    dict_writer = csv.DictWriter(a_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(data)
    a_file.close()


def save_gif(volume, filename, task_name, out_folder, gif_folder):
    volume = volume.astype(np.float64) / np.max(volume)  # normalize the data to 0 - 1
    volume = volume * 255  # Now scale by 255
    volume = volume.astype(np.uint8)
    base_path = os.path.join(
        "U:\\",
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        out_folder,
        gif_folder)
    path_to_gif = os.path.join(base_path, f"{filename}.mp4")
    if not os.path.exists(base_path):
        print("Creating gifs directory")
        os.mkdir(base_path)
    imageio.mimsave(path_to_gif, volume, fps=5)
    return path_to_gif

