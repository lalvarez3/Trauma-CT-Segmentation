import os
import SimpleITK as sitk
from monai.transforms import (
    Orientation,
)
from nnunet.evaluation.evaluator import Evaluator, aggregate_scores
import numpy as np
from collections import Counter
from skimage.morphology import disk, dilation, binary_dilation, ball
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import MapTransform
from monai.config.type_definitions import NdarrayOrTensor
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.transform import Transform
from monai.utils import (
    TransformBackends,
)
from monai.utils.type_conversion import convert_to_dst_type
from collections import OrderedDict

from skimage.segmentation import (
    morphological_geodesic_active_contour,
    inverse_gaussian_gradient,
)

import os
from monai.transforms import (
    Compose,
    CropForegroundd,
    AsDiscrete,
    KeepLargestConnectedComponentd,
    LoadImaged, 
    LoadImage,
    AsChannelFirstd, AddChanneld,
)
import glob
from monai.transforms.transform import MapTransform
from monai.transforms.inverse import InvertibleTransform
import SimpleITK as sitk
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch
# import cc3d
# import morphsnakes as ms
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

from batchgenerators.utilities.file_and_folder_operations import  subfiles, join

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
            print(cluster_size)
            if cluster_size < min_size:
                labelling[labelling == n] = 0
                removed[n] = cluster_size
                counter +=1
            else:
                result[n] = cluster_size
        for n in range(1, label_count+1):
            if n in result.keys():
                labelling[labelling == n] = selected_label
        if counter == label_count:
            labelling = init_label

        return labelling, init_clusters, result, removed

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        if self.organ == 'Liver': fixed_settings = {'min_injury_size': 7000, 'small_injury_size': 25000, 'cube_size': 2, 'expanding_size' : 2}
        elif self.organ == 'Spleen': fixed_settings = {'min_injury_size': 500, 'small_injury_size': 5000, 'cube_size': 8, 'expanding_size' : 3}
        else: print("Select correct organ, options: Liver, Spleen")

        original_label = d["label"].squeeze()
        old_old_mask_organ = np.where((original_label != 1), 0, original_label)
        cropped_injury = CropForegroundd(keys=["image", "label"],  source_key="label")(d)
        init_label = cropped_injury["label"].squeeze()#d["label"]#cropped_injury["label"].squeeze()
        old_old_mask_injury = np.where((init_label != 2), 0, init_label) # save only the injury
        old_mask_injury, init_clusters, result, removed = self.get_connected_components(init_label=old_old_mask_injury.astype(np.int8), selected_label=2, min_size=fixed_settings['min_injury_size']) #liver 7000
        scanner = cropped_injury["image"].squeeze()#d["image"]#cropped_injury["image"].squeeze()
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
        inv_cropped = CropForegroundd(keys=["image", "label"], source_key="label").inverse(cropped_injury) #{"label": np.expand_dims(ls,0)} # 
        final_mask_injury = torch.where(((inv_cropped["label"][0]) == 1), 2, 0)
        final_mask = old_old_mask_organ + final_mask_injury
        final_mask = np.where((final_mask == 3), 2, final_mask)

        d["label"] =  np.expand_dims(final_mask,0)

        return d

class DilationLabel(Transform):

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        old_mask_injury = np.where((img != 2), 0, img)

        if ORGAN == "Spleen":
            injury_size = np.sum(old_mask_injury) / 2
            if injury_size < 4500:
                radius = int((injury_size / 1000) * 2)
                old_mask_organ = np.where((img != 1), 0, img)
                final_mask_injury = dilation(
                    old_mask_injury[0, :, :, :], footprint=ball(radius=6)
                )
                final_mask_injury = np.expand_dims(final_mask_injury, 0).astype(np.int8)
                final_mask = old_mask_organ + final_mask_injury
                final_mask = np.where((final_mask == 3), 2, final_mask)
                img = final_mask
            else:
                old_mask_organ = np.where((img != 1), 0, img)
                final_mask_injury = dilation(
                    old_mask_injury[0, :, :, :], footprint=ball(radius=2)
                )
                final_mask_injury = np.expand_dims(final_mask_injury, 0).astype(np.int8)
                final_mask = old_mask_organ + final_mask_injury
                final_mask = np.where((final_mask == 3), 2, final_mask)
                img = final_mask
        if ORGAN == "Liver":
            old_mask_organ = np.where((img != 1), 0, img)
            mask = disk(2)
            new_mask_injury = list()
            for slice in range(old_mask_injury.shape[1]):
                result = dilation(old_mask_injury[0, slice, :, :], footprint=mask)
                new_mask_injury.append(result)
            final_mask_injury = np.stack(new_mask_injury)
            final_mask_injury = np.expand_dims(final_mask_injury, 0).astype(np.int8)
            final_mask = old_mask_organ + final_mask_injury
            final_mask = np.where((final_mask == 3), 2, final_mask)
            img = final_mask
        return img


class NiftiEvaluatorv2(Evaluator):
    def __init__(self, *args, **kwargs):

        self.test_nifti = None
        self.reference_nifti = None
        super(NiftiEvaluatorv2, self).__init__(*args, **kwargs)

    def set_test(self, test):
        """Set the test segmentation."""
        if test is not None:
            self.test_path = test
            print("reading image", test)
            self.test_nifti = sitk.ReadImage(test)
            post_pro = True
            # if not post_pro: test_preprocess = sitk.GetArrayFromImage(self.test_nifti)
            # print("preprocessing size", test_preprocess.shape)
            post_transformation = Compose(
                [
                    LoadImaged(keys=["label", 'image']), #LoadImaged(keys=["image", "label"]),
                    AsChannelFirstd(keys=[ "label", 'image']), #AsChannelFirstd(keys=["image", "label"]),
                    AddChanneld(keys=["label", 'image']), #AddChanneld(keys=["image", "label"]),
                    # KeepLargestConnectedComponentd(
                    #     keys=["label"],
                    #     applied_labels=[1, 2],
                    #     is_onehot=False,
                    #     independent=False,
                    #     connectivity=None,
                    # ),
                    injury_postprocessing(keys=["image", "label"],  organ=ORGAN, settings = {"iterations": 2, "smoothing": 2, "balloon": 0, "threshold": 'auto', 'sigma':2, 'alpha': 7000}),
                ]
            )
            
            label_name = os.path.basename(test)
            if post_pro:
                image_name = label_name.split('.')[0] + '_0000' + '.' + label_name.split('.')[1] +  '.' + label_name.split('.')[2]  
                test_preprocess = [{"image":  test.replace(OUT_FOLDER, "imagesTs").replace(label_name, image_name), "label": test}]
            else: 
                test_preprocess = [{"label": test}]
            # print(test_preprocess)

            test_postprocess = post_transformation(
            test_preprocess #np.expand_dims(test_preprocess, 0)
            ) #.squeeze()
            # test_postprocess = test_preprocess[0]["label"].squeeze()
            test_postprocess = test_postprocess[0]["label"].squeeze()
            print("post processing size", test_postprocess.shape)
            super(NiftiEvaluatorv2, self).set_test(test_postprocess)
        else:
            self.test_nifti = None
            super(NiftiEvaluatorv2, self).set_test(test)

    def set_reference(self, reference):
        """Set the reference segmentation."""

        if reference is not None:
            self.reference_path = reference
            # self.reference_nifti = sitk.ReadImage(reference)
            # reference_img = sitk.GetArrayFromImage(self.reference_nifti)
            T = Compose(
                [
                    LoadImaged(keys=["label"]),
                    AsChannelFirstd(keys=["label"]),
                ]
             )
            reference_img = T([{"label": reference}])[0]["label"]
            # reference_img = Orientation(axcodes="RAS")(reference_img)
            print("reference size", reference_img.shape)
            super(NiftiEvaluatorv2, self).set_reference(reference_img)
        else:
            self.reference_nifti = None
            super(NiftiEvaluatorv2, self).set_reference(reference)

    def evaluate(self, test=None, reference=None, voxel_spacing=None, **metric_kwargs):

        if voxel_spacing is None:
            voxel_spacing = np.array(self.test_nifti.GetSpacing())[::-1]
            metric_kwargs["voxel_spacing"] = voxel_spacing
        try:

            a = super(NiftiEvaluatorv2, self).evaluate(test, reference, **metric_kwargs)

        except Exception as e:
            print("========ERROR========")
            print(e)
            print(f"path to test: {self.test_path}")
            print(f"path to reference: {self.reference_path}")
            print("=======================")
            a = {
                "1": OrderedDict(
                    [
                        ("Accuracy", "error"),
                        ("Dice", "error"),
                        ("False Discovery Rate", "error"),
                        ("False Negative Rate", "error"),
                        ("False Omission Rate", "error"),
                        ("False Positive Rate", "error"),
                        ("Jaccard", "error"),
                        ("Negative Predictive Value", "error"),
                        ("Precision", "error"),
                        ("Recall", "error"),
                        ("Total Positives Reference", "error"),
                        ("Total Positives Test", "error"),
                        ("True Negative Rate", "error"),
                    ]
                ),
                "2": OrderedDict(
                    [
                        ("Accuracy", "error"),
                        ("Dice", "error"),
                        ("False Discovery Rate", "error"),
                        ("False Negative Rate", "error"),
                        ("False Omission Rate", "error"),
                        ("False Positive Rate", "error"),
                        ("Jaccard", "error"),
                        ("Negative Predictive Value", "error"),
                        ("Precision", "error"),
                        ("Recall", "error"),
                        ("Total Positives Reference", "error"),
                        ("Total Positives Test", "error"),
                        ("True Negative Rate", "error"),
                    ]
                ),
                # "3": OrderedDict(
                #     [
                #         ("Accuracy", "error"),
                #         ("Dice", "error"),
                #         ("False Discovery Rate", "error"),
                #         ("False Negative Rate", "error"),
                #         ("False Omission Rate", "error"),
                #         ("False Positive Rate", "error"),
                #         ("Jaccard", "error"),
                #         ("Negative Predictive Value", "error"),
                #         ("Precision", "error"),
                #         ("Recall", "error"),
                #         ("Total Positives Reference", "error"),
                #         ("Total Positives Test", "error"),
                #         ("True Negative Rate", "error"),
                #     ]
                # ),
                # "4": OrderedDict(
                #     [
                #         ("Accuracy", "error"),
                #         ("Dice", "error"),
                #         ("False Discovery Rate", "error"),
                #         ("False Negative Rate", "error"),
                #         ("False Omission Rate", "error"),
                #         ("False Positive Rate", "error"),
                #         ("Jaccard", "error"),
                #         ("Negative Predictive Value", "error"),
                #         ("Precision", "error"),
                #         ("Recall", "error"),
                #         ("Total Positives Reference", "error"),
                #         ("Total Positives Test", "error"),
                #         ("True Negative Rate", "error"),
                #     ]
                # ),
            }
        return a

def evaluate_folder(folder_with_gts: str, folder_with_predictions: str, labels: tuple, **metric_kwargs):
    """
    writes a summary.json to folder_with_predictions
    :param folder_with_gts: folder where the ground truth segmentations are saved. Must be nifti files.
    :param folder_with_predictions: folder where the predicted segmentations are saved. Must be nifti files.
    :param labels: tuple of int with the labels in the dataset. For example (0, 1, 2, 3) for Task001_BrainTumour.
    :return:
    """
    files_gt = subfiles(folder_with_gts, suffix=".nii.gz", join=False)
    files_pred = subfiles(folder_with_predictions, suffix=".nii.gz", join=False)
    assert all([i in files_pred for i in files_gt]), "files missing in folder_with_predictions"
    assert all([i in files_gt for i in files_pred]), "files missing in folder_with_gts"
    test_ref_pairs = [(join(folder_with_predictions, i), join(folder_with_gts, i)) for i in files_pred]
    res = aggregate_scores(test_ref_pairs, json_output_file=join(folder_with_predictions, "summary-nuevo-3.json"),
                           num_threads=2, labels=labels, **metric_kwargs)
    return res


def run_metrics(task_name, labels):
    predicitions = os.path.join(
        HOME,
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        OUT_FOLDER#"out_unet",
    )

    true_labels = os.path.join(
        HOME,
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        "labelsTs",
    )

    evaluate_folder(true_labels, predicitions, labels, evaluator=NiftiEvaluatorv2)

OUT_FOLDER =  'out' # 'out_unet'
ORGAN = 'Spleen'

def main():
    task_name = "Task511_SpleenTraumaCV"  # "Task510_LiverTraumaDGX"
    # task_name = "Task511_SpleenTraumaCV"  # "Task510_LiverTraumaDGX"
    labels = (1, 2)  # (1, 2, 3, 4)
    run_metrics(task_name, labels)
    print("Finished")

if __name__ == "__main__":
    HOME = 'U:\\'#'/mnt/netcache/diag/' #'U:\\'#"/mnt/netcache/diag/"
    # ORGAN = "Liver"  # "Liver" Spleen
    main()
