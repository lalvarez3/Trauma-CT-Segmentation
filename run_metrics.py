"""

This script is used to run the calculate the metrics on the test data. The metrics are calculated using the nnunet framework. The script is run from the command line using the following command:
The original function from the nnunet authors: https://github.com/MIC-DKFZ/nnUNet/blob/aa53b3b87130ad78f0a28e6169a83215d708d659/nnunet/evaluation/evaluator.py
has been modified according to our needs. The original function is called from the main function of this script.

"""
import os
from collections import OrderedDict

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import join, subfiles
from monai.transforms import (AddChanneld, AsChannelFirstd, Compose,
                              KeepLargestConnectedComponentd,
                              LoadImaged)
from nnunet.evaluation.evaluator import Evaluator, aggregate_scores

from utils import injury_postprocessing


# Reimplementation of the original NiftiEvaluator class from the nnuent
class NiftiEvaluatorv2(Evaluator):
    def __init__(self, *args, **kwargs):

        self.test_nifti = None  # prediction
        self.reference_nifti = None  # golden label
        super(NiftiEvaluatorv2, self).__init__(*args, **kwargs)

    def set_test(self, test):
        """Set the test segmentation."""
        if test is not None:
            self.test_path = test  # prediction
            print("reading image", test)
            self.test_nifti = sitk.ReadImage(test)
            post_pro = True  # Set to True if you want to post process the prediction
            # if not post_pro: test_preprocess = sitk.GetArrayFromImage(self.test_nifti)

            post_transformation = Compose(
                [
                    LoadImaged(keys=["label", "image"]),  # Load the image and the label
                    AsChannelFirstd(
                        keys=["label", "image"]
                    ),  # Change the channel dimension to the first dimension
                    AddChanneld(
                        keys=["label", "image"]
                    ),  # Add a channel dimension to the label
                    KeepLargestConnectedComponentd(
                        keys=["label"],
                        applied_labels=[1, 2],
                        is_onehot=False,
                        independent=False,
                        connectivity=None,
                    ),  # Keep the largest connected component
                    injury_postprocessing(
                        keys=["image", "label"],
                        organ=ORGAN,
                        settings={
                            "iterations": 2,
                            "smoothing": 2,
                            "balloon": 0,
                            "threshold": "auto",
                            "sigma": 2,
                            "alpha": 7000,
                        }, # Settings for the post processing
                    ),
                ]
            )

            label_name = os.path.basename(test) # Get the name of the prediction
            if post_pro: # If post processing is True the image also needs to be loaded
                image_name = ( # Get the name of the image
                    label_name.split(".")[0]
                    + "_0000"
                    + "."
                    + label_name.split(".")[1]
                    + "."
                    + label_name.split(".")[2]
                )
                test_preprocess = [ 
                    {
                        "image": test.replace(OUT_FOLDER, "imagesTs").replace(
                            label_name, image_name
                        ),
                        "label": test,
                    }
                ]
            else: # If post processing is False the image does not need to be loaded
                test_preprocess = [{"label": test}]
            # print(test_preprocess)

            test_postprocess = post_transformation(
                test_preprocess  # Post process the prediction
            )
            test_postprocess = test_postprocess[0]["label"].squeeze() # Get the post processed label
            super(NiftiEvaluatorv2, self).set_test(test_postprocess)
        else:
            self.test_nifti = None
            super(NiftiEvaluatorv2, self).set_test(test)

    def set_reference(self, reference):
        """Set the reference (golden label) segmentation."""

        if reference is not None: 
            self.reference_path = reference
            T = Compose(
                [
                    LoadImaged(keys=["label"]), # Load the label
                    AsChannelFirstd(keys=["label"]), # Change the channel dimension to the first dimension
                ]
            )
            reference_img = T([{"label": reference}])[0]["label"] # Load the label
            # reference_img = Orientation(axcodes="RAS")(reference_img) # Change the orientation to RAS
            super(NiftiEvaluatorv2, self).set_reference(reference_img) # Set the reference
        else: # if error loading the reference
            self.reference_nifti = None  
            super(NiftiEvaluatorv2, self).set_reference(reference)

    def evaluate(self, test=None, reference=None, voxel_spacing=None, **metric_kwargs):
        """Evaluate the test segmentation against the reference segmentation."""
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
            }
        return a


def evaluate_folder(
    folder_with_gts: str, folder_with_predictions: str, labels: tuple, **metric_kwargs
):
    """
    writes a summary.json to folder_with_predictions
    :param folder_with_gts: folder where the ground truth segmentations are saved. Must be nifti files.
    :param folder_with_predictions: folder where the predicted segmentations are saved. Must be nifti files.
    :param labels: tuple of int with the labels in the dataset. For example (0, 1, 2, 3) for Task001_BrainTumour.
    :return:
    """
    files_gt = subfiles(folder_with_gts, suffix=".nii.gz", join=False)
    files_pred = subfiles(folder_with_predictions, suffix=".nii.gz", join=False)
    assert all(
        [i in files_pred for i in files_gt]
    ), "files missing in folder_with_predictions"
    assert all([i in files_gt for i in files_pred]), "files missing in folder_with_gts"
    test_ref_pairs = [
        (join(folder_with_predictions, i), join(folder_with_gts, i)) for i in files_pred
    ]
    res = aggregate_scores(
        test_ref_pairs,
        json_output_file=join(folder_with_predictions, "summary-nuevo-3.json"),
        num_threads=2,
        labels=labels,
        **metric_kwargs,
    )
    return res


def run_metrics(task_name, labels):
    """
    Calculates the metrics for the task_name dataset.

    Args:
        task_name (str): Name of the task. For example, Task510_LiverTraumaDGX.
        labels (tuple): Tuple of int with the labels in the dataset. For example (0, 1, 2) for Task510_LiverTraumaDGX.

    """

    # Get the path to the dataset
    predicitions = os.path.join(
        HOME,
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        OUT_FOLDER,  # Folder where predictions are saved
    )

    # Get the path to the ground truth
    true_labels = os.path.join(
        HOME,
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        "labelsTs",
    )

    # Calculate the metrics
    evaluate_folder(true_labels, predicitions, labels, evaluator=NiftiEvaluatorv2)


OUT_FOLDER = "out"  # 'out_unet' #  Folder where predictions are saved
ORGAN = "Spleen" # 'Liver' #  Name of the task organ
HOME = "U:\\" # Base path to the dataset

def main():
    task_name = "Task511_SpleenTraumaCV"  # "Task510_LiverTraumaDGX"
    labels = (1, 2)  # (1, 2, 3, 4)
    run_metrics(task_name, labels)
    print("Finished")


if __name__ == "__main__":
    main()
