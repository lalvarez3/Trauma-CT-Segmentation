import os

import SimpleITK as sitk
from monai.transforms import AsDiscrete, Compose, KeepLargestConnectedComponent, Orientation
from nnunet.evaluation.evaluator import Evaluator, evaluate_folder
import numpy as np


class NiftiEvaluatorv2(Evaluator):
    def __init__(self, *args, **kwargs):

        self.test_nifti = None
        self.reference_nifti = None
        super(NiftiEvaluatorv2, self).__init__(*args, **kwargs)

    def set_test(self, test):
        """Set the test segmentation."""

        if test is not None:
            self.test_nifti = sitk.ReadImage(test)
            test_preprocess = sitk.GetArrayFromImage(self.test_nifti)
            print("preprocessing size", test_preprocess.shape)
            post_transformation = Compose(
                [
                    Orientation(axcodes="RAS"),
                    KeepLargestConnectedComponent(
                        [1, 2],
                        is_onehot=False,
                        independent=False,  # Pensar como pasar esto sin hardcodear
                    ),
                ]
            )

            test_postprocess = post_transformation(
                np.expand_dims(test_preprocess, 0)
            ).squeeze()
            print("post processing size", test_preprocess.shape)
            super(NiftiEvaluatorv2, self).set_test(test_postprocess)
        else:
            self.test_nifti = None
            super(NiftiEvaluatorv2, self).set_test(test)

    def set_reference(self, reference):
        """Set the reference segmentation."""

        if reference is not None:
            self.reference_nifti = sitk.ReadImage(reference)
            reference_img = sitk.GetArrayFromImage(self.reference_nifti)
            reference_img = Orientation(axcodes="RAS")(reference_img)
            print("reference size", reference_img.shape)
            super(NiftiEvaluatorv2, self).set_reference(
                reference_img
            )
        else:
            self.reference_nifti = None
            super(NiftiEvaluatorv2, self).set_reference(reference)

    def evaluate(self, test=None, reference=None, voxel_spacing=None, **metric_kwargs):

        if voxel_spacing is None:
            voxel_spacing = np.array(self.test_nifti.GetSpacing())[::-1]
            metric_kwargs["voxel_spacing"] = voxel_spacing

        return super(NiftiEvaluatorv2, self).evaluate(test, reference, **metric_kwargs)


def run_metrics(task_name, labels):
    predicitions = os.path.join(
        "U:\\",
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        "out",
    )
    true_labels = os.path.join(
        "U:\\",
        "lauraalvarez",
        "nnunet",
        "nnUNet_raw_data",
        task_name,
        "labelsTs",
    )
    evaluate_folder(true_labels, predicitions, labels, evaluator=NiftiEvaluatorv2)


def main():
    task_name = "Task506_VITrauma"
    labels = (1, 2)
    run_metrics(task_name, labels)
    print("Finished")


if __name__ == "__main__":
    main()
