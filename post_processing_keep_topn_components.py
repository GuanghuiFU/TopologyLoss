import numpy as np
import nibabel as nib
from scipy import ndimage


def load_image_data(path):
    # load the MRI data from the given path
    mri = nib.load(path)
    # convert the MRI data to numpy array format
    mri_data = np.asarray(mri.get_fdata(dtype=np.float32))
    # get the affine of the MRI data
    mri_affine = mri.affine
    # return the MRI data and its affine
    return mri_data, mri_affine


def save_nii(mri_np, mri_affine, save_path):
    mri_np = np.array(mri_np, dtype=np.float32)
    mri_nii = nib.Nifti1Image(mri_np, mri_affine)
    nib.save(mri_nii, save_path)


def get_top_n_largest_connected_components(volume, n):
    # Label connected components in the volume
    labeled_volume, num_components = ndimage.label(volume)
    # Get the size of each connected component
    component_sizes = ndimage.sum(volume, labeled_volume, range(1, num_components + 1))
    # Get the indices of the top N largest connected components
    top_n_indices = np.argsort(-component_sizes)[:n]
    # Create an empty volume to store the top N connected components
    top_n_volume = np.zeros_like(volume)
    # Fill in the top N connected components
    for i in top_n_indices:
        top_n_volume[labeled_volume == i + 1] = 1
    return top_n_volume


def main():
    # Number of connected components you want to keep
    num_cc = 2
    data_path = "your/data/path"
    data_cleaned_path = "your/data/save_path"
    pred_np, affine = load_image_data(data_path)
    pred_np_top2 = get_top_n_largest_connected_components(pred_np, num_cc)
    save_nii(pred_np_top2, affine, data_cleaned_path)


if __name__ == '__main__':
    main()
