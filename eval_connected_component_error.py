# import necessary libraries
import nibabel as nib
import numpy as np
import skimage.measure
import glob
import os


def load_image_data(path):
    # load the MRI data from the given path
    mri = nib.load(path)
    # convert the MRI data to numpy array format
    mri_data = np.asarray(mri.get_fdata(dtype=np.float32))
    # get the affine of the MRI data
    mri_affine = mri.affine
    # return the MRI data and its affine
    return mri_data, mri_affine


def calculate_connect_component(mri_data):
    # calculate the number of connect components in the MRI data
    _, num_connect_component = skimage.measure.label(mri_data, return_num=True)
    return num_connect_component


def evaluate_connected_component_2d(predicted_image, label_image):
    # get the number of slices in the predicted image
    num_slice = predicted_image.shape[2]
    error_2d_list = []
    # for each slice in the predicted image
    for i in range(num_slice):
        # get the current slice from the predicted image and the label image
        pred_slice = predicted_image[:, :, i]
        label_slice = label_image[:, :, i]
        # if there is any non-zero pixel in the current slice
        if np.sum(label_slice) != 0 or np.sum(pred_slice) != 0:
            # calculate the number of connect components in the current slice of the predicted image and the label image
            pred_cc = calculate_connect_component(pred_slice)
            label_cc = calculate_connect_component(label_slice)
            # calculate the absolute difference of the number of connect components between the predicted image and the label image
            error_2d = abs(label_cc - pred_cc)
            # append the error to the list
            error_2d_list.append(error_2d)
    # convert the list of errors to a numpy array
    error_2d_np = np.array(error_2d_list)
    # return the mean of the errors
    return np.mean(error_2d_np)


def evaluate_connected_component_3d(predicted_image, label_image):
    # calculate the number of connect components in the predicted image and the label image
    pred_cc = calculate_connect_component(predicted_image)
    label_cc = calculate_connect_component(label_image)
    # return the absolute difference of the number of connect components between the predicted image and the label image
    return abs(label_cc - pred_cc)


def main():
    # define the directories where the predicted and label files are located
    pred_dir = '/your/prediction/path'
    label_dir = '/your/label/path'
    # get a list of all files in the prediction directory with .nii.gz extension
    pred_files = glob.glob(os.path.join(pred_dir, '*.nii.gz'))
    error_dict = {}
    # for each file in the prediction directory
    for pred_file in pred_files:
        # get the name of the file
        pred_file_name = os.path.basename(pred_file)
        # create the path to the corresponding label file
        label_file = os.path.join(label_dir, pred_file_name)
        # load the predicted image and the label image
        predicted_image, _ = load_image_data(pred_file)
        label_image, _ = load_image_data(label_file)
        # calculate the 2D error and 3D error between the predicted image and the label image
        cc_error_2d = evaluate_connected_component_2d(predicted_image, label_image)
        cc_error_3d = evaluate_connected_component_3d(predicted_image, label_image)
        print(f'Filename:{pred_file_name}: 2d CC: {cc_error_2d}, 3d CC: {cc_error_3d}')
        # store the errors in a dictionary
        error_dict[pred_file_name] = (cc_error_2d, cc_error_3d)
    # print the dictionary of errors
    print(error_dict)


# call the main function
if __name__ == '__main__':
    main()
