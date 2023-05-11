import numpy as np
import nibabel as nib
from topology_loss import *


def path2tensor(path):
    mri_nii = nib.load(path)
    mri_np = np.asarray(mri_nii.get_fdata(dtype=np.float32))
    mri_tensor = torch.from_numpy(mri_np)
    return mri_tensor


def test_topological_pooling_loss():
    print('Testing by using random tensor')
    batch_size = 4
    num_classes = 2  # binary segmentation
    threshold = 0.5

    # 3D case
    input_3d = torch.rand(batch_size, num_classes, 16, 16, 16)  # Synthetic 3D input data
    input_3d = (input_3d > threshold).float()
    target_3d = torch.randint(0, num_classes, (batch_size, 16, 16, 16)).float()  # Synthetic 3D labels
    target_3d = torch.nn.functional.one_hot(target_3d.to(torch.int64), num_classes).permute(0, 4, 1, 2, 3).float()

    loss_fn = TopologicalPoolingLoss(kernel_list=[2, 4])

    loss_3d = loss_fn(input_3d, target_3d)
    assert 0 <= loss_3d.item(), f"Invalid loss value for 3D case: {loss_3d.item()}"
    print('[3D case] Topology pooling difference between input and label:', loss_3d)

    # 2D case
    input_2d = torch.rand(batch_size, num_classes, 16, 16)  # Synthetic 2D input data
    input_2d = (input_2d > threshold).float()
    target_2d = torch.randint(0, num_classes, (batch_size, 16, 16)).float()  # Synthetic 2D labels
    target_2d = torch.nn.functional.one_hot(target_2d.to(torch.int64), num_classes).permute(0, 3, 1, 2).float()

    loss_fn = TopologicalPoolingLoss(kernel_list=[2, 4])

    loss_2d = loss_fn(input_2d, target_2d)
    assert 0 <= loss_2d.item(), f"Invalid loss value for 2D case: {loss_2d.item()}"
    print('[2D case] Topology pooling difference between input and label:', loss_2d)

    print("Test passed for both 3D and 2D cases.")


def test_topological_pooling_loss_ball():
    print('Testing by using synthetics ball data')
    batch_size = 1
    num_classes = 1  # binary segmentation
    input_path = './syn_ball_input.nii.gz'
    label_path = './syn_ball_label.nii.gz'
    input_tensor_3d = path2tensor(input_path)
    label_tensor_3d = path2tensor(label_path)
    # 3D case
    input_tensor_3d = input_tensor_3d.view(batch_size, num_classes, *input_tensor_3d.size())
    label_tensor_3d = label_tensor_3d.view(batch_size, num_classes, *label_tensor_3d.size())

    loss_fn = TopologicalPoolingLoss(start_channel=0)
    loss_3d = loss_fn(input_tensor_3d, label_tensor_3d)
    assert 0 <= loss_3d.item(), f"Invalid loss value for 3D case: {loss_3d.item()}"
    print('[3D case] Topology pooling difference between input and label:', loss_3d)

    # 2D case
    input_tensor_2d = input_tensor_3d[:, :, :, :, 15]
    label_tensor_2d = label_tensor_3d[:, :, :, :, 15]
    loss_fn = TopologicalPoolingLoss(start_channel=0)
    loss_2d = loss_fn(input_tensor_2d, label_tensor_2d)
    assert 0 <= loss_2d.item(), f"Invalid loss value for 2D case: {loss_2d.item()}"
    print('[2D case] Topology pooling difference between input and label:', loss_2d)

    print("Test passed for both 3D and 2D cases.")


if __name__ == '__main__':
    test_topological_pooling_loss()
    test_topological_pooling_loss_ball()
