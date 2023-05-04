# Introducing Soft Topology Constraints in Deep Learning-based Segmentation using Projected Pooling Loss

* Conference version: https://inria.hal.science/hal-03832309/file/proceedings_version.pdf

## Introduction

This loss function focuses on capturing topological differences between the ground truth and the prediction of a 3D volume. It achieves this by projecting the 3D volume onto three planes (axial, coronal, and sagittal) and then characterizing the topological features using 2D MaxPooling with different kernel sizes. The topological loss is the mean absolute difference between the topological features of the ground truth and the prediction.

## Code Explanation

### 1. 3D to 2D Projection using 3D MaxPooling

```python
def project_pooling_3d_tensor(mri_tensor, kernel_size):
    project_pooling = nn.MaxPool3d(kernel_size=kernel_size, stride=1)
    return project_pooling(mri_tensor)
```

This function takes a 3D MRI tensor and applies 3D max pooling with the specified kernel size. It returns the projected tensor.

### 2. Topology Feature Extraction using 2D MaxPooling

```python
def topology_pooling_2d_tensor(mri_tensor, kernel, stride):
    topology_2d_pooling = nn.MaxPool2d(kernel_size=kernel, stride=stride)
    return topology_2d_pooling(mri_tensor)
```

This function takes a 2D MRI tensor and applies 2D max pooling with the specified kernel size and stride. It returns the pooled tensor.

### 3. 3D to 2D Projection and Topology Feature Extraction

```python
def topological_pooling(mri_tensor, kernel, stride, dim):
    projection_kernels = [(1, 1, mri_tensor.size(4)), (mri_tensor.size(2), 1, 1), (1, mri_tensor.size(3), 1)]
    mri_project_pooling_3d_tensor = project_pooling_3d_tensor(mri_tensor, kernel_size=projection_kernels[dim])
    mri_project_pooling_3d_tensor = mri_project_pooling_3d_tensor.squeeze(dim + 2)
    mri_2d_pooling = topology_pooling_2d_tensor(mri_project_pooling_3d_tensor, kernel=kernel, stride=stride)
    return mri_2d_pooling
```

This function takes a 3D MRI tensor, kernel size, stride, and a dimension, and performs the topological pooling operation. First, it calculates the required kernel sizes for each projection plane (axial, sagittal, and coronal), then projects the 3D tensor onto the specified plane using `project_pooling_3d_tensor`. The resulting tensor is squeezed to remove the singleton dimension, and finally, 2D max pooling is applied using `topology_pooling_2d_tensor`. The resulting pooled tensor is returned.

The `projection_kernels` list contains three kernel sizes, each corresponding to the projection onto a specific plane: axial, sagittal, and coronal. The dimensions of these kernels are determined by the shape of the input 3D MRI tensor.

Here's a breakdown of the projection kernels and their corresponding dimensions:

1. Axial projection (dim = 0): 
   - Kernel size: (1, 1, mri_tensor.size(4))
   - This kernel size is for the axial projection because it keeps the width and height dimensions of the input tensor (by setting kernel size to 1 for these dimensions) and takes the maximum value along the slice dimension (by setting kernel size to mri_tensor.size(4), the number of slices).

2. Sagittal projection (dim = 1):
   - Kernel size: (mri_tensor.size(2), 1, 1)
   - This kernel size is for the sagittal projection because it keeps the height and slice dimensions of the input tensor (by setting kernel size to 1 for these dimensions) and takes the maximum value along the width dimension (by setting kernel size to mri_tensor.size(2), the width of the tensor).

3. Coronal projection (dim = 2):
   - Kernel size: (1, mri_tensor.size(3), 1)
   - This kernel size is for the coronal projection because it keeps the width and slice dimensions of the input tensor (by setting kernel size to 1 for these dimensions) and takes the maximum value along the height dimension (by setting kernel size to mri_tensor.size(3), the height of the tensor).

When the `topological_pooling` function is called with a specific `dim` value (0, 1, or 2), the corresponding kernel size is selected from the `projection_kernels` list and used in the `project_pooling_3d_tensor` function to perform the 3D MaxPooling projection onto the desired plane.

#### 4. Compute per-channel topology component

```python
def compute_per_channel_topology_component(input, target, kernel_list, stride_list):
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    difference_ks_list = []
    for kernel, stride in zip(kernel_list, stride_list):
        pooling_diff = []
        for dim in range(3):
            pred_pooling = topological_pooling(input, kernel=kernel, stride=stride, dim=dim)
            label_pooling = topological_pooling(target, kernel=kernel, stride=stride, dim=dim)
            sum_pred_pooling = torch.sum(pred_pooling, dim=(2, 3))[:, 1, ...]
            sum_label_pooling = torch.sum(label_pooling, dim=(2, 3))[:, 1, ...]
            pooling_diff.append(torch.abs(sum_pred_pooling - sum_label_pooling))
        difference_ks_list.append(torch.mean(torch.stack(pooling_diff)))
    return torch.mean(torch.stack(difference_ks_list))

```

This function takes the input and target tensors, a list of kernel sizes, and a list of strides. It checks that the input and target tensors have the same shape, and calculates the difference in topological features for each kernel size and stride in the provided lists. It computes the topological pooling for the input and target tensors and sums the pooling results. Then, it calculates the absolute difference between the summed pooling results and appends it to a list. Finally, the function returns the mean of the differences across all kernel sizes.

### 5. Topological Pooling Loss Class

```python
class TopologicalPoolingLoss(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.classes = classes
        self.normalization = nn.Sigmoid()
        self.kernel_list = [4, 5, 8, 10, 20]
        self.stride_list = self.kernel_list

    def forward(self, input, target):
        input = self.normalization(input)
        assert input.dim() == target.dim() == 5, "'input' and 'target' have different number of dims"
        per_channel_topology_component = compute_per_channel_topology_component(input, target, self.kernel_list, self.stride_list)
        return per_channel_topology_component

```

This class defines the topological pooling loss. It inherits from the `nn.Module` class and takes an optional argument, `classes`, which defaults to 2. The constructor initializes a Sigmoid normalization layer and defines the kernel and stride lists. The `forward` function takes an input and a target tensor, normalizes the input, checks that the input and target tensors have the same dimensions, and computes the topological pooling loss using the `compute_per_channel_topology_component` function.

### README

Topological Pooling Loss is a custom loss function for 3D segmentation tasks, designed to better capture topological features and improve segmentation performance. The loss function works by projecting the 3D volume onto three planes (axial, sagittal, and coronal) and characterizing topological features using pooling layers with different kernel sizes.

The implementation includes a `TopologicalPoolingLoss` class that inherits from PyTorch's `nn.Module` class. The class takes an optional argument for the number of classes (default is 2) and calculates the topological pooling loss between input and target tensors.

The main functions provided are:

- `project_pooling_3d_tensor`: Projects a 3D MRI tensor onto a plane using 3D max pooling with the specified kernel size.
- `topology_pooling_2d_tensor`: Applies 2D max pooling to a 2D MRI tensor with the specified kernel size and stride.
- `topological_pooling`: Performs topological pooling on a 3D MRI tensor by projecting it onto a specified plane and applying 2D max pooling.
- `compute_per_channel_topology_component`: Computes the difference in topological features between input and target tensors for each kernel size and stride in the provided lists.
- `TopologicalPoolingLoss`: A custom loss function class that calculates topological pooling loss between input and target tensors.

To use the loss function, simply instantiate the `TopologicalPoolingLoss` class and use it in your training loop like any other PyTorch loss function.

### Citing us

* Fu, Guanghui, Rosana El Jurdi, Lydia Chougar, Didier Dormont, Romain Valabregue, Stéphane Lehéricy, and Olivier Colliot. "Introducing Soft Topology Constraints in Deep Learning-based Segmentation using Projected Pooling Loss." In *SPIE Medical Imaging 2023*. 2023.

```
@inproceedings{fu2023introducing,
  title={Introducing Soft Topology Constraints in Deep Learning-based Segmentation using Projected Pooling Loss},
  author={Fu, Guanghui and El Jurdi, Rosana and Chougar, Lydia and Dormont, Didier and Valabregue, Romain and Leh{\'e}ricy, St{\'e}phane and Colliot, Olivier},
  booktitle={SPIE Medical Imaging 2023},
  year={2023}
}
```
