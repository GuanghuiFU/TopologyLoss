# Introducing Soft Topology Constraints in Deep Learning-based Segmentation using Projected Pooling Loss

* Conference version: https://inria.hal.science/hal-03832309/file/proceedings_version.pdf 

## Introduction

Our loss function is specifically designed to capture topological differences between the ground truth and the prediction of a 3D volume. To accomplish this, we utilize a projection technique onto three planes (axial, coronal, and sagittal) and extract topological features using 2D MaxPooling with varying kernel sizes. The topological loss is then calculated as the mean absolute difference between the ground truth and predicted topological features. Although our code can also be applied to 2D data by skipping the projection step, we haven't conducted experiments to validate its effectiveness in this context. We encourage other users to experiment with our code and share their results and feedback with us. **It's worth noting that our loss function typically requires only a few additional training epochs after pre-training to achieve the best performance.** However, for consistency and control purposes, we train for the same number of epochs as the pretrained model in the paper. Feel free to explore our code and reach out to us for further discussions and collaborations.

The main functions `topology_loss.py` provided are:

- `project_pooling_3d_tensor`: Projects a 3D MRI tensor onto a plane using 3D max pooling with the specified kernel size.
- `topology_pooling_2d_tensor`: Applies 2D max pooling to a 2D MRI tensor with the specified kernel size and stride.
- `topological_pooling`: Performs topological pooling on a 3D MRI tensor by projecting it onto a specified plane and applying 2D max pooling.
- `compute_per_channel_topology_component`: Computes the difference in topological features between input and target tensors for each kernel size and stride in the provided lists.
- `TopologicalPoolingLoss`: A custom loss function class that calculates topological pooling loss between input and target tensors.

To use the loss function, simply instantiate the `TopologicalPoolingLoss` class and use it in your training loop like any other PyTorch loss function.

We also provide the following codes:

1. `eval_connected_component_error.py`: load 3d nii.gz data and evaluate at both 3D and 2D level.
2. `post_processing_keep_topn_components.py`: the post-processing method we use to clean the predictions into a target number of connected components.
3. `test.py`: an example to test our loss function by using synthetic data (`syn_ball_input.nii.gz` as input, and `syn_ball_label.nii.gz` as label) and random tensor. 

## Code Explanation

### 1. 3D to 2D Projection using 3D MaxPooling

```python
def project_pooling_3d_tensor(input_tensor, kernel_size):
    """Applies max pooling on the 3D tensor with the specified kernel size."""
    project_pooling = nn.MaxPool3d(kernel_size=kernel_size, stride=1)
    return project_pooling(input_tensor)
```

This function takes a 3D MRI tensor and applies 3D max pooling with the specified kernel size. It returns the projected tensor.

### 2. Topology Feature Extraction using 2D MaxPooling

```python
def topology_pooling_2d_tensor(input_tensor, kernel, stride):
    """Applies max pooling on the 2D tensor with the specified kernel size and stride."""
    abstract_2d_pooling = nn.MaxPool2d(kernel_size=kernel, stride=stride)
    abstract_pooling = abstract_2d_pooling(input_tensor)
    return abstract_pooling
```

This function takes a 2D MRI tensor and applies 2D max pooling with the specified kernel size and stride. It returns the pooled tensor.

### 3. 3D to 2D Projection and Topology Feature Extraction

```python
def topological_pooling(input_tensor, kernel, stride, dim):
    """Performs topological pooling on the input tensor."""
    if input_tensor.dim() == 5:  # 3D volumes
        projection_kernels = [(1, 1, input_tensor.size(4)), (input_tensor.size(2), 1, 1), (1, input_tensor.size(3), 1)]
        input_project_pooling_3d_tensor = project_pooling_3d_tensor(input_tensor, kernel_size=projection_kernels[dim])
        if dim == 0: squeeze_dim = 4
        else: squeeze_dim = 1
        input_project_pooling_3d_tensor = input_project_pooling_3d_tensor.squeeze(dim + squeeze_dim)
    elif input_tensor.dim() == 4:  # 2D images
        input_project_pooling_3d_tensor = input_tensor
    else:
        raise ValueError("'input_tensor' must be 4D or 5D tensors")
    input_2d_pooling = topology_pooling_2d_tensor(input_project_pooling_3d_tensor, kernel=kernel, stride=stride)
    return input_2d_pooling
```

This function takes a 3D or 2D tensor, kernel size, stride, and a dimension, and performs the topological pooling operation. First, it calculates the required kernel sizes for each projection plane (axial, sagittal, and coronal), then projects the tensor onto the specified plane using `project_pooling_3d_tensor` for 3D input, and skip this step for 2D input. The resulting tensor is squeezed to remove the singleton dimension, and finally, 2D max pooling is applied using `topology_pooling_2d_tensor`. The resulting pooled tensor is returned.

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
def compute_per_channel_topology_component(input, target, start_channel, kernel_list, stride_list):
    """Computes the per-channel topology component of the input and target tensors."""
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    num_channels = input.size(1)
    num_dims = input.dim() - 2  # Calculate the number of dimensions: 3 for 3D, 2 for 2D
    difference_ks_list = []
    for kernel, stride in zip(kernel_list, stride_list):
        pooling_diff = []
        for dim in range(num_dims):  # Change the loop range to accommodate 2D and 3D tensors
            pred_pooling = topological_pooling(input, kernel=kernel, stride=stride, dim=dim)
            label_pooling = topological_pooling(target, kernel=kernel, stride=stride, dim=dim)
            channel_pooling_diff = []
            for channel in range(start_channel, num_channels):  # start from 1 to ignore the background channel.
                sum_pred_pooling = torch.sum(pred_pooling, dim=(-2, -1))[:, channel, ...]
                sum_label_pooling = torch.sum(label_pooling, dim=(-2, -1))[:, channel, ...]
                difference = torch.abs(sum_pred_pooling - sum_label_pooling)
                channel_pooling_diff.append(difference)
            pooling_diff.append(torch.mean(torch.stack(channel_pooling_diff)))
        difference_ks_list.append(torch.mean(torch.stack(pooling_diff)))
    return torch.mean(torch.stack(difference_ks_list))
```

This function takes the input and target tensors, a list of kernel sizes, and a list of strides. It checks that the input and target tensors have the same shape, and calculates the difference in topological features for each kernel size and stride in the provided lists. It computes the topological pooling for the input and target tensors and sums the pooling results. Then, it calculates the absolute difference between the summed pooling results and appends it to a list. Finally, the function returns the mean of the differences across all kernel sizes.

### 5. Topological Pooling Loss Class

```python
class TopologicalPoolingLoss(nn.Module):
    def __init__(self, start_channel=1, kernel_list=None, stride_list=None):
        """Initializes the TopologicalPoolingLoss class."""
        super().__init__()
        self.start_channel = start_channel
        self.kernel_list = kernel_list or [4, 5, 8, 10, 20]
        self.stride_list = stride_list or self.kernel_list

    def forward(self, input, target):
        """Computes the topological pooling loss for the input and target tensors."""
        if input.dim() != target.dim():
            raise ValueError("'input' and 'target' have different number of dimensions")
        if input.dim() not in (4, 5):
            raise ValueError("'input' and 'target' must be 4D or 5D tensors")
        per_channel_topology_component = compute_per_channel_topology_component(input, target, self.start_channel, self.kernel_list, self.stride_list)
        return per_channel_topology_component
```

This class defines the topological pooling loss. It inherits from the `nn.Module` class and takes an optional argument, `classes`, which defaults to 2. The constructor initializes a Sigmoid normalization layer and defines the kernel and stride lists. The `forward` function takes an input and a target tensor, normalizes the input, checks that the input and target tensors have the same dimensions, and computes the topological pooling loss using the `compute_per_channel_topology_component` function.

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
