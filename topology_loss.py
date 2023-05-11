import torch
import torch.nn as nn

def project_pooling_3d_tensor(input_tensor, kernel_size):
    """Applies max pooling on the 3D tensor with the specified kernel size."""
    project_pooling = nn.MaxPool3d(kernel_size=kernel_size, stride=1)
    return project_pooling(input_tensor)

def topology_pooling_2d_tensor(input_tensor, kernel, stride):
    """Applies max pooling on the 2D tensor with the specified kernel size and stride."""
    abstract_2d_pooling = nn.MaxPool2d(kernel_size=kernel, stride=stride)
    abstract_pooling = abstract_2d_pooling(input_tensor)
    return abstract_pooling

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
