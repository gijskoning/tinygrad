from tinygrad.helpers import dtypes

from tinygrad.tensor import Tensor


def to_one_hot(array: Tensor, layout="NCDHW", channel_axis=1):
  # https://github.com/mlcommons/training/blob/00f04c57d589721aabce4618922780d29f73cf4e/image_segmentation/pytorch/model/losses.py#L63
  array = array.squeeze(dim=channel_axis) #todo could move squeeze up
  num_classes = 3
  array = Tensor.eye(num_classes, dtype=dtypes.int32, device=array.device)[array] # this is the F.one_hot function
  array = array.permute(0, 4, 1, 2, 3)
  return array