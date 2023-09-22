from pathlib import Path
from typing import Any
import torch
from tinygrad import nn
from tinygrad.helpers import getenv
from tinygrad.jit import TinyJit
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from tinygrad.tensor import Tensor
from extra.utils import download_file, get_child


class ConvBlock:
  def __init__(
    self,
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    conv_type="regular",
  ) -> None:
    self.conv = nn.Conv3d(
      in_channels,
      out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      bias=False,
    )
    self.norm = nn.InstanceNorm(out_channels, affine=True)
    self.weight = self.conv.weight # might need cleaning

  def __call__(self, x: Tensor) -> Tensor:
    x = self.conv(x)
    x = self.norm(x)
    return x.relu()


class ConvTransposeBlock:
  def __init__(
    self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
  ) -> None:
    self.conv = nn.ConvTranspose3d(
      in_channels,
      out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      bias=False,
    )

  def __call__(self, x: Tensor) -> Tensor:
    return self.conv(x)


class DownsampleBlock:
  def __init__(self, in_channels, out_channels):
    super(DownsampleBlock, self).__init__()
    self.conv1 = ConvBlock(in_channels, out_channels, stride=2)
    self.conv2 = ConvBlock(out_channels, out_channels)

  def __call__(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    return x


class UpsampleBlock:
  def __init__(self, in_channels, out_channels):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.upsample_conv = ConvTransposeBlock(
      in_channels, out_channels, kernel_size=2, stride=2, padding=0
    )
    self.conv1 = ConvBlock(2 * out_channels, out_channels)
    self.conv2 = ConvBlock(
      out_channels,
      out_channels,
    )

  def __call__(self, x, skip):
    x = self.upsample_conv(x)
    x = Tensor.cat(x, skip, dim=1)
    x = self.conv1(x)
    x = self.conv2(x)
    return x


class InputBlock:
  def __init__(self, in_channels, out_channels):
    # self.conv1 = [ConvBlock(in_channels, out_channels), ...] # is actually the layout for the weights from pretrained

    self.conv1 = ConvBlock(in_channels, out_channels)
    self.conv2 = ConvBlock(out_channels, out_channels)

  def __call__(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    return x


class OutputLayer:
  def __init__(self, in_channels, n_class):
    super(OutputLayer, self).__init__()
    self.conv = nn.Conv3d(
      in_channels, n_class, kernel_size=1, stride=1, padding=0, bias=True
    )

  def __call__(self, x):
    return self.conv(x)


class UNet3D:
  def __init__(self, in_channels, n_class):
    # filters = [32, 64, 128, 256, 320]
    filters = [max(1,i//256) for i in [32, 64, 128, 256, 320]] # todo fix. This makes it fit on my pc
    filters = [1, 2] # todo fix. This makes it fit on my pc
    self.filters = filters

    self.inp = filters[:-1]
    self.out = filters[1:]
    input_dim = filters[0]

    self.input_block = InputBlock(in_channels, input_dim)

    self.downsample = [
      DownsampleBlock(i, o) for idx, (i, o) in enumerate(zip(self.inp, self.out))
    ]
    self.bottleneck = DownsampleBlock(filters[-1], filters[-1])
    upsample = [UpsampleBlock(filters[-1], filters[-1])]
    upsample.extend(
      [
        UpsampleBlock(i, o)
        for idx, (i, o) in enumerate(
          zip(reversed(self.out), reversed(self.inp))
        )
      ]
    )
    self.upsample = upsample
    self.output = OutputLayer(input_dim, n_class)

  def __call__(self, x):

    x = self.input_block(x)
    outputs = [x]

    for downsample in self.downsample:
      x = downsample(x)
      outputs.append(x)

    x = self.bottleneck(x)

    for upsample, skip in zip(self.upsample, reversed(outputs)):
      x = upsample(x, skip)

    x = self.output(x)
    params = get_parameters(self)
    for p in params: p.realize()
    return x.realize()

  # def load_from_pretrained(self):
  #   fn = Path(__file__).parents[1] / "weights" / "unet-3d.ckpt"
  #   download_file("https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch.ptc?download=1", fn)
  #   state_dict = torch.jit.load(fn, map_location=torch.device("cpu")).state_dict()
  #   for k, v in state_dict.items():
  #     obj = get_child(self, k)
  #     assert obj.shape == v.shape, (k, obj.shape, v.shape)
  #     obj.assign(v.numpy())

  def load_from_pretrained(self, dtype="float32"):
    # raise NotImplementedError("TODO: load pretrained weights")
    fn = Path(__file__).parent.parent / "weights" / "unet-3d.ckpt"
    download_file(
      "https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch.ptc?download=1",
      fn,
    )
    state_dict = torch.jit.load(fn, map_location=torch.device("cpu")).state_dict()
    for k, v in state_dict.items():
      obj = get_child(self, k)
      assert obj.shape == v.shape, (k, obj.shape, v.shape)
      # obj.assign(v.numpy().astype(dtype))
      obj.assign(v.numpy())

if __name__ == "__main__":
  Tensor.training = True

  model = UNet3D(1, 3)
  if getenv("FP16"):
    print("FP16 yes")
    weights = get_state_dict(model)
    for k, v in weights.items():
      weights[k] = v.cpu().half()
    load_state_dict(model, weights)
  params = get_parameters(model)
  optimizer = optim.SGD(params, lr=0.1, momentum=0.1, weight_decay=1e-4)

  Tensor.training = True
  optimizer.zero_grad()

  loss_fn = lambda x,y: (x-y).mean()
  def step(optimizer:optim.SGD, x):
    output = model(x)
    label = Tensor.rand(*output.shape) # temp

    loss_value = loss_fn(output, label)

    loss_value.backward()

    optimizer.step()
    return optimizer, loss_value
    # y, params = model(x)
    # y2 = Tensor.rand(*y.shape)

    # loss_value.backward()
    # optimizer.step()
  if getenv("JIT"):
    step = TinyJit(step)
  for _ in range(8):
    x = Tensor.rand(1, 1, 128, 128, 128)

    optimizer, loss_value = step(optimizer, x)
    print('loss_value', loss_value.numpy())
    optimizer.zero_grad()

  #JIT=1 FP16=1 python ../../../models/unet3d.py