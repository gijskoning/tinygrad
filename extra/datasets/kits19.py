import random
import functools
from pathlib import Path
import requests
import numpy as np
import nibabel as nib
import scipy
import torch
import torch.nn.functional as F
from tinygrad.tensor import Tensor

BASEDIR = Path(__file__).parent / "kits19" / "data"

"""
To download the dataset:
```sh
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
cd ..
mv kits extra/datasets
```
"""

@functools.lru_cache(None)
def get_val_files():
  data = requests.get("https://raw.githubusercontent.com/mlcommons/training/master/image_segmentation/pytorch/evaluation_cases.txt")
  return sorted([x for x in BASEDIR.iterdir() if x.stem.split("_")[-1] in data.text.split("\n")])

@functools.lru_cache(None)
def get_train_files():
  return sorted([x for x in BASEDIR.iterdir() if x.stem.startswith("case") and int(x.stem.split("_")[-1]) < 210 and x not in get_val_files()])

def load_pair(file_path):
  image, label = nib.load(file_path / "imaging.nii.gz"), nib.load(file_path / "segmentation.nii.gz")
  image_spacings = image.header["pixdim"][1:4].tolist()
  image, label = image.get_fdata().astype(np.float32), label.get_fdata().astype(np.uint8)
  image, label = np.expand_dims(image, 0), np.expand_dims(label, 0)
  return image, label, image_spacings

def resample3d(image, label, image_spacings, target_spacing=(1.6, 1.2, 1.2)):
  if image_spacings != target_spacing:
    spc_arr, targ_arr, shp_arr = np.array(image_spacings), np.array(target_spacing), np.array(image.shape[1:])
    new_shape = (spc_arr / targ_arr * shp_arr).astype(int).tolist()
    image = F.interpolate(torch.from_numpy(np.expand_dims(image, axis=0)), size=new_shape, mode="trilinear", align_corners=True)
    label = F.interpolate(torch.from_numpy(np.expand_dims(label, axis=0)), size=new_shape, mode="nearest")
    image = np.squeeze(image.numpy(), axis=0)
    label = np.squeeze(label.numpy(), axis=0)
  return image, label

def normal_intensity(image, min_clip=-79.0, max_clip=304.0, mean=101.0, std=76.9):
  image = np.clip(image, min_clip, max_clip)
  image = (image - mean) / std
  return image

def pad_to_min_shape(image, label, roi_shape=(128, 128, 128)):
  current_shape = image.shape[1:]
  bounds = [max(0, roi_shape[i] - current_shape[i]) for i in range(3)]
  paddings = [(0, 0)] + [(bounds[i] // 2, bounds[i] - bounds[i] // 2) for i in range(3)]
  image = np.pad(image, paddings, mode="edge")
  label = np.pad(label, paddings, mode="edge")
  return image, label

# ***** transformation helpers (for training set) *****

def rand_foreg_cropd(image, label, patch_size=(128,128,128)):
  def adjust(foreg_slice, patch_size, label, idx):
    diff = patch_size[idx-1] - (foreg_slice[idx].stop - foreg_slice[idx].start)
    sign, diff = -1 if diff < 0 else 1, abs(diff)
    ladj = 0 if diff == 0 else random.randrange(diff)
    hadj = diff - ladj
    low = max(0, foreg_slice[idx].start - sign * ladj)
    high = min(label.shape[idx], foreg_slice[idx].stop + sign * hadj)
    diff = patch_size[idx-1] - (high - low)
    if diff > 0 and low == 0:
      high += diff
    elif diff > 0:
      low -= diff
    return low, high
  cl = np.random.choice(np.unique(label[label>0]))
  foreg_slices = scipy.ndimage.find_objects(scipy.ndimage.label(label==cl)[0])
  foreg_slices = [x for x in foreg_slices if x is not None]
  slice_volumes = [np.prod([s.stop - s.start for s in sl]) for sl in foreg_slices]
  foreg_slices = [foreg_slices[i] for i in np.argsort(slice_volumes)[-2:]]
  if not foreg_slices:
    return rand_crop(image, label)
  foreg_slice = foreg_slices[random.randrange(len(foreg_slices))]
  low_x, high_x = adjust(foreg_slice, patch_size, label, 1)
  low_y, high_y = adjust(foreg_slice, patch_size, label, 2)
  low_z, high_z = adjust(foreg_slice, patch_size, label, 3)
  image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
  label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
  return image, label

def rand_crop(image, label, patch_size=(128,128,128)):
  ranges = [s - p for s, p in zip(image.shape[1:], patch_size)]
  cord = [0 if x == 0 else random.randrange(x) for x in ranges]
  low_x, high_x = cord[0], cord[0] + patch_size[0]
  low_y, high_y = cord[1], cord[1] + patch_size[1]
  low_z, high_z = cord[2], cord[2] + patch_size[2]
  image = image[:, low_x:high_x, low_y:high_y, low_z:high_z]
  label = label[:, low_x:high_x, low_y:high_y, low_z:high_z]
  return image, label

def rand_balanced_crop(image, label, roi_shape, oversampling=0.4):
  if random.random() < oversampling:
    image, label = rand_foreg_cropd(image, label, roi_shape)
  else:
    image, label = rand_crop(image, label, roi_shape)
  return image, label

def rand_flip(image, label, axis=(1,2,3)):
  if random.random() <  1 / len(axis):
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
  return image, label

def random_brightness_augmentation(image, factor=0.3, prob=0.1):
  if random.random() < prob:
    factor = np.random.uniform(low=1.0-factor, high=1.0+factor, size=1)
    image = (image * (1 + factor)).astype(image.dtype)
  return image

def gaussian_noise(image, mean=0.0, std=0.1, prob=0.1):
  if random.random() < prob:
    scale = np.random.uniform(low=0.0, high=std)
    noise = np.random.normal(loc=mean, scale=scale, size=image.shape).astype(image.dtype)
    image += noise
  return image

def preprocess(file_path, val=False, roi_shape=(128,128,128)):
  image, label, image_spacings = load_pair(file_path)
  image, label = resample3d(image, label, image_spacings)
  image = normal_intensity(image.copy())
  image, label = pad_to_min_shape(image, label, roi_shape)
  if not val:
    image, label = rand_balanced_crop(image, label, roi_shape)
    image, label = rand_flip(image, label)
    image = random_brightness_augmentation(image)
    image = gaussian_noise(image)
  return image, label

def iterate(BS=1, val=True, shuffle=False, roi_shape=(128,128,128)):
  files = get_val_files() if val else get_train_files()
  if shuffle: random.shuffle(files)
  file_num = 0
  while file_num < len(files):
    Xs, Ys = [], []
    for _ in range(BS):
      if file_num >= len(files):
        break
      X, Y = preprocess(files[file_num], val=val, roi_shape=roi_shape)
      file_num += 1
      Xs.append(X)
      Ys.append(Y)
    yield (np.array(Xs), np.array(Ys))

def gaussian_kernel(n, std):
  gaussian_1d = scipy.signal.gaussian(n, std)
  gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
  gaussian_3d = np.outer(gaussian_2d, gaussian_1d)
  gaussian_3d = gaussian_3d.reshape(n, n, n)
  gaussian_3d = np.cbrt(gaussian_3d)
  gaussian_3d /= gaussian_3d.max()
  return gaussian_3d

def pad_input(volume, roi_shape, strides, padding_mode="constant", padding_val=-2.2, dim=3):
  bounds = [(strides[i] - volume.shape[2:][i] % strides[i]) % strides[i] for i in range(dim)]
  bounds = [bounds[i] if (volume.shape[2:][i] + bounds[i]) >= roi_shape[i] else bounds[i] + strides[i] for i in range(dim)]
  paddings = [bounds[2]//2, bounds[2]-bounds[2]//2, bounds[1]//2, bounds[1]-bounds[1]//2, bounds[0]//2, bounds[0]-bounds[0]//2, 0, 0, 0, 0]
  return F.pad(torch.from_numpy(volume), paddings, mode=padding_mode, value=padding_val).numpy(), paddings

def sliding_window_inference(model, inputs, labels, roi_shape=(128, 128, 128), overlap=0.5):
  from tinygrad.jit import TinyJit
  mdl_run = TinyJit(lambda x: model(x).realize())
  image_shape, dim = list(inputs.shape[2:]), len(inputs.shape[2:])
  strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]
  bounds = [image_shape[i] % strides[i] for i in range(dim)]
  bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(dim)]
  inputs = inputs[
    ...,
    bounds[0]//2:image_shape[0]-(bounds[0]-bounds[0]//2),
    bounds[1]//2:image_shape[1]-(bounds[1]-bounds[1]//2),
    bounds[2]//2:image_shape[2]-(bounds[2]-bounds[2]//2),
  ]
  labels = labels[
    ...,
    bounds[0]//2:image_shape[0]-(bounds[0]-bounds[0]//2),
    bounds[1]//2:image_shape[1]-(bounds[1]-bounds[1]//2),
    bounds[2]//2:image_shape[2]-(bounds[2]-bounds[2]//2),
  ]
  inputs, paddings = pad_input(inputs, roi_shape, strides)
  padded_shape = inputs.shape[2:]
  size = [(inputs.shape[2:][i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]
  result = np.zeros((1, 3, *padded_shape), dtype=np.float32)
  norm_map = np.zeros((1, 3, *padded_shape), dtype=np.float32)
  norm_patch = gaussian_kernel(roi_shape[0], 0.125 * roi_shape[0])
  norm_patch = np.expand_dims(norm_patch, axis=0)
  for i in range(0, strides[0] * size[0], strides[0]):
    for j in range(0, strides[1] * size[1], strides[1]):
      for k in range(0, strides[2] * size[2], strides[2]):
        out = mdl_run(Tensor(inputs[..., i:roi_shape[0]+i,j:roi_shape[1]+j, k:roi_shape[2]+k])).numpy()
        result[..., i:roi_shape[0]+i, j:roi_shape[1]+j, k:roi_shape[2]+k] += out * norm_patch
        norm_map[..., i:roi_shape[0]+i, j:roi_shape[1]+j, k:roi_shape[2]+k] += norm_patch
  result /= norm_map
  result = result[..., paddings[4]:image_shape[0]+paddings[4], paddings[2]:image_shape[1]+paddings[2], paddings[0]:image_shape[2]+paddings[0]]
  return result, labels

if __name__ == "__main__":
  for X, Y in iterate(val=False):
    print(X.shape, Y.shape)
