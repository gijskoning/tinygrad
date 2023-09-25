import random
from typing import Dict, Optional

import numpy as np
import torch
# import torch.random
from tqdm import tqdm

from examples.mlperf.unet3d.inference import evaluate
from extra.datasets.kits19 import sliding_window_inference
from extra.lr_scheduler import MultiStepLR
from extra.training import lr_warmup
from tinygrad.helpers import dtypes, getenv
from tinygrad.jit import TinyJit
from models.unet3d import UNet3D

from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from tinygrad.ops import Device
from tinygrad.tensor import Tensor

from examples.mlperf.unet3d.data_loader import get_data_loaders
from examples.mlperf.unet3d.losses import DiceCELoss, DiceScore
from examples.mlperf.unet3d.flags import Flags
from models.unet3d import UNet3D

def train(flags, model:UNet3D, train_loader, val_loader, loss_fn, score_fn):
  is_successful, diverged = False, False
  optimizer = optim.SGD(get_parameters(model), lr=flags.learning_rate, momentum=flags.momentum, weight_decay=flags.weight_decay)
  # scaler = GradScaler() # scalar is only needed when doing mixed precision. The default args have this disabled.
  if flags.lr_decay_epochs:
    scheduler = MultiStepLR(optimizer, milestones=flags.lr_decay_epochs, gamma=flags.lr_decay_factor)
  next_eval_at = flags.start_eval_at

  def training_step(x, label, lr):
    print("No jit (yet)")
    optimizer.lr = lr
    output = model(x)
    loss_value = loss_fn(output, label)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    return output, loss_value.realize() # loss_value.realize is needed
  training_step_fn = TinyJit(training_step) if getenv("JIT") else training_step
  model_eval_fn = TinyJit(model) if getenv("JIT") else model

  Tensor.training = True
  for epoch in range(1, flags.max_epochs + 1):
    print('epoch', epoch)
    cumulative_loss = []
    if epoch <= flags.lr_warmup_epochs:
      lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, epoch, flags.lr_warmup_epochs)
    loss_value = None
    for iteration, batch in enumerate(tqdm(train_loader, disable=not flags.verbose)):
      print('optimizer.lr', optimizer.lr.numpy())
      print('iteration', iteration)
      image, label = batch

      dtype_img = dtypes.half if getenv("FP16") else dtypes.float
      image, label = Tensor(image.numpy(), dtype=dtype_img), Tensor(label.numpy(),dtype=dtype_img)

      output, loss_value = training_step_fn(image, label, optimizer.lr)

      del output, image, label
      # print('eval model output', test_output[:1,0,0,0,:5])
      # print('model output', output[:1,0,0,0,:5].numpy())
      # if epoch == 4:
      #   print('model weight', model.input_block.conv1.weight.numpy()[:10])
      #   print('optimizerb2', optimizer.b[0].numpy()[0, 0, 0, :10])
      #   exit()
      print('grad', loss_value.grad.numpy())
      cumulative_loss.append(loss_value)
      print('loss_value', loss_value.numpy())
      if flags.lr_decay_epochs:
        scheduler.step()

    if epoch == next_eval_at:
      next_eval_at += flags.evaluate_every
      Tensor.training = False

      eval_metrics = evaluate(flags, model_eval_fn, val_loader, score_fn, epoch)
      eval_metrics["train_loss"] = (sum(cumulative_loss) / len(cumulative_loss)).numpy().item()

      Tensor.training = True
      print('eval_metrics', [(k, f"{m:.7f}") for k,m in eval_metrics.items()])
      if eval_metrics["mean_dice"] >= flags.quality_threshold:
        print("SUCCESSFULL", eval_metrics["mean_dice"], ">", flags.quality_threshold)
        # is_successful = True
      elif eval_metrics["mean_dice"] < 1e-6:
        print("MODEL DIVERGED. ABORTING.", eval_metrics["mean_dice"], "<", 1e-6)
        # diverged = True

    if is_successful or diverged:
      break

def test_sliding_inference():
  model = UNet3D(1, 3, debug_speed=getenv("SPEED", 3), filters=getenv("FILTERS", ()))

  flags = Flags(batch_size=2, verbose=True, data_dir=getenv("DATA_DIR", '/home/gijs/code_projects/kits19/data'))#os.environ["KITS19_DATA_DIR"])
  _, val_loader = get_data_loaders(flags, 1, 0) # todo change to tinygrad loader
  dtype_img = dtypes.half
  loader = val_loader
  # def get_score(image, label):
  #   output, label = sliding_window_inference(model, image, label, flags.val_input_shape, jit=Flags)
  #   # output = output[:, :, :128, :256, :256]  # todo temp
  #   # label = label[:, :, :128, :256, :256]
  #   # s += score_fn(output, label).mean().numpy()
  #   return output.realize()
  # get_score = TinyJit(get_score)
  model_jit = TinyJit(model)
  for iteration, batch in enumerate(tqdm(loader, disable=not flags.verbose)):
    print(iteration)
    image, label = batch
    image, label = Tensor(image.numpy(), dtype=dtype_img), Tensor(label.numpy(), dtype=dtype_img)
    # output = get_score(image, label)# todo might need to give model?
    sliding_window_inference(model_jit, image, label, flags.val_input_shape)
    # output = output[:, :, :128, :256, :256]  # todo temp
    # label = label[:, :, :128, :256, :256]
    # s += score_fn(output, label).mean().numpy()
    if iteration == 3:
      break

if __name__ == "__main__":
  # test_sliding_inference()
  print('Device', Device.DEFAULT)
  import os
  # ~ doesnt work here
  # batch_size 2 is default: https://github.com/mlcommons/training/blob/00f04c57d589721aabce4618922780d29f73cf4e/image_segmentation/pytorch/runtime/arguments.py
  # this is the real starting script: https://github.com/mlcommons/training/blob/00f04c57d589721aabce4618922780d29f73cf4e/image_segmentation/pytorch/run_and_time.sh
  flags = Flags(batch_size=2, verbose=True, data_dir=getenv("DATA_DIR", '/home/gijs/code_projects/kits19/data'))#os.environ["KITS19_DATA_DIR"])
  flags.num_workers = 0 # for debugging
  seed = flags.seed # TODOOOOOO should check mlperf unet training too. It has different losses
  flags.evaluate_every = 20 # todo
  flags.start_eval_at = 10 # todo
  if seed is not None:
    Tensor._seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
  model = UNet3D(1, 3, debug_speed=getenv("SPEED", 3), filters=getenv("FILTERS", ()))
  if getenv("FP16"):
    weights = get_state_dict(model)
    for k, v in weights.items():
      weights[k] = v.cpu().half()
    load_state_dict(model, weights)
  print("Model params: {:,.0f}".format(sum([p.numel() for p in get_parameters(model)])))

  train_loader, val_loader = get_data_loaders(flags, 1, 0) # todo change to tinygrad loader
  loss_fn = DiceCELoss()
  score_fn = DiceScore()
  if getenv("OVERFIT"):
    val_loader = train_loader
  train(flags, model, train_loader, val_loader, loss_fn, score_fn)
# FP16=1 JIT=1 python training.py
# DATA_DIR=kits19/data_processed SPEED=1 FP16=1 JIT=1 python training.py
# HIP=1 WINO=1 DATA_DIR=kits19/data_processed SPEED=0 FP16=1 JIT=1 python training.py
# reference: https://github.com/mlcommons/training/blob/00f04c57d589721aabce4618922780d29f73cf4e/image_segmentation/pytorch/model/losses.py#L63

# todo eventually cleanup duplicate stuff. There is also things in extra/kits19