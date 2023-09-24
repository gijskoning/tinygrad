from typing import Dict, Optional

import numpy as np
import torch.random
from tqdm import tqdm
from examples.mlperf.unet3d.inference import evaluate
from tinygrad.helpers import dtypes, getenv
from tinygrad.jit import TinyJit
from models.unet3d import UNet3D

from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor




def lr_warmup(optimizer, init_lr, lr, current_epoch, warmup_epochs):
  scale = current_epoch / warmup_epochs
  optimizer.lr = init_lr + (lr - init_lr) * scale
        
def train(flags, model:UNet3D, train_loader, val_loader, loss_fn, score_fn):
  optimizer = optim.SGD(get_parameters(model), lr=flags.learning_rate, momentum=flags.momentum, weight_decay=flags.weight_decay)
  # scaler = GradScaler() # TODO: add grad scaler
  
  next_eval_at = flags.start_eval_at

  if flags.lr_decay_epochs:
    raise NotImplementedError("TODO: lr decay")

  def step(x, label): # todo giving a list to jit works. Could refactor jit to also handle dictionaries (for gpt2)
    print("No jit (yet)")
    output = model(x)
    loss_value = loss_fn(output, label)
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    return output, loss_value.realize() # loss_value.realize is needed
  step_fn = TinyJit(step) if getenv("JIT") else step

  Tensor.training = True

  for epoch in range(1, flags.max_epochs + 1):
    cumulative_loss = []
    if epoch <= flags.lr_warmup_epochs and flags.lr_warmup_epochs > 0:
      # lr_warmup(optimizer, flags.init_learning_rate, flags.learning_rate, epoch, flags.lr_warmup_epochs)
      # a = Tensor.rand(*label.shape, dtype=dtypes.int32).realize()
      loss_value = None
      optimizer.zero_grad()
      print("HI")
      for iteration, batch in enumerate(tqdm(train_loader, disable=not flags.verbose)):
        print('i', iteration)
        image, label = batch
        image = Tensor(image.numpy(), requires_grad=False)
        label = Tensor(label.numpy(), requires_grad=False)
        output, loss_value = step_fn(image, label)
        grad = loss_value.grad
        print(iteration)
        print('optimizerb2', optimizer.b[0].numpy()[0,0,0,:10])

        if epoch == 7:
          exit()
        print('grad', grad.numpy())
        # loss_value = reduce_tensor(loss_value, world_size).detach().cpu().numpy() # TODO: reduce tensor for distributed training
        cumulative_loss.append(loss_value)
        print('loss_value', loss_value.numpy())
      # if flags.lr_decay_epochs:
      #   # pass
      #   # scheduler.step()
      #
      #   if epoch == next_eval_at:
      #     next_eval_at += flags.evaluate_every
      #     # del output
      #
      #     eval_metrics = evaluate(flags, model, val_loader, loss_fn, score_fn, epoch)
      #     eval_metrics["train_loss"] = sum(cumulative_loss) / len(cumulative_loss)
      #
      #     model.train()
      #
      #     if eval_metrics["mean_dice"] >= flags.quality_threshold:
      #       is_successful = True
      #     elif eval_metrics["mean_dice"] < 1e-6:
      #       print("MODEL DIVERGED. ABORTING.")
      #       diverged = True
      #
      #   if is_successful or diverged:
      #     break
   
if __name__ == "__main__":
  seed = 0
  if seed is not None:
    Tensor._seed = seed
    np.random.seed(seed)
    torch.random.manual_seed(seed)
  from examples.mlperf.unet3d.data_loader import get_data_loaders
  from examples.mlperf.unet3d.losses import DiceCELoss, DiceScore
  from examples.mlperf.unet3d import Flags
  from models.unet3d import UNet3D
  import os
  # ~ doesnt work here
  flags = Flags(batch_size=1, verbose=True, data_dir='/home/gijs/code_projects/kits19/data')#os.environ["KITS19_DATA_DIR"])
  model = UNet3D(1, 3)
  if getenv("FP16"):
    weights = get_state_dict(model)
    for k, v in weights.items():
      weights[k] = v.cpu().half()
    load_state_dict(model, weights)
  print("Model params: {:,.0f}".format(sum([p.numel() for p in get_parameters(model)])))

  train_loader, val_loader = get_data_loaders(flags, 1, 0)
  loss_fn = DiceCELoss()
  score_fn = DiceScore()
  train(flags, model, train_loader, val_loader, loss_fn, score_fn)
# FP16=1 JIT=1 python training.py