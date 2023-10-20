import torch

"""
For categorical representation
reference : https://github.com/werner-duvaud/muzero-general
In my opinion, support size have to cover the range of maximum absolute value of 
reward and value of entire trajectories. Support_size 30 can cover almost [-900,900].
"""
support_size = 300  # efficientzero even has 300 here
eps = 0.001
from line_profiler_pycharm import profile

support_tensor = None


@profile
def to_scalar(x):
  global support_tensor
  x = torch.softmax(x, dim=-1)
  probabilities = x
  if support_tensor is None:
    support_tensor = (torch.tensor([x for x in range(-support_size, support_size + 1)]).expand(probabilities.shape).float().to(x.device))
  x = torch.sum(support_tensor * probabilities, dim=-1, keepdim=True)
  scalar = torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)
  return scalar


def to_cr(x):
  x = x.squeeze(-1).unsqueeze(0)
  x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x
  x = torch.clip(x, -support_size, support_size)
  floor = x.floor()
  under = x - floor
  floor_prob = (1 - under)
  under_prob = under
  floor_index = floor + support_size
  under_index = floor + support_size + 1
  logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).type(torch.float32).to(x.device)
  logits.scatter_(2, floor_index.long().unsqueeze(-1), floor_prob.unsqueeze(-1))
  under_prob = under_prob.masked_fill_(2 * support_size < under_index, 0.0)
  under_index = under_index.masked_fill_(2 * support_size < under_index, 0.0)
  logits.scatter_(2, under_index.long().unsqueeze(-1), under_prob.unsqueeze(-1))
  return logits.squeeze(0)