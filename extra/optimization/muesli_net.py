# Small neural nets with PyTorch
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from extra.optimization.helpers import MAX_DIMS, ast_str_to_lin, lin_to_feats
from tinygrad.codegen import search
from tinygrad.codegen.search import bufs_from_lin, get_linearizer_actions, time_linearizer

num_filters = 4  # todo was 16
num_blocks = 1  # was 4


class Conv(nn.Module):
  def __init__(self, filters0, filters1, kernel_size, bn=False):
    super().__init__()
    self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
    self.bn = None
    if bn:
      self.bn = nn.BatchNorm2d(filters1)

  def forward(self, x):
    h = self.conv(x)
    if self.bn is not None:
      h = self.bn(h)
    return h


class ResidualBlock(nn.Module):
  def __init__(self, size):
    super().__init__()
    # self.conv = Conv(filters, filters, 3, True)
    self.conv = nn.Linear(size, size)

  def forward(self, x):
    return F.relu(x + (self.conv(x)))


class Representation(nn.Module):
  ''' Conversion from observation to inner abstract state '''

  def __init__(self, input_shape):
    super().__init__()
    self.input_shape = input_shape
    # self.board_size = self.input_shape[1] * self.input_shape[2]

    # self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
    self.layer0 = nn.Linear(self.input_shape, 128)
    self.blocks = nn.ModuleList([ResidualBlock(size=128) for _ in range(num_blocks)])

  def forward(self, x):
    h = F.relu(self.layer0(x))
    for block in self.blocks:
      h = block(h)
    return h

  def inference(self, x):
    self.eval()
    with torch.no_grad():
      rp = self(torch.from_numpy(x).unsqueeze(0))
    return rp.cpu().numpy()[0]


class Prediction(nn.Module):
  ''' Policy and value prediction from inner abstract state '''

  def __init__(self, input_shape, action_shape):
    super().__init__()
    # self.board_size = np.prod(input_shape[1:])
    # self.action_size = input_shape[0] * self.board_size
    self.action_size = action_shape
    INNER = 256
    self.linear1_p = nn.Linear(input_shape, INNER)
    self.linear2_p = nn.Linear(INNER, INNER)
    self.linear3_p = nn.Linear(INNER, self.action_size)
    # self.conv_p1 = Conv(num_filters, 4, 1, bn=True)
    # self.conv_p2 = Conv(4, 1, 1)
    #
    # self.conv_v = Conv(num_filters, 4, 1, bn=True)
    # self.fc_v = nn.Linear(self.board_size * 4, 1, bias=False)
    self.linear1_v = nn.Linear(input_shape, INNER)
    self.linear2_v = nn.Linear(INNER, INNER)
    self.linear3_v = nn.Linear(INNER, 1, bias=False)

  def forward(self, rp):
    h_p = F.relu(self.linear1_p(rp))
    h_p = F.relu(self.linear2_p(h_p))
    # todo check if shape is okay here
    h_p = F.relu(self.linear3_p(h_p)).view(-1, self.action_size)
    return F.softmax(h_p, dim=-1), self.value(rp)

  def inference(self, rp):
    self.eval()
    with torch.no_grad():
      p, v = self(torch.from_numpy(rp).float().unsqueeze(0))
    return p.cpu().numpy()[0], v.cpu().numpy()[0][0]

  def value(self, rp):
    h_v = F.relu(self.linear1_v(rp))
    h_v = F.relu(self.linear2_v(h_v))
    h_v = self.linear3_v(h_v)
    # return h_v
    return torch.tanh(h_v)*3 # todo assumes everything is between -3 and 3

class Dynamics(nn.Module):
  '''Abstract state transition'''

  def __init__(self, rp_shape, act_shape):
    super().__init__()
    self.rp_shape = rp_shape
    # self.layer0 = Conv(rp_shape[0] + act_shape[0], num_filters, 3, bn=True)
    self.layer0 = nn.Linear(rp_shape + act_shape, 128)
    # self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])
    self.blocks = nn.ModuleList([ResidualBlock(128) for _ in range(num_blocks)])

  def forward(self, rp, a):
    h = torch.cat([rp, a], dim=1)
    h = self.layer0(h)
    for block in self.blocks:
      h = block(h)
    return h

  def inference(self, rp, a):
    self.eval()
    with torch.no_grad():
      rp = self(torch.from_numpy(rp).unsqueeze(0), torch.from_numpy(a).unsqueeze(0))
    return rp.cpu().numpy()[0]


class Net(nn.Module):
  '''Whole net'''

  def __init__(self):
    super().__init__()
    # input_shape = State.feature_shape()
    input_shape = State.feature_shape()[0]
    action_output_shape = State.action_length()
    # rp_shape = (num_filters, *input_shape[1:]) # for now no dynamics
    inner_state_size = 128
    self.representation = Representation(input_shape)
    self.prediction = Prediction(inner_state_size, action_output_shape)
    self.dynamics = Dynamics(inner_state_size, action_output_shape)

  def predict(self, state0, path):
    '''Predict p and v from original state and path'''
    outputs = []
    x = state0.feature()
    rp = self.representation.inference(x)
    outputs.append(self.prediction.inference(rp))
    for action in path:
      a = state0.action_feature(action)
      rp = self.dynamics.inference(rp, a)
      outputs.append(self.prediction.inference(rp))
    return outputs


class State:

  def __init__(self, ast_str):
    self.ast_str = ast_str
    self.original_lin = ast_str_to_lin(ast_str)  # debug single ast
    self.tm = self.last_tm = self.base_tm = None
    # self.rawbufs = bufs_from_lin(self.original_lin)
    # rawbufs = bufs_from_lin(self.original_lin)
    self.lin = deepcopy(self.original_lin)
    # print(self.state_lin)
    self.steps = 0
    self.terminal = False
    self._feature = None

  def step(self, act) -> 'State':
    # assert isinstance(act, int), f'act must be int, got {act, type(act)}'
    state = deepcopy(self)
    state._feature = None
    # state = State(ast_str=self.ast_str)
    # state.steps = self.steps
    # state.rawbufs = self.rawbufs
    # state.state_lin = deepcopy(self.state_lin)
    # state.tm,state.last_tm,state.base_tm = self.tm,self.last_tm,self.base_tm

    state.steps += 1
    state.terminal = state.steps == MAX_DIMS - 1 or act == 0
    if act == 0:
      return state
    state.lin.apply_opt(search.actions[act - 1])
    # state.state_lin =
    return state

  def terminal_reward(self):
    try:
      rawbufs = bufs_from_lin(self.original_lin)
      if self.base_tm is None:
        self.tm = self.last_tm = self.base_tm = time_linearizer(self.original_lin, rawbufs)
        assert not math.isinf(self.tm)
      tm = time_linearizer(self.lin, rawbufs)
      assert not math.isinf(tm)
      # time_penalty = ((self.last_tm - tm) / self.base_tm)
      reward = ((self.last_tm - tm) / self.base_tm)
      # time_penalty = (self.last_tm - tm)
      self.last_tm = tm
    except AssertionError as e:
      print(e)
      reward = -0.5

    return reward

  def feature(self):

    # if self._feature is None:
    #   self._feature = lin_to_feats(self.lin)
    return lin_to_feats(self.lin)

  @staticmethod
  def feature_shape():
    return (1021,)  # could also be 888 with sts

  @staticmethod
  def action_feature():
    return (1021,)  # could also be 888 with sts

  @staticmethod
  def action_length():
    return len(search.actions) + 1

  def get_masked_probs(self, p_root):
    # mask valid actions
    # probs = net(Tensor([feat])).exp()[0].numpy()
    valid_action_mask = np.zeros(self.action_length(), dtype=np.float32)
    for x in get_linearizer_actions(self.lin): valid_action_mask[x] = 1
    p_root *= valid_action_mask
    p_root /= sum(p_root)
    return p_root

  def get_valid_action(self, probs):
    probs = deepcopy(probs)

    for j in range(len(probs)):
      act = np.random.choice(len(probs), p=probs)
      try:
        lin = self.step(act).lin
        up, lcl = 1, 1
        try:
          for s, c in zip(lin.full_shape, lin.colors()):
            if c in {"magenta", "yellow"}: up *= s
            if c in {"cyan", "green", "white"}: lcl *= s
          if up <= 256 and lcl <= 256:
            return act
        except Exception as e:
          pass
          # print("exception at step", e)
      except AssertionError as e:
        pass
      probs[act] = 0
      _sum = probs.sum()
      assert _sum > 0., f'{j, len(probs)}'
      probs = probs / _sum