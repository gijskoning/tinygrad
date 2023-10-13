# Small neural nets with PyTorch
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from extra.optimization.helpers import MAX_DIMS, ast_str_to_lin, lin_to_feats
from tinygrad.codegen.search import ACTIONS, bufs_from_lin, get_linearizer_actions, time_linearizer

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
  def __init__(self, filters):
    super().__init__()
    self.conv = Conv(filters, filters, 3, True)

  def forward(self, x):
    return F.relu(x + (self.conv(x)))


class Representation(nn.Module):
  ''' Conversion from observation to inner abstract state '''

  def __init__(self, input_shape):
    super().__init__()
    self.input_shape = input_shape
    # self.board_size = self.input_shape[1] * self.input_shape[2]

    self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
    self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

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
    self.linear1_p = nn.Linear(input_shape[0], INNER)
    self.linear2_p = nn.Linear(INNER, self.action_size)
    # self.conv_p1 = Conv(num_filters, 4, 1, bn=True)
    # self.conv_p2 = Conv(4, 1, 1)
    #
    # self.conv_v = Conv(num_filters, 4, 1, bn=True)
    # self.fc_v = nn.Linear(self.board_size * 4, 1, bias=False)
    self.linear1_v = nn.Linear(input_shape[0], INNER)
    self.linear2_v = nn.Linear(INNER, 1, bias=False)

  def forward(self, rp):
    h_p = F.relu(self.linear1_p(rp))
    # todo check if shape is okay here
    h_p = F.relu(self.linear2_p(h_p)).view(-1, self.action_size)
    # h_p = F.relu(self.conv_p1(rp))
    # h_p = self.conv_p2(h_p).view(-1, self.action_size)
    #
    # h_v = F.relu(self.conv_v(rp))
    # h_v = self.fc_v(h_v.view(-1, self.board_size * 4))
    h_v = F.relu(self.linear1_v(rp))
    h_v = self.linear2_v(h_v)
    # range of value is -1 ~ 1
    return F.softmax(h_p, dim=-1), torch.tanh(h_v)

  def inference(self, rp):
    self.eval()
    with torch.no_grad():
      p, v = self(torch.from_numpy(rp).float().unsqueeze(0))
    return p.cpu().numpy()[0], v.cpu().numpy()[0][0]


class Dynamics(nn.Module):
  '''Abstract state transition'''

  def __init__(self, rp_shape, act_shape):
    super().__init__()
    self.rp_shape = rp_shape
    self.layer0 = Conv(rp_shape[0] + act_shape[0], num_filters, 3, bn=True)
    self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

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
    # state = State()
    input_shape = State.feature_shape()
    action_input_shape = State.action_feature()
    action_output_shape = State.action_length()
    # rp_shape = (num_filters, *input_shape[1:]) # for now no dynamics

    # self.representation = Representation(input_shape) # todo skip for now
    self.prediction = Prediction(action_input_shape, action_output_shape)
    # self.dynamics = Dynamics(rp_shape, action_shape)

  # def predict(self, state0, path):
  #   '''Predict p and v from original state and path'''
  #   outputs = []
  #   x = state0.feature()
  #   rp = self.representation.inference(x)
  #   outputs.append(self.prediction.inference(rp))
  #   for action in path:
  #     a = state0.action_feature(action)
  #     rp = self.dynamics.inference(rp, a)
  #     outputs.append(self.prediction.inference(rp))
  #   return outputs


class State:

  def __init__(self, ast_str):
    self.orginal_lin = ast_str_to_lin(ast_str)  # debug single ast

    rawbufs = bufs_from_lin(self.orginal_lin)
    self.tm = self.last_tm = self.base_tm = time_linearizer(self.orginal_lin, rawbufs)
    self.state_lin = deepcopy(self.orginal_lin)
    # print(self.state_lin)
    self.steps = 0
    self.terminal = False

  def step(self, act) -> 'State':
    # assert isinstance(act, int), f'act must be int, got {act, type(act)}'
    state = deepcopy(self)
    state.steps += 1
    state.terminal = state.steps == MAX_DIMS - 1 or act == 0
    if act == 0:
      return state
    state.state_lin.apply_opt(ACTIONS[act - 1])
    # state.state_lin =
    return state

  def terminal_reward(self):
    try:
      rawbufs = bufs_from_lin(self.orginal_lin)

      # lin.apply_opt(ACTIONS[act - 1])
      tm = time_linearizer(self.state_lin, rawbufs)
      if math.isinf(tm): raise Exception("failed")
      # rews.append(((self.last_tm - self.tm) / self.base_tm))
      reward = ((self.last_tm - tm) / self.base_tm)
      self.last_tm = tm
    except Exception:
      reward = -0.5
    return reward

  def feature(self):
    return lin_to_feats(self.state_lin)

  @staticmethod
  def feature_shape():
    return (888,)  # could also be 888 with sts

  @staticmethod
  def action_feature():
    return (888,)  # could also be 888 with sts

  @staticmethod
  def action_length():
    return len(ACTIONS) + 1

  def get_masked_probs(self, p_root):
    # mask valid actions
    # probs = net(Tensor([feat])).exp()[0].numpy()
    valid_action_mask = np.zeros(self.action_length(), dtype=np.float32)
    for x in get_linearizer_actions(self.state_lin): valid_action_mask[x] = 1
    p_root *= valid_action_mask
    p_root /= sum(p_root)
    return p_root