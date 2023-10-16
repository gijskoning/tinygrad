import functools
import os
import shelve
import time
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from extra.optimization.attention import SelfAttention

os.environ["GPU"] = '1'

import math
from copy import deepcopy
from typing import Tuple
# import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='raise')
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
# adapted from https://github.com/Itomigna2/Muesli-lunarlander/blob/main/Muesli_LunarLander.ipynb
# ! pip install git+https://github.com/cmpark0126/pytorch-polynomial-lr-decay.git
from torch_poly_lr_decay import PolynomialLRDecay

from extra.optimization.helpers import MAX_DIMS, ast_str_to_lin, lin_to_feats, load_worlds
from tinygrad.codegen import search
from tinygrad.codegen.search import beam_search, bufs_from_lin, time_linearizer

@functools.lru_cache(None)
def get_tm_cached(ast_str, handcoded=True):
  lin = ast_str_to_lin(ast_str)
  rawbufs = bufs_from_lin(lin)
  tm = time_linearizer(lin, rawbufs)
  if handcoded:
    lin.hand_coded_optimizations()
    tmhc = time_linearizer(lin, rawbufs)
    return tm, tmhc
  return tm

class State:

  def __init__(self, ast_strs, ast_num):
    self.ast_str = ast_strs[ast_num]
    self.original_lin = ast_str_to_lin(self.ast_str)  # debug single ast
    self.tm = self.last_tm = self.base_tm = None
    self.handcoded_tm = None
    self.lin = deepcopy(self.original_lin)
    self.steps = 0
    self.terminal = False
    self._feature = None
    self.failed = False
    self._reward = None

  def step(self, act, dense_reward=True) -> Tuple['State', np.ndarray,float,bool, None]:
    state = deepcopy(self)
    state._feature = None

    state.steps += 1
    state.terminal = state.steps == MAX_DIMS - 1 or act == 0
    info = None
    if state.terminal:
      state.failed, r = state.reward()
      return state, state.feature(), r, state.terminal, info
    state.lin.apply_opt(search.actions[act - 1])
    state._reward = None

    if dense_reward:
      state.failed, r = state.reward()
    else:
      state.failed, r = False, 0.
    state.terminal = state.terminal and not state.failed
    return state, state.feature(), r, state.terminal, info

  def _step(self, act):
    state = deepcopy(self)
    state.lin.apply_opt(search.actions[act - 1])
    state._reward = None
    return state

  def set_base_tm(self):
    if self.base_tm is None:
      tm, self.handcoded_tm = get_tm_cached(self.ast_str)
      self.last_tm = self.base_tm = tm
      assert not math.isinf(self.last_tm)
      if math.isinf(self.handcoded_tm):
        print("HANDCODED ERROR")
    return self.base_tm
  def reward(self):
    failed = False
    assert self.base_tm is not None
    try:
      rawbufs = bufs_from_lin(self.original_lin)
      # if self.base_tm is None:
      #   self.last_tm = self.base_tm = time_linearizer(self.original_lin, rawbufs)
      #   assert not math.isinf(self.tm)
      if self._reward is None:
        tm = time_linearizer(self.lin, rawbufs)
        assert not math.isinf(tm)
        self._reward = (self.last_tm - tm)
        self.last_tm = tm
    except AssertionError as e:
      self._reward = -self.base_tm
      failed = True

    return failed, self._reward

  def feature(self):
    assert self.base_tm is not None
    tm_reward = 0.
    if self._reward is not None:
      tm_reward = self._reward
    feats = lin_to_feats(self.lin)+[tm_reward, self.base_tm]
    return np.asarray(feats).astype(np.float32)

  @staticmethod
  def feature_shape():
    # +1 for reward
    return (1021+1+1,)

  @staticmethod
  def action_length():
    return len(search.actions) + 1

  def get_valid_action(self, probs, select_best=False):
    probs = deepcopy(probs)

    for j in range(len(probs)):
      if select_best:
        act = np.argmax(probs)
      else:
        act = np.random.choice(len(probs), p=probs)
      if act == 0:
        return act
      try:
        lin = self._step(act).lin
        try:
          lin_to_feats(lin) # todo this is temp, it can sometimes raise an error
        except IndexError as e:
          print('index error', e)
        up, lcl = 1, 1
        try:
          for s, c in zip(lin.full_shape, lin.colors()):
            if c in {"magenta", "yellow"}: up *= s
            if c in {"cyan", "green", "white"}: lcl *= s
          if up <= 256 and lcl <= 256:
            return act
        except (AssertionError, IndexError) as e:
          print('asset or index error', e)
      except AssertionError as e:
        pass
        # print("exception at step", e)

      probs[act] = 0
      _sum = probs.sum()
      assert _sum > 0., f'{j, len(probs)}'
      probs = probs / _sum

def get_sinusoid_pos_encoding(total_len, embed_dim):
    """
    Standard sinusoid positional encoding method outlined in the original
    Transformer paper. In this case, we use the encodings not to represent
    each token's position in a sequence but to represent the distance
    between two tokens (i.e. as a *relative* positional encoding).
    """
    pos = torch.arange(total_len).unsqueeze(1)
    enc = torch.arange(embed_dim).float()
    enc = enc.unsqueeze(0).repeat(total_len, 1)
    enc[:, ::2] = torch.sin(pos / 10000**(2*enc[:, ::2]/embed_dim))
    enc[:, 1::2] = torch.cos(pos / 10000**(2*enc[:, 1::2]/embed_dim))
    return enc

class Representation(nn.Module):
  """Representation Network

  Representation network produces hidden state from observations.
  Hidden state scaled within the bounds of [-1,1].
  Simple mlp network used with 1 skip connection.

  input : raw input
  output : hs(hidden state)
  """

  def __init__(self, input_dim, output_dim, width):
    super().__init__()
    self.input_dim = input_dim

    self.total_len = MAX_DIMS
    self.embed_dim = 512
    self.pos_dim = 32

    self.embed_obs = torch.nn.Linear(input_dim+self.pos_dim, self.embed_dim) # might want to add selfattention here
    R = get_sinusoid_pos_encoding(self.total_len, self.pos_dim)
    self.R = torch.flip(R, dims=(0,)).to(device)
    num_heads = 4
    self.w_r = nn.Linear(self.pos_dim, self.pos_dim, bias=False)
    self.transformer_model = SelfAttention(num_heads, embed_dimension=self.embed_dim, dropout=0.1)
    self.skip = torch.nn.Linear(input_dim, output_dim)
    self.layer1 = torch.nn.Linear(self.embed_dim, width)
    self.layer2 = torch.nn.Linear(width, width)
    # self.layer3 = torch.nn.Linear(width, width)
    # self.layer4 = torch.nn.Linear(width, width)
    self.layer5 = torch.nn.Linear(width, output_dim)

  def forward(self, hist_x, end_pos=None):
    # hist_x is sorted such that the first element is the oldest
    # x: stacked obs
    hist_x = hist_x.reshape(-1, num_state_stack, self.input_dim) # batch, num_seqs, dim
    # hist_x = hist_x[-steps:] # the history was stacked first. But we don't want the stacked frames. HACKKYY
    # could try to just use all hist??
    batch_num = hist_x.shape[0]
    # position embedding:
    full_r = self.w_r(self.R).expand(batch_num, self.total_len, self.pos_dim)
    r = full_r # this probably has to change with bigger length
    # r = torch.zeros_like(full_r)
    # for i in range(batch_num):
    #   r[i, -steps[i]:] = full_r[i,:steps[i]]# full_r is the pos embedding from 0 to seq_len. However since we mask it partially we need to move it
    embed_x = self.embed_obs(torch.cat((hist_x, r), dim=-1)) # could also sum instead of cat (see gpt2.py)
    # start_pos = self.seq_len - steps
    mask = torch.full((batch_num, 1, self.total_len, self.total_len), float("-inf"), device=device)
    for i in range(batch_num):
      mask[i] = mask[i].triu(end_pos[i])

    out_transformer = self.transformer_model(embed_x, mask=mask)
    out_transformer = out_transformer[:, -1] # get last output
    # do we need softmax??
    last_obs = hist_x[:,0]
    s = self.skip(last_obs)
    x = self.layer1(out_transformer)

    x = torch.nn.functional.relu(x)
    x = self.layer2(x)
    x = torch.nn.functional.relu(x)
    # x = self.layer3(x)
    # x = torch.nn.functional.relu(x)
    # x = self.layer4(x)
    # x = torch.nn.functional.relu(x)
    x = self.layer5(x)
    x = torch.nn.functional.relu(x + s)
    assert torch.isfinite(x).all()
    x = 2 * (x - x.min(-1, keepdim=True)[0]) / (x.max(-1, keepdim=True)[0] - x.min(-1, keepdim=True)[0]) - 1
    assert torch.isfinite(x).all()
    return x.squeeze()


class Dynamics(nn.Module):
  """Dynamics Network

  Dynamics network transits (hidden state + action) to next hidden state and inferences reward model.
  Hidden state scaled within the bounds of [-1,1]. Action encoded to one-hot representation.
  Zeros tensor is used for action -1.

  Output of the reward head is categorical representation, instaed of scalar value.
  Categorical output will be converted to scalar value with 'to_scalar()',and when
  traning target value will be converted to categorical target with 'to_cr()'.

  input : hs, action
  output : next_hs, reward
  """

  def __init__(self, input_dim, output_dim, width, action_space):
    super().__init__()
    self.layer1 = torch.nn.Linear(input_dim + action_space, width)
    self.layer2 = torch.nn.Linear(width, width)
    self.hs_head = torch.nn.Linear(width, output_dim)
    self.reward_head = nn.Sequential(
      nn.Linear(width, width),
      nn.ReLU(),
      nn.Linear(width, width),
      nn.ReLU(),
      nn.Linear(width, support_size * 2 + 1)
    )
    self.one_hot_act = torch.cat((F.one_hot(torch.arange(0, action_space) % action_space, num_classes=action_space),
                                  torch.zeros(action_space).unsqueeze(0)),
                                 dim=0).to(device)

  def forward(self, x, action):
    action = self.one_hot_act[action.squeeze(1)]
    x = torch.cat((x, action.to(device)), dim=1)
    x = self.layer1(x)
    x = torch.nn.functional.relu(x)
    x = self.layer2(x)
    x = torch.nn.functional.relu(x)
    hs = self.hs_head(x)
    hs = torch.nn.functional.relu(hs)
    reward = self.reward_head(x)
    hs = 2 * (hs - hs.min(-1, keepdim=True)[0]) / (hs.max(-1, keepdim=True)[0] - hs.min(-1, keepdim=True)[0]) - 1
    return hs, reward


class Prediction(nn.Module):
  """Prediction Network

  Prediction network inferences probability distribution of policy and value model from hidden state.

  Output of the value head is categorical representation, instaed of scalar value.
  Categorical output will be converted to scalar value with 'to_scalar()',and when
  traning target value will be converted to categorical target with 'to_cr()'.

  input : hs
  output : P, V
  """

  def __init__(self, input_dim, output_dim, width):
    super().__init__()
    self.layer1 = torch.nn.Linear(input_dim, width)
    self.layer2 = torch.nn.Linear(width, width)
    self.policy_head = nn.Sequential(
      nn.Linear(width, width),
      nn.ReLU(),
      nn.Linear(width, width),
      nn.ReLU(),
      nn.Linear(width, output_dim)
    )
    self.value_head = nn.Sequential(
      nn.Linear(width, width),
      nn.ReLU(),
      nn.Linear(width, width),
      nn.ReLU(),
      nn.Linear(width, support_size * 2 + 1)
    )

  def forward(self, x):
    x = self.layer1(x)
    x = torch.nn.functional.relu(x)
    x = self.layer2(x)
    x = torch.nn.functional.relu(x)
    P = self.policy_head(x)
    P = torch.nn.functional.softmax(P, dim=-1)
    V = self.value_head(x)
    return P, V


"""
For categorical representation
reference : https://github.com/werner-duvaud/muzero-general
In my opinion, support size have to cover the range of maximum absolute value of 
reward and value of entire trajectories. Support_size 30 can cover almost [-900,900].
"""
support_size = 30
eps = 0.001


def to_scalar(x):
  x = torch.softmax(x, dim=-1)
  probabilities = x
  support = (torch.tensor([x for x in range(-support_size, support_size + 1)]).expand(probabilities.shape).float().to(device))
  x = torch.sum(support * probabilities, dim=1, keepdim=True)
  scalar = torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)
  return scalar


def to_scalar_no_soft(x):  ## test purpose
  probabilities = x
  support = (torch.tensor([x for x in range(-support_size, support_size + 1)]).expand(probabilities.shape).float().to(device))
  x = torch.sum(support * probabilities, dim=1, keepdim=True)
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
  logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).type(torch.float32).to(device)
  logits.scatter_(2, floor_index.long().unsqueeze(-1), floor_prob.unsqueeze(-1))
  under_prob = under_prob.masked_fill_(2 * support_size < under_index, 0.0)
  under_index = under_index.masked_fill_(2 * support_size < under_index, 0.0)
  logits.scatter_(2, under_index.long().unsqueeze(-1), under_prob.unsqueeze(-1))
  return logits.squeeze(0)


##Target network
class Target(nn.Module):
  """Target Network

  Target network is used to approximate v_pi_prior, q_pi_prior, pi_prior.
  It contains older network parameters. (exponential moving average update)
  """

  def __init__(self, state_dim, action_dim, width):
    super().__init__()
    # self.representation_network = Representation(state_dim * num_state_stack, state_dim * 4, width)
    self.representation_network = Representation(state_dim, state_dim * 4, width)
    self.dynamics_network = Dynamics(state_dim * 4, state_dim * 4, width, action_dim)
    self.prediction_network = Prediction(state_dim * 4, action_dim, width)
    self.to(device)


##Muesli agent
class Agent(nn.Module):
  """Agent Class"""

  def __init__(self, state_dim, action_dim, width):
    super().__init__()
    self.representation_network = Representation(state_dim, state_dim * 4, width)
    self.dynamics_network = Dynamics(state_dim * 4, state_dim * 4, width, action_dim)
    self.prediction_network = Prediction(state_dim * 4, action_dim, width)
    self.optimizer = torch.optim.AdamW(self.parameters(), lr=3e-3, weight_decay=1e-4)
    self.scheduler = PolynomialLRDecay(self.optimizer, max_decay_steps=episode_nums, end_learning_rate=0.0000)
    self.to(device)

    self.state_replay = []
    self.action_replay = []
    self.P_replay = []
    self.r_replay = []
    self.ast_strs = load_worlds()

    self.action_space = action_dim
    self.env = None

    self.var = 0
    self.beta_product = 1.0

    self.var_m = [0 for _ in range(unroll_steps)]
    self.beta_product_m = [1.0 for _ in range(unroll_steps)]

  def self_play_mu(self, target:Target, ast_num=None, eval=False, max_timestep=10000):
    """Self-play and save trajectory to replay buffer

    (Originally target network used to inference policy, but i used agent network instead) # todo

    Eight previous observations stacked -> representation network -> prediction network
    -> sampling action follow policy -> next env step
    """
    # state = self.env#.reset()no reset for now
    if ast_num is None:
      ast_num = np.random.choice(first_ast_nums)
    # print('running astnum', ast_num)
    self.env = State(self.ast_strs, ast_num)
    base_tm = self.env.set_base_tm()
    state = self.env.feature()
    state_dim = len(state)
    state_traj, action_traj, P_traj, r_traj = [], [], [], []
    for i in range(MAX_DIMS):
      start_state = state
      if num_state_stack == 1:
        stacked_state = state
      else:
        if i == 0:
          stacked_state = np.concatenate(tuple(state for _ in range(num_state_stack)), axis=0) # for now just stack 2
        else:
          # stacked_state = np.roll(stacked_state, state_dim, axis=0)
          stacked_state[state_dim*i:state_dim*(i+1)] = state
      # it combines all observations into one transformer model I think one could see it as the representation network
      # to add to observation
      # https://arxiv.org/pdf/2301.07608.pdf
      # name, dim
      # TRIALS REMAINING 5 # one hot. so max of 5 trials for example
      # reward previous step 1
      # should add:
      # basetiming to state
      with torch.no_grad():
        # st = time.perf_counter()
        steps = i+1
        hs = target.representation_network(torch.from_numpy(stacked_state).float().to(device), end_pos=np.array([steps]))
        # print('time for representation', time.perf_counter() - st)
        P, v = target.prediction_network(hs)
      probs = P.detach().cpu().numpy()
      action = self.env.get_valid_action(probs, select_best=eval)
      # action = np.random.choice(np.arange(self.action_space), p=P.detach().cpu().numpy())
      self.env, state, r, done, info = self.env.step(action, dense_reward=not eval)
      if i == 0:
        for _ in range(num_state_stack):
          state_traj.append(start_state)
      else:
        state_traj.append(start_state)
      action_traj.append(action)
      P_traj.append(P.cpu().numpy())
      r_traj.append(r)
      # print('action', action,'step reward', r, 'done', done)
      # game_score += r

      ## For fix lunarlander-v2 env does not return reward -100 when 'TimeLimit.truncated'
      if done:
        # if (info['TimeLimit.truncated'] == True) and abs(r) != 100:
        #   game_score -= 100
        #   self.r_traj[-1] = -100
        #   r = -100
        last_frame = i
        break

    # print('self_play: score, r, done, info, lastframe', int(game_score), r, done, info, i)

    # for update inference over trajectory length
    for _ in range(unroll_steps):
      state_traj.append(np.zeros_like(state))

    for _ in range(unroll_steps+1):
      r_traj.append(0.0)
      action_traj.append(-1)
    if not eval:
      # traj append to replay
      self.state_replay.append(state_traj)
      self.action_replay.append(action_traj)
      self.P_replay.append(P_traj)
      self.r_replay.append(r_traj)

    game_score = (base_tm - self.env.last_tm) / base_tm
    base_to_handcoded = (base_tm - self.env.handcoded_tm) / base_tm
    base_to_beam = (base_tm - beam_results[ast_num]) / base_tm
    return game_score, base_to_handcoded, base_to_beam, r, last_frame, ast_num

  def update_weights_mu(self, target):
    """Optimize network weights.

    Iteration: 20
    Mini-batch size: 16 (4 replay, 4 sequences in 1 replay)
    Replay: Uniform replay with on-policy data
    Discount: 0.997
    Unroll: 5 step
    L_m: 5 step(Muesli)
    Observations: Stack 8 frame
    regularizer_multiplier: 5
    Loss: L_pg_cmpo + L_v/6/4 + L_r/5/1 + L_m
    """
    for _ in range(train_updates):
      state_traj_i = []
      state_traj = []
      action_traj = []
      P_traj = []
      r_traj = []
      G_arr_mb = []

      for epi_sel in range(num_replays): # this is q
        if (epi_sel > 0):  ## replay proportion
          sel = np.random.randint(0, max(1,len(self.state_replay)-1))
        else:
          sel = -1

        ## multi step return G (orignally retrace used)
        G = 0
        G_arr = []
        for r in self.r_replay[sel][::-1]:
          G = 0.997 * G + r
          G_arr.append(G)
        G_arr.reverse()
        for i in np.random.randint(len(self.state_replay[sel]) - unroll_steps - (num_state_stack - 1), size=seq_in_replay):
          state_traj_i.append(i)
          state_traj.append(self.state_replay[sel][i:i + unroll_steps + num_state_stack])
          action_traj.append(self.action_replay[sel][i:i + unroll_steps])
          r_traj.append(self.r_replay[sel][i:i + unroll_steps])
          G_arr_mb.append(G_arr[i:i + unroll_steps + 1])
          P_traj.append(self.P_replay[sel][i])
      state_traj_i = np.array(state_traj_i)
      state_traj = torch.from_numpy(np.array(state_traj)).to(device)
      action_traj = torch.from_numpy(np.array(action_traj)).unsqueeze(2).to(device)
      P_traj = torch.from_numpy(np.array(P_traj)).to(device)
      G_arr_mb = torch.from_numpy(np.array(G_arr_mb)).unsqueeze(2).float().to(device)
      r_traj = torch.from_numpy(np.array(r_traj)).unsqueeze(2).float().to(device)
      inferenced_P_arr = []

      ## stacking
      stacked_state_0 = torch.cat(tuple(state_traj[:, i] for i in range(num_state_stack)), dim=1)

      ## agent network inference (5 step unroll)
      hs_s = []
      v_logits_s = []
      r_logits_s = []
      for i in range(1+unroll_steps):
        if i == 0:
          hs = self.representation_network(stacked_state_0, end_pos=state_traj_i+1)
        else:
          hs, r_logits = self.dynamics_network(hs, action_traj[:, i-1])
          r_logits_s.append(r_logits)

        P, v_logits = self.prediction_network(hs)
        hs_s.append(hs)
        v_logits_s.append(v_logits)
        inferenced_P_arr.append(P)
      first_P = inferenced_P_arr[0]

      ## target network inference
      with torch.no_grad():
        t_first_hs = target.representation_network(stacked_state_0, end_pos=state_traj_i+1)
        t_first_P, t_first_v_logits = target.prediction_network(t_first_hs)

        ## normalized advantage
      beta_var = 0.99
      self.var = beta_var * self.var + (1 - beta_var) * (torch.sum((G_arr_mb[:, 0] - to_scalar(t_first_v_logits)) ** 2) / minibatch_size)
      self.beta_product *= beta_var
      var_hat = self.var / (1 - self.beta_product)
      under = torch.sqrt(var_hat + 1e-12)

      ## L_pg_cmpo first term (eq.10)
      importance_weight = torch.clip(first_P.gather(1, action_traj[:, 0])
                                     / (P_traj.gather(1, action_traj[:, 0])),
                                     0, 1
                                     )
      first_term = -1 * importance_weight * (G_arr_mb[:, 0] - to_scalar(t_first_v_logits)) / under

      ## eq. 14. Lookahead inferences (one step look-ahead to some actions to estimate q_prior, from target network)
      with torch.no_grad():
        r1_arr = []
        v1_arr = []
        a1_arr = []
        for _ in range(lookahead_samples):  # sample <= N(action space), now N
          action1_stack = []
          for p in t_first_P:
            action1_stack.append(np.random.choice(np.arange(self.action_space), p=p.detach().cpu().numpy()))
          hs, r1 = target.dynamics_network(t_first_hs, torch.unsqueeze(torch.tensor(action1_stack), 1))
          _, v1 = target.prediction_network(hs)

          r1_arr.append(to_scalar(r1))
          v1_arr.append(to_scalar(v1))
          a1_arr.append(torch.tensor(action1_stack))

      ## z_cmpo_arr (eq.12)
      with torch.no_grad():
        exp_clip_adv_arr = [torch.exp(torch.clip((r1_arr[k] + 0.997 * v1_arr[k] - to_scalar(t_first_v_logits)) / under, -1, 1)) # -c, c
                            .tolist() for k in range(lookahead_samples)]
        exp_clip_adv_arr = torch.tensor(exp_clip_adv_arr).to(device)
        z_cmpo_arr = []
        for k in range(lookahead_samples): # k != i is fixed by "- exp_clip_adv_arr[k]"
          z_cmpo = (1 + torch.sum(exp_clip_adv_arr[k], dim=0) - exp_clip_adv_arr[k]) / lookahead_samples
          z_cmpo_arr.append(z_cmpo.tolist())
      z_cmpo_arr = torch.tensor(z_cmpo_arr).to(device)

      ## L_pg_cmpo second term (eq.11)
      second_term = 0
      for k in range(lookahead_samples):
        second_term += exp_clip_adv_arr[k] / z_cmpo_arr[k] * torch.log(first_P.gather(1, torch.unsqueeze(a1_arr[k], 1).to(device)))
      second_term *= -1 * regularizer_multiplier / lookahead_samples

      ## L_pg_cmpo
      L_pg_cmpo = first_term + second_term

      ## L_m eq. 13
      L_m = 0
      for i in range(unroll_steps):
        stacked_state = torch.cat(tuple(state_traj[:, i + j+1] for j in range(num_state_stack)), dim=1)
        with torch.no_grad():
          t_hs = target.representation_network(stacked_state, end_pos=state_traj_i+1+i)
          t_P, t_v_logits = target.prediction_network(t_hs)

        beta_var = 0.99
        self.var_m[i] = beta_var * self.var_m[i] + (1 - beta_var) * (torch.sum((G_arr_mb[:, i + 1] - to_scalar(t_v_logits)) ** 2) / minibatch_size)
        self.beta_product_m[i] *= beta_var
        var_hat = self.var_m[i] / (1 - self.beta_product_m[i])
        under = torch.sqrt(var_hat + 1e-12)

        with torch.no_grad():
          r1_arr = []
          v1_arr = []
          a1_arr = []
          for j in range(lookahead_samples2):
            action1_stack = []
            for p in t_P:
              action1_stack.append(np.random.choice(np.arange(self.action_space), p=p.detach().cpu().numpy()))
            hs, r1 = target.dynamics_network(t_hs, torch.unsqueeze(torch.tensor(action1_stack), 1))
            _, v1 = target.prediction_network(hs)

            r1_arr.append(to_scalar(r1))
            v1_arr.append(to_scalar(v1))
            a1_arr.append(torch.tensor(action1_stack))

        with torch.no_grad():
          exp_clip_adv_arr = [torch.exp(torch.clip((r1_arr[k] + 0.997 * v1_arr[k] - to_scalar(t_v_logits)) / under, -1, 1))
                              .tolist() for k in range(lookahead_samples2)]
          exp_clip_adv_arr = torch.tensor(exp_clip_adv_arr).to(device)

        ## Paper appendix F.2 : Prior policy
        t_P = 0.967 * t_P + 0.03 * P_traj + 0.003 * (torch.ones((minibatch_size,self.env.action_length()))/self.env.action_length()).to(device)

        pi_cmpo_all = [(t_P.gather(1, torch.unsqueeze(a1_arr[k], 1).to(device))
                        * exp_clip_adv_arr[k])
                       .squeeze(-1).tolist() for k in range(lookahead_samples2)]

        pi_cmpo_all = torch.tensor(pi_cmpo_all).transpose(0, 1).to(device)
        pi_cmpo_all = pi_cmpo_all / torch.sum(pi_cmpo_all, dim=1).unsqueeze(-1)
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        # I think this is right. todo but not completely sure
        P_arr_selected_actions = torch.gather(inferenced_P_arr[i + 1], 1, torch.stack(a1_arr).T.to(device))
        L_m += kl_loss(torch.log(P_arr_selected_actions), pi_cmpo_all)

      L_m /= unroll_steps

      ## L_v
      ls = nn.LogSoftmax(dim=-1)
      L_v = 0
      for i in range(unroll_steps + 1): L_v += (to_cr(G_arr_mb[:, i]) * ls(v_logits_s[i])).sum(-1, keepdim=True)
      L_v = -1 * L_v
      ## L_r
      L_r = 0.
      for i in range(unroll_steps): L_r += (to_cr(r_traj[:, i]) * ls(r_logits_s[i])).sum(-1, keepdim=True)
      L_r = -1 * L_r
      ## start of dynamics network gradient *0.5
      for i in range(unroll_steps+1): hs_s[i].register_hook(lambda grad: grad * 0.5)

      ## total loss
      # todo not sure if 4 belongs to seq_in_replay,  or  num_replays
      L_total = L_pg_cmpo + L_v / (unroll_steps+1) / 4 + L_r / unroll_steps / 1 + L_m # todo check if unroll_steps is good here
      # todo divided by 1?
      ## optimize
      self.optimizer.zero_grad()
      L_total.mean().backward()
      nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)
      self.optimizer.step()

      ## target network(prior parameters) moving average update
      params1 = self.named_parameters()
      params2 = target.named_parameters()
      dict_params2 = dict(params2)
      for name1, param1 in params1:
        if name1 in dict_params2:
          dict_params2[name1].data.copy_(alpha_target * param1.data + (1 - alpha_target) * dict_params2[name1].data)
      target.load_state_dict(dict_params2)

    self.scheduler.step()

    writer.add_scalar('Loss/L_total', L_total.mean(), global_i)
    writer.add_scalar('Loss/L_pg_cmpo', L_pg_cmpo.mean(), global_i)
    writer.add_scalar('Loss/L_v', (L_v / (unroll_steps+1) / 4).mean(), global_i)
    writer.add_scalar('Loss/L_r', (L_r / unroll_steps / 1).mean(), global_i)
    writer.add_scalar('Loss/L_m', (L_m).mean(), global_i)
    writer.add_scalars('vars', {'self.var': self.var,
                                'self.var_m': self.var_m[0]
                                }, global_i)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
score_arr = []
score_arr_handcoded = []

num_replays = 64
seq_in_replay = 1
minibatch_size = num_replays * seq_in_replay
num_state_stack = MAX_DIMS
first_ast_nums = 2
train_updates = 30
episodes_per_train_update = 5
episodes_per_eval = 5
start_training_epoch = 5 # todo set higher
alpha_target = 0.05 # 5% new to target

regularizer_multiplier = 1 # was 5 but that seems very high

lookahead_samples = 16
lookahead_samples2 = 16 # this is in the unroll
unroll_steps = 3 # todo cannot be changed at the moment
ast_strs = load_worlds()
env = State(ast_strs, 0)
observation_space = env.feature_shape()
action_space = env.action_length()
episode_nums = 1000
timing_str = datetime.now().strftime("%Y%m%d-%H%M%S")
# exp_str = 'test_'+timing_str # 1,
exp_str = 'test_'+timing_str # 1,
del env
target = Target(observation_space[0], action_space, 256)
target.eval()
agent = Agent(observation_space[0], action_space, 256)
# target = Target(env.observation_space.shape[0], env.action_space.n, 128)
# agent = Agent(env.observation_space.shape[0], env.action_space.n, 128)
# print(agent)
# env.close()

## initialization
target.load_state_dict(agent.state_dict())
best_ast_num_scores = np.zeros(first_ast_nums)
handcoded_best_ast_num_scores = np.zeros(first_ast_nums)
global_db = shelve.open("/tmp/greedy_cache")

beam_results = []
beams = 8
# beam search 4 takes [39 and 5] for 2 ast nums
# beam search 8 takes [94,9,17, 91]


for i in range(first_ast_nums):
  st = time.perf_counter()
  lin = ast_str_to_lin(ast_strs[i])
  rawbufs = bufs_from_lin(lin)
  key = str((lin.ast, beams))
  if key in global_db:
    for ao in global_db[key]:
      lin.apply_opt(ao)
  else:
    print("running beam search for num", i)
    lin = beam_search(lin, rawbufs, beams)
    global_db[key] = lin.applied_opts
  tm = time_linearizer(lin, rawbufs, should_copy=False)
  beam_results.append(tm)
  print('beam result', tm, 'time', time.perf_counter() - st)
## Self play & Weight update loop
for i in range(episode_nums):
  writer = SummaryWriter(log_dir=f'scalar{exp_str}/')
  global_i = i
  game_score, base_to_handcoded, base_to_beam, last_r, frame, ast_num = agent.self_play_mu(target)
  more_than_handcoded = game_score - base_to_handcoded
  if best_ast_num_scores[ast_num] < game_score:
    best_ast_num_scores[ast_num] = game_score
  if handcoded_best_ast_num_scores[ast_num] == 0.:
    handcoded_best_ast_num_scores[ast_num] = base_to_handcoded

  print(f'steps done {frame}. relative speed score {game_score:.2f} relative % more than handcoded {more_than_handcoded:.2f} iteration {global_i}')
  # diff_best_handcoded = best_ast_num_scores - handcoded_best_ast_num_scores
  # print('Diff with best handcoded scores for each ast num', (diff_best_handcoded).round(2).tolist())
  # print('mean diff', diff_best_handcoded.mean().round(2).tolist())
  # writer.add_scalar('relative_to_base_score', game_score, global_i)
  # writer.add_scalar('more_than_handcoded', more_than_handcoded, global_i)
  # writer.add_scalar('diff_to_best_handcoded', diff_best_handcoded.mean(), global_i)

  score_arr.append(game_score)
  score_arr_handcoded.append(more_than_handcoded)

  if i % 100 == 0:
    torch.save(agent.state_dict(), f'weights_id{exp_str}.pt')

  # if mean_score > 250 and np.mean(np.array(score_arr[-5:])) > 250:
  #   torch.save(agent.state_dict(), f'weights_id{exp_str}.pt')
  #   print('Done')
  #   break
  if i % episodes_per_train_update == 0 and i >= start_training_epoch:
    print("Training update")
    agent.update_weights_mu(target)
    print("Done update")

  if i % episodes_per_eval == 0 and i >= start_training_epoch:
    print('best scores for each ast num', best_ast_num_scores.round(2).tolist())
    best_score_related_to_handcoded = (best_ast_num_scores-handcoded_best_ast_num_scores)[best_ast_num_scores != 0.]
    writer.add_scalar('best_found_score_related_to_handcoded', best_score_related_to_handcoded.mean(), global_i)
    mean_score = np.mean(np.array(score_arr[i - 30:i+1]))
    more_than_handcoded_mean = np.mean(np.array(score_arr_handcoded[i - 30:i+1]))
    print(f'episode {i}, avg {mean_score:.2f}, avg handcoded {more_than_handcoded_mean:.2f}')
    scores = []
    for i in range(first_ast_nums):
      game_score, base_to_handcoded, base_to_beam, _, frame, _ = agent.self_play_mu(target, ast_num=i, eval=True)
      more_than_handcoded = game_score - base_to_handcoded
      more_than_beam = game_score - base_to_beam

      scores.append([game_score, more_than_handcoded, base_to_beam])
    scores = np.array(scores)
    writer.add_scalar('eval/relative_to_base_score', scores[:,0].mean(), global_i)
    writer.add_scalar('eval/more_than_handcoded', scores[:,1].mean(), global_i)
    writer.add_scalar('eval/more_than_beam', scores[:,2].mean(), global_i)

    print('eval relative_to_base_score', scores[:,0])
    print('eval more_than_handcoded', scores[:,1])
    print('eval more_than_beam', scores[:,2])
  # writer.close()

torch.save(agent.state_dict(), f'weights_id{exp_str}.pt')
agent.env.close()
# python -m tensorboard.main --logdir optimization