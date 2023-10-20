import functools
import os
import random
import shelve
import time
from datetime import datetime

from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter

from extra.optimization.efficientzero_funcs import negative_cosine_similarity
from extra.optimization.mcts_ptree import EfficientZeroMCTSPtree, select_action
from extra.optimization.muesli_funcs import support_size, to_cr, to_scalar

os.environ["GPU"] = '1'
# os.environ["PYOPENCL_NO_CACHE"] = '1'

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
from tinygrad.codegen.search import beam_search, bufs_from_lin, get_linearizer_actions, time_linearizer


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
    self.trial = 0

  def step(self, act, dense_reward=True) -> Tuple['State', np.ndarray, float, bool, None]:
    state = deepcopy(self)
    state._feature = None

    state.steps += 1
    if act == 0:
      state.trial += 1
      state.terminal = state.trial == max_trials
      if not state.terminal:
        print('resetting trial to', state.trial + 1, 'reward was ', state._reward)
    state.terminal = state.terminal or state.steps == MAX_DIMS - 1
    if state.terminal:
      state.failed, r = False, 0.
      return state, state.feature(), r, state.terminal, None

    state._reward = None

    if act == 0:
      # new trial
      state.lin = deepcopy(state.original_lin)
      state.tm = state.last_tm = state.base_tm
      state.steps = 0
    else:
      state.lin.apply_opt(search.actions[act - 1])

    if dense_reward:
      state.failed, r = state.reward()
    else:  # todo not really using
      state.failed, r = False, 0.
    state.terminal = state.terminal or state.failed
    return state, state.feature(), r, state.terminal, None

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
      if self._reward is None:
        tm = time_linearizer(self.lin, rawbufs)
        assert not math.isinf(tm)
        # we clip the rewards to ensure stability. Note:
        self._reward = max((self.last_tm - tm) / self.base_tm, -5.)

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
    trial_onehot = [0] * max_trials
    if self.trial == max_trials:
      trial_onehot[self.trial - 1] = 1
    else:
      trial_onehot[self.trial] = 1
    feats = lin_to_feats(self.lin, use_sts=use_sts) + [tm_reward, self.base_tm] + trial_onehot
    return np.asarray(feats).astype(np.float32)

  @staticmethod
  def feature_shape():
    base = State.base_feature_shape()
    # +1 for reward, base_reward, and max_trial encoding
    return (base + 1 + 1 + max_trials,)

  @staticmethod
  def base_feature_shape():
    return 1019 + 32 if use_sts else 272 + 32

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
        return act, probs
      try:
        lin = self._step(act).lin
        try:
          lin_to_feats(lin, use_sts=use_sts)  # todo this is temp, it can sometimes raise an error
        except IndexError as e:
          print('index error', e)
        up, lcl = 1, 1
        try:
          for s, c in zip(lin.full_shape, lin.colors()):
            if c in {"magenta", "yellow"}: up *= s
            if c in {"cyan", "green", "white"}: lcl *= s
          if up <= 256 and lcl <= 256:
            return act, probs
        except (AssertionError, IndexError) as e:
          print('asset or index error', e)
      except AssertionError as e:
        pass
        # print("exception at step", e)

      probs[act] = 0
      _sum = probs.sum()
      if _sum == 0:
        # print("not good sum")
        return 0, probs

        # assert _sum > 0., f'{j, len(probs)}'
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
  enc[:, ::2] = torch.sin(pos / 10000 ** (2 * enc[:, ::2] / embed_dim))
  enc[:, 1::2] = torch.cos(pos / 10000 ** (2 * enc[:, 1::2] / embed_dim))
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
    self.skip = torch.nn.Linear(input_dim, output_dim)
    self.layer1 = torch.nn.Linear(input_dim, width)
    self.layer2 = torch.nn.Linear(width, width)
    self.layer3 = torch.nn.Linear(width, width)
    self.layer4 = torch.nn.Linear(width, width)
    self.layer5 = torch.nn.Linear(width, output_dim)

  def forward(self, x):
    s = self.skip(x)
    x = self.layer1(x)
    x = torch.nn.functional.relu(x)
    x = self.layer2(x)
    x = torch.nn.functional.relu(x)
    x = self.layer3(x)
    x = torch.nn.functional.relu(x)
    x = self.layer4(x)
    x = torch.nn.functional.relu(x)
    x = self.layer5(x)
    x = torch.nn.functional.relu(x + s)
    x = 2 * (x - x.min(-1, keepdim=True)[0]) / (x.max(-1, keepdim=True)[0] - x.min(-1, keepdim=True)[0]) - 1
    return x


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
    self.layer3 = torch.nn.Linear(width, width)
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
    x = torch.cat((x, action.to(device)), dim=-1)
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = F.relu(self.layer3(x))
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
    activation = nn.ReLU(inplace=True)

    self.layer1 = torch.nn.Linear(input_dim, width)
    self.layer2 = torch.nn.Linear(width, width)
    self.policy_head = nn.Sequential(
      nn.Linear(width, width * 2),
      activation,
      nn.Linear(width * 2, width * 2),
      activation,
      nn.Linear(width * 2, output_dim)
    )
    head_out = support_size * 2 + 1 if support_size != 1 else 1
    self.value_head = nn.Sequential(
      nn.Linear(width, width),
      activation,
      nn.Linear(width, width),
      activation,
      nn.Linear(width, width),
      activation,
      nn.Linear(width, head_out)
    )
    # from efficientnet
    self.projection_input_dim = input_dim
    self.pred_hid = 512
    self.proj_hid = 1024
    self.proj_out = 1024
    self.pred_out = 1024
    self.projection = nn.Sequential(
      nn.Linear(self.projection_input_dim, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
      nn.Linear(self.proj_hid, self.proj_hid), nn.BatchNorm1d(self.proj_hid), activation,
      nn.Linear(self.proj_hid, self.proj_out), nn.BatchNorm1d(self.proj_out)
    )
    self.prediction_head = nn.Sequential(
      nn.Linear(self.proj_out, self.pred_hid),
      nn.BatchNorm1d(self.pred_hid),
      activation,
      nn.Linear(self.pred_hid, self.pred_out),
    )

  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    P = self.policy_head(x)
    P = torch.nn.functional.softmax(P, dim=-1)
    V = self.value_head(x)
    return P, V


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


class Agent(nn.Module):
  """Agent Class"""

  def __init__(self, state_dim, action_dim, width):
    super().__init__()
    self.representation_network = Representation(state_dim, state_dim * 4, width)
    self.dynamics_network = Dynamics(state_dim * 4, state_dim * 4, width, action_dim)  # todo change to lstm
    self.prediction_network = Prediction(state_dim * 4, action_dim, width)
    self.optimizer = torch.optim.AdamW(self.parameters(), lr=3e-3, weight_decay=1e-4)
    self.scheduler = PolynomialLRDecay(self.optimizer, max_decay_steps=episode_nums, end_learning_rate=0.0000)
    self.to(device)

    self.state_replay = []
    self.action_replay = []
    self.P_replay = []
    self.r_replay = []

    self.action_space = action_dim
    self.env = None

    self.var = 0
    self.beta_product = 1.0

    self.var_m = [0 for _ in range(unroll_steps)]
    self.beta_product_m = [1.0 for _ in range(unroll_steps)]

  def self_play_mu(self, target: Target, ast_num=None, evaluation=False, episode_num=0):
    """Self-play and save trajectory to replay buffer

    (Originally target network used to inference policy, but i used agent network instead) # todo

    Eight previous observations stacked -> representation network -> prediction network
    -> sampling action follow policy -> next env step
    """
    mcts_temperature = max(0.25*0.99 ** episode_num, 0.01)
    cfg = dict(num_simulations=15 if not evaluation else 20,
               discount_factor=gamma,
               device=device)
    mcts_tree = EfficientZeroMCTSPtree(EasyDict(cfg))
    # state = self.env#.reset()no reset for now
    if ast_num is None:
      ast_num = np.random.choice(first_ast_nums_correct)

    self.env = State(ast_strs, ast_num)
    base_tm = self.env.set_base_tm()
    state_feature = self.env.feature()
    # state_dim = len(state_feature)
    state_traj, action_traj, P_traj, r_traj = [], [], [], []
    hs = target.representation_network(torch.from_numpy(state_feature).to(device))
    _, base_v_logits = target.prediction_network(hs)
    base_v = to_scalar(base_v_logits.unsqueeze(0))

    for i in range(max_episode_length):
      current_state_state = state_feature
      # it combines all observations into one transformer model I think one could see it as the representation network
      # to add to observation
      # https://arxiv.org/pdf/2301.07608.pdf
      # name, dim
      # TRIALS REMAINING 5 # one hot. so max of 5 trials for example
      # reward previous step 1
      # should add:
      # basetiming to state
      with torch.no_grad():
        hs = target.representation_network(torch.from_numpy(current_state_state).to(device))
        P, v = target.prediction_network(hs)
      probs = P.detach().cpu().numpy()
      legal_actions = [list(get_linearizer_actions(self.env.lin).keys())]  # todo get_linearizer_actions is slow might improve
      action_mask = np.zeros((1, action_space))
      action_mask[0, legal_actions[0]] = 1
      parallel_num = 1
      greedy = not evaluation and (episode_num < 40 or np.random.rand() < (0.98**episode_nums))# episode 100 is 0.13 chance greedy
      if greedy:
        results = []
        for action in legal_actions[0]:
          new_env, state_feature, r, done, info = self.env.step(action, dense_reward=True)  # todo might want to change to "not eval"
          v = 0.
          results.append((new_env, state_feature, r, done, info, action, v))
        best = np.argmax([result[2] for result in results])
        self.env, state_feature, r, done, info, action, _ = results[best]
      else:
        roots = EfficientZeroMCTSPtree.roots(parallel_num, legal_actions)
        noises = [np.random.dirichlet([mcts_tree._cfg.root_dirichlet_alpha] * len(legal_actions[j])
                                      ).astype(np.float32).tolist() for j in range(parallel_num)]

        roots.prepare(mcts_tree._cfg.root_noise_weight, noises, probs.reshape(1, -1))
        # we have no reward_hidden_state_roots. we just have a single latent
        latent_state = hs  # (the hidden state in latent space)
        # latent_state = None # this requires an lstm that we dont have
        mcts_tree.search(roots, target, latent_state.detach().cpu().numpy().reshape(1, -1))
        roots_visit_count_distributions = roots.get_distributions()
        roots_values = roots.get_values()  # shape: {list: batch_size}
        ready_env_id = None
        if ready_env_id is None:
          ready_env_id = np.arange(parallel_num)
        action = -1
        for i, env_id in enumerate(ready_env_id):  # currently only 1 thing
          distributions, value = roots_visit_count_distributions[i], roots_values[i]
          # normal collect
          # NOTE: Only legal actions possess visit counts, so the ``action_index_in_legal_action_set`` represents
          # the index within the legal action set, rather than the index in the entire action set.
          action_index_in_legal_action_set, visit_count_distribution_entropy = select_action(
            distributions, temperature=mcts_temperature, deterministic=evaluation
          )
          # NOTE: Convert the ``action_index_in_legal_action_set`` to the corresponding ``action`` in the entire action set.
          action = np.where(action_mask[i] == 1.0)[0][action_index_in_legal_action_set]
        self.env, state_feature, r, done, info = self.env.step(action, dense_reward=True)
      if i == 0:
        for _ in range(num_state_stack):
          state_traj.append(current_state_state)
      else:
        state_traj.append(current_state_state)
      action_traj.append(action)
      P_traj.append(P.cpu().numpy())
      # print('r', r, 'basetm', base_tm, 'lasttm', self.env.last_tm)
      r_traj.append(r)

      ## For fix lunarlander-v2 env does not return reward -100 when 'TimeLimit.truncated'
      if done:
        last_frame = i
        break

    # for update inference over trajectory length
    for _ in range(unroll_steps):
      state_traj.append(np.zeros_like(state_feature))
    print('total r', sum(r_traj), 'base_v', base_v.item())
    for _ in range(unroll_steps + 1):
      r_traj.append(0.0)
      action_traj.append(-1)
    if not evaluation:
      # traj append to replay
      self.state_replay.append(state_traj)
      self.action_replay.append(action_traj)
      self.P_replay.append(P_traj)
      self.r_replay.append(r_traj)
    self.env.reward()  # set last_tm
    game_score = (base_tm - self.env.last_tm) / base_tm
    base_to_handcoded = (base_tm - self.env.handcoded_tm) / base_tm
    base_to_beam = ((base_tm - beam_results[ast_num]) / base_tm) if beam else None
    return game_score, base_to_handcoded, base_to_beam, r, last_frame, ast_num

  # from line_profiler_pycharm import profile
  # @profile
  def update_weights_mu(self, target: Target, episode_num):
    """Optimize network weights.

    Iteration: 20
    Mini-batch size: 16 (4 replay, 4 sequences in 1 replay)
    Replay: Uniform replay with on-policy data
    Discount: gamma
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
      for epi_sel in range(num_replays):  # this is q
        if (epi_sel > num_replays // 4):  ## 1/4 of replay buffer is used for on-policy data
          ep_i = np.random.randint(0, max(1, len(self.state_replay) - 1))
        else:
          ep_i = -np.random.randint(max(1, episodes_per_train_update))  # pick one of the newest episodes

        ## multi step return G (orignally retrace used)
        G = 0
        G_arr = []
        for r in self.r_replay[ep_i][::-1]:
          G = gamma * G + r
          G_arr.append(G)
        G_arr.reverse()
        for i in np.random.randint(len(self.state_replay[ep_i]) - unroll_steps - (num_state_stack - 1), size=seq_in_replay):
          state_traj_i.append(i)
          state_traj.append(self.state_replay[ep_i][i:i + unroll_steps + num_state_stack])
          action_traj.append(self.action_replay[ep_i][i:i + unroll_steps + num_state_stack])
          r_traj.append(self.r_replay[ep_i][i:i + unroll_steps])
          G_arr_mb.append(G_arr[i:i + unroll_steps + 1])
          P_traj.append(self.P_replay[ep_i][i])

      state_traj_i = np.array(state_traj_i)
      state_traj = torch.from_numpy(np.array(state_traj)).to(device)
      action_traj = torch.from_numpy(np.array(action_traj)).unsqueeze(2).to(device)
      P_traj = torch.from_numpy(np.array(P_traj)).to(device)
      G_arr_mb = torch.from_numpy(np.array(G_arr_mb)).unsqueeze(2).float().to(device)
      r_traj = torch.from_numpy(np.array(r_traj)).unsqueeze(2).float().to(device)
      mask_batch = torch.zeros_like(action_traj)
      mask_batch[action_traj != -1] = 1  # all states with action not -1 are valid states

      inferenced_P_arr = []

      ## stacking
      state_0 = state_traj[:, 0]

      ## agent network inference (5 step unroll)
      latent_states = []
      v_logits_s = []
      r_logits_s = []
      for i in range(1 + unroll_steps):
        if i == 0:
          hs = self.representation_network(state_0)
        else:
          hs, r_logits = self.dynamics_network(hs, action_traj[:, i - 1])
          r_logits_s.append(r_logits)

        P, v_logits = self.prediction_network(hs)
        latent_states.append(hs)
        v_logits_s.append(v_logits)
        inferenced_P_arr.append(P)
      first_P = inferenced_P_arr[0]

      # target network inference
      with torch.no_grad():
        t_first_hs = target.representation_network(state_0)
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
      def lookahead(P_tensor, hs, _lookahead_samples, base_v_logits):
        with torch.no_grad():
          P_np = P_tensor.cpu().numpy()
          action1_stack_all_samples = np.zeros((len(P_np), _lookahead_samples), dtype=np.int64)
          for j, p in enumerate(P_np):
            action1_stack_all_samples[j] = np.random.choice(action_space_arange, p=p, size=_lookahead_samples)  # this is very slow!
          action1_stack_all_samples = torch.from_numpy(action1_stack_all_samples).T
          hs, r1 = target.dynamics_network(hs.expand((*action1_stack_all_samples.shape, -1)), action1_stack_all_samples)
          _, v1 = target.prediction_network(hs)
          r1_arr = to_scalar(r1)
          v1_arr = to_scalar(v1)
          a1_arr = action1_stack_all_samples

          torch.exp(torch.clip((r1_arr + gamma * v1_arr - to_scalar(base_v_logits)) / under, -1, 1))
          exp_clip_adv_arr = torch.exp(torch.clip((r1_arr + gamma * v1_arr - to_scalar(base_v_logits)) / under, -1, 1))
        return r1_arr, v1_arr, a1_arr, exp_clip_adv_arr

      r1_arr, v1_arr, a1_arr, exp_clip_adv_arr = lookahead(t_first_P, t_first_hs, lookahead_samples, t_first_v_logits)
      ## z_cmpo_arr (eq.12)
      with torch.no_grad():
        z_cmpo_arr = []
        for k in range(lookahead_samples):  # k != i is fixed by "- exp_clip_adv_arr[k]"
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
      L_consistency = 0.
      for i in range(unroll_steps):
        state = state_traj[:, i]

        # this is efficientzero way to get a better inner state
        dynamic_proj = self.prediction_network.projection(latent_states[i])  # take the dynamics state. really want to have an lstm state here
        self.prediction_network.prediction_head(dynamic_proj)
        with torch.no_grad():
          hs = self.representation_network(state)
          observation_proj = self.prediction_network.projection(hs)  #
          observation_proj.detach()

        L_consistency += negative_cosine_similarity(dynamic_proj, observation_proj) * mask_batch[:, i]

      limited_unroll_steps = 0
      for i in range(unroll_steps):
        _mask_batch = mask_batch[:, i + 1]  # we need to mask finished episodes
        if torch.sum(_mask_batch == 1) < 2:
          print("Not enough correct steps")
          break
        limited_unroll_steps = i + 1
        stacked_state = state_traj[:, i + 1]
        with torch.no_grad():
          t_hs = target.representation_network(stacked_state)
          t_P, t_v_logits = target.prediction_network(t_hs)

        beta_var = 0.99
        masked_minibatch_size = _mask_batch.sum()
        self.var_m[i] = beta_var * self.var_m[i] + (1 - beta_var) * (
          torch.sum(((G_arr_mb[:, i + 1] - to_scalar(t_v_logits)) * _mask_batch) ** 2) / masked_minibatch_size)
        self.beta_product_m[i] *= beta_var
        var_hat = self.var_m[i] / (1 - self.beta_product_m[i])
        under = torch.sqrt(var_hat + 1e-12)

        r1_arr, v1_arr, a1_arr, exp_clip_adv_arr = lookahead(t_P, t_hs, lookahead_samples2, t_v_logits)

        exp_clip_adv_arr = exp_clip_adv_arr * _mask_batch
        # ## Paper appendix F.2 : Prior policy
        t_P = 0.967 * t_P + 0.03 * P_traj + 0.003 * (torch.ones((minibatch_size, self.env.action_length())) / self.env.action_length()).to(device)

        pi_cmpo_all = [(t_P.gather(1, torch.unsqueeze(a1_arr[k], 1).to(device))
                        * exp_clip_adv_arr[k])
                       .squeeze(-1).tolist() for k in range(lookahead_samples2)]

        pi_cmpo_all = (torch.tensor(pi_cmpo_all).transpose(0, 1).to(device))[_mask_batch.squeeze() == 1]
        pi_cmpo_all = pi_cmpo_all / torch.sum(pi_cmpo_all, dim=1).unsqueeze(-1)
        kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
        # # I think this is right. todo but not completely sure
        P_arr_selected_actions = torch.gather(inferenced_P_arr[i + 1], 1, a1_arr.T.to(device))
        L_m += kl_loss(torch.log(P_arr_selected_actions)[_mask_batch.squeeze() == 1], pi_cmpo_all)
      if limited_unroll_steps > 0:
        L_m /= limited_unroll_steps

      ## L_v
      ls = nn.LogSoftmax(dim=-1)
      L_v_dist = 0
      L_v = 0.
      # todo this might be a weird mean,  if some of them are zerod out. (But grads stay very similar though)
      for i in range(unroll_steps + 1): L_v_dist -= (to_cr(G_arr_mb[:, i]) * ls(v_logits_s[i])).sum(-1, keepdim=True) * mask_batch[:, i]
      # Just for statistics:
      for i in range(unroll_steps + 1): L_v += (to_scalar(v_logits_s[i]) - G_arr_mb[:, i]) ** 2 * mask_batch[:, i]
      print('L_v', L_v.mean().item())
      # ## consistency_loss
      print('consistency_loss', L_consistency.mean().item())

      # ## L_r
      L_r = 0.
      for i in range(unroll_steps): L_r -= (to_cr(r_traj[:, i]) * ls(r_logits_s[i])).sum(-1, keepdim=True)
      ## start of dynamics network gradient *0.5 # todo weird?
      # for i in range(unroll_steps + 1): latent_states[i].register_hook(lambda grad: grad * 0.5)

      ## total loss
      # L_total = L_pg_cmpo + L_v / (unroll_steps + 1) / 4 + L_r / unroll_steps / 1 + L_m
      L_total = L_pg_cmpo + L_v_dist + L_consistency + L_r + L_m
      ## optimize
      self.optimizer.zero_grad()
      L_total.mean().backward()
      nn.utils.clip_grad_value_(self.parameters(), clip_value=1.0)
      if episode_num <= 8:  # first few need smaller lr. Otherwise Nan can occur
        for param_group in self.optimizer.param_groups:
          param_group['lr'] = 3e-4
      self.optimizer.step()

      ## target network(prior parameters) moving average update
      params1 = list(self.named_parameters()) + list(self.named_buffers())
      params2 = list(target.named_parameters()) + list(target.named_buffers())
      dict_params2 = dict(params2)
      for name1, param1 in params1:
        if name1 in dict_params2:
          dict_params2[name1].data.copy_(alpha_target * param1.data + (1 - alpha_target) * dict_params2[name1].data)
      target.load_state_dict(dict_params2)

    self.scheduler.step()
    print('L_total', L_total.mean().item())
    writer.add_scalar('Loss/L_total', L_total.mean(), global_i)
    writer.add_scalar('Loss/L_pg_cmpo', L_pg_cmpo.mean(), global_i)
    writer.add_scalar('Loss/L_consistency', (L_consistency / unroll_steps).mean(), global_i)
    writer.add_scalar('Loss/L_v', (L_v / (unroll_steps + 1) / 4).mean(), global_i)
    writer.add_scalar('Loss/L_v_dist', (L_v_dist / (unroll_steps + 1) / 4).mean(), global_i)
    writer.add_scalar('Loss/L_r', (L_r / unroll_steps / 1).mean(), global_i)
    writer.add_scalar('Loss/L_m', (L_m).mean(), global_i)
    # writer.add_scalars('vars', {'self.var': self.var,
    #                             'self.var_m': self.var_m[0]
    #                             }, global_i)


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
score_arr = []
score_arr_handcoded = []

ast_strs = load_worlds()
random.shuffle(ast_strs)

first_ast_nums_correct = list(range(50))
# for i in deepcopy(first_ast_nums_correct):
#   try:
#     env = State(ast_strs, i)
#     env.set_base_tm()
#
#     print(i)
#   except:
#     print('removing astnum', i)
#     del env
#     del first_ast_nums_correct[i]
first_ast_nums = len(first_ast_nums_correct)
num_replays = 64
seq_in_replay = 1
minibatch_size = num_replays * seq_in_replay
max_trials = 1
use_sts = True
max_episode_length = MAX_DIMS * max_trials
num_state_stack = 1

train_eval_size = 10
test_size = 5
train_updates = 2
episodes_per_train_update = 4  # was 4
episodes_per_eval = 256
start_training_epoch = episodes_per_train_update  # todo set higher
assert start_training_epoch >= episodes_per_train_update
use_transformer = True
alpha_target = 0.05  # 5% new to target

regularizer_multiplier = 2  # was 5 but that seems very high
gamma = 0.999
lookahead_samples = 32
lookahead_samples2 = 32  # this is in the unroll
unroll_steps = 5

env = State(ast_strs, 0)
observation_space = env.feature_shape()
env.base_tm = -1
assert observation_space[0] == len(env.feature())
action_space = env.action_length()
action_space_arange = np.arange(action_space)

episode_nums = 1000
timing_str = datetime.now().strftime("%Y%m%d-%H%M%S")
# exp_str = 'test_'+timing_str # 1,
# exp_str = 'test_no_transformer_'+timing_str # 1,
# exp_str = 'test_new_transformer_less_training_updates'+timing_str # 1,
# exp_str = f'greedy_sts{use_sts}_astnums{len(first_ast_nums_correct)}_' + timing_str  # 1,
# exp_str = f'0.5greedy_sts{use_sts}_astnums{len(first_ast_nums_correct)}_' + timing_str  # 1,
# exp_str = f'no_greedy_sts{use_sts}_astnums{len(first_ast_nums_correct)}_' + timing_str  # 1,
exp_str = f'partially_greedy_sts{use_sts}_astnums{len(first_ast_nums_correct)}_' + timing_str  # 1,
del env
target = Target(observation_space[0], action_space, 1024)
target.eval()
agent = Agent(observation_space[0], action_space, 1024)
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
beam = False
if beam:
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
  game_score, base_to_handcoded, base_to_beam, last_r, frame, ast_num = agent.self_play_mu(target, episode_num=i)
  more_than_handcoded = game_score - base_to_handcoded
  ast_num_i = first_ast_nums_correct.index(ast_num)
  if best_ast_num_scores[ast_num_i] < game_score:
    best_ast_num_scores[ast_num_i] = game_score
  if handcoded_best_ast_num_scores[ast_num_i] == 0.:
    handcoded_best_ast_num_scores[ast_num_i] = base_to_handcoded

  print(f'steps done {frame}. relative speed score {game_score:.2f} relative % more than handcoded {more_than_handcoded:.2f} iteration {global_i}')
  # diff_best_handcoded = best_ast_num_scores - handcoded_best_ast_num_scores
  # print('Diff with best handcoded scores for each ast num', (diff_best_handcoded).round(2).tolist())
  # print('mean diff', diff_best_handcoded.mean().round(2).tolist())
  writer.add_scalar('relative_to_base_score', max(game_score, -1), global_i)
  writer.add_scalar('more_than_handcoded', max(more_than_handcoded, -1), global_i)
  # writer.add_scalar('diff_to_best_handcoded', diff_best_handcoded.mean(), global_i)

  score_arr.append(game_score)
  score_arr_handcoded.append(more_than_handcoded)

  # if i % 100 == 0:
  #   torch.save(agent.state_dict(), f'weights_id{exp_str}.pt')

  # if mean_score > 250 and np.mean(np.array(score_arr[-5:])) > 250:
  #   torch.save(agent.state_dict(), f'weights_id{exp_str}.pt')
  #   print('Done')
  #   break
  if i % episodes_per_train_update == 0 and i >= start_training_epoch:
    print("Training update")
    agent.update_weights_mu(target, i)
    print("Done update")

  best_score_related_to_handcoded = (best_ast_num_scores - handcoded_best_ast_num_scores)[best_ast_num_scores != 0.]
  if len(best_score_related_to_handcoded) == 0:
    best_score_related_to_handcoded = np.zeros(1)
  writer.add_scalar('best_found_score_related_to_handcoded', best_score_related_to_handcoded.mean(), global_i)
  if i % episodes_per_eval == 0 and i >= start_training_epoch:
    print('best scores for each ast num', best_ast_num_scores.round(2).tolist())
    mean_score = np.mean(np.array(score_arr[i - 30:i + 1]))
    more_than_handcoded_mean = np.mean(np.array(score_arr_handcoded[i - 30:i + 1]))
    print(f'episode {i}, avg {mean_score:.2f}, avg handcoded {more_than_handcoded_mean:.2f}')
    scores, test_scores = [], []
    for i in np.random.choice(first_ast_nums_correct[:-test_size], size=train_eval_size):  # eval
      game_score, base_to_handcoded, base_to_beam, _, frame, _ = agent.self_play_mu(target, ast_num=i, evaluation=True)
      more_than_handcoded = game_score - base_to_handcoded
      if beam:
        more_than_beam = game_score - base_to_beam

      scores.append([game_score, more_than_handcoded, base_to_beam])
    for i in first_ast_nums_correct[-test_size:]:  # eval
      game_score, base_to_handcoded, base_to_beam, _, frame, _ = agent.self_play_mu(target, ast_num=i, evaluation=True)
      more_than_handcoded = game_score - base_to_handcoded
      if beam:
        more_than_beam = game_score - base_to_beam

      test_scores.append([game_score, more_than_handcoded, base_to_beam])
    scores = np.array(scores)
    test_scores = np.array(test_scores)
    writer.add_scalar('eval_train/relative_to_base_score', scores[:, 0].mean(), global_i)
    writer.add_scalar('eval_train/more_than_handcoded', scores[:, 1].mean(), global_i)
    # writer.add_scalar('eval_train/more_than_beam', scores[:,2].mean(), global_i)

    writer.add_scalar('eval_test/relative_to_base_score', test_scores[:, 0].mean(), global_i)
    writer.add_scalar('eval_test/more_than_handcoded', test_scores[:, 1].mean(), global_i)
    # writer.add_scalar('eval_test/more_than_beam', scores[:,2].mean(), global_i)

    # print('eval relative_to_base_score', scores[:,0].round(2))
    # print('eval more_than_handcoded', scores[:,1].round(2))
    # print('eval more_than_beam', scores[:,2].round(2))
  # writer.close()

torch.save(agent.state_dict(), f'weights_id{exp_str}.pt')
agent.env.close()
# python -m tensorboard.main --logdir optimization