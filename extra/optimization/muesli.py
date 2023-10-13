import os
import time
from copy import deepcopy

import numpy as np
import math, random

import torch.random
from torch import nn, optim

from extra.optimization.muesli_net import Net, State
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from extra.optimization.helpers import load_worlds, ast_str_to_lin, lin_to_feats



def gen_target(ep):
  '''Generate inputs and targets for training'''
  # path, reward, observation, action, policy
  ep_length = len(ep['feature'])
  turn_idx = np.random.randint(ep_length)

  x = ep['feature'][turn_idx]
  ps, rs, acts, axs = [], [], [], []
  sas, seas, szs = [], [], []
  for t in range(turn_idx, turn_idx + K + 1):
    if t < ep_length:
      p = ep['policy'][t]
      a = ep['action'][t]
      ax = ep['action_feature'][t]
      sa = ep['sampled_info'][t]['a']
      sea = ep['sampled_info'][t]['exadv']
      sz = ep['sampled_info'][t]['z']
    else:  # state after finishing game
      p = np.zeros_like(ep['policy'][-1])
      # random action selection
      a = np.random.randint(State.action_length())
      # ax = state.action_feature(a)
      ax = state.step(act).feature()
      # ax = state.action_feature()
      sa = np.random.randint(state.action_length(), size=len(sa))
      sea = np.ones_like(sea)
      sz = np.ones_like(sz)

    rs.append([ep['reward'] if t % 2 == 0 else -ep['reward']])
    acts.append([a])
    axs.append(ax)
    ps.append(p)
    sas.append(sa)
    seas.append(sea)
    szs.append(sz)

  return x, rs, acts, axs, ps, sas, seas, szs
def train(episodes, net, opt):
  '''Train neural net'''
  pg_loss_sum, cmpo_loss_sum, v_loss_sum = 0, 0, 0
  net.train()
  # state = State()
  for _ in range(num_steps):
    targets = [gen_target(state, episodes[np.random.randint(len(episodes))]) for j in range(batch_size)]
    x, r, a, ax, p_prior, sa, sea, sz = zip(*targets)
    x = torch.from_numpy(np.array(x))
    r = torch.from_numpy(np.array(r))
    a = torch.from_numpy(np.array(a))
    ax = torch.from_numpy(np.array(ax))
    p_prior = torch.from_numpy(np.array(p_prior))
    sa = torch.from_numpy(np.array(sa))
    sea = torch.from_numpy(np.array(sea))
    sz = torch.from_numpy(np.array(sz))

    # Compute losses for k (+ current) steps
    ps, vs = [], []
    rp = net.representation(x)
    for t in range(K + 1):
      p, v = net.prediction(rp)
      ps.append(p)
      vs.append(v)
      rp = net.dynamics(rp, ax[:, t])

    cmpo_loss, v_loss = 0, 0
    for t in range(K, -1, -1):
      cmpo_loss += -torch.mean(sea[:, t] / sz[:, t] * torch.log(ps[t].gather(1, sa[:, t])), dim=1).sum()
      v_loss += torch.sum(((vs[t] - r[:, t]) ** 2) / 2)

    p_selected = ps[0].gather(1, a[:, 0])
    p_selected_prior = p_prior[:, 0].gather(1, a[:, 0])
    clipped_rho = torch.clamp(p_selected.detach() / p_selected_prior, 0, 1)
    pg_loss = torch.sum(-clipped_rho * torch.log(p_selected) * (r[:, 0] - vs[0]))

    pg_loss_sum += pg_loss.item()
    cmpo_loss_sum += cmpo_loss.item() / (K + 1)
    v_loss_sum += v_loss.item() / (K + 1)

    optimizer.zero_grad()
    (pg_loss + cmpo_loss + v_loss).backward()
    optimizer.step()

  data_count = num_steps * batch_size
  print('pg_loss %f cmpo_loss %f v_loss %f' % (pg_loss_sum / data_count, cmpo_loss_sum / data_count, v_loss_sum / data_count))
  return net



if __name__ == "__main__":
  batch_size = 32
  num_steps = 100
  num_epochs = 10
  gen_kernels_per_training_step = 10
  K = 1
  C = 1

  # DENSE_REWARD = True
  Tensor.manual_seed(0)
  random.seed(0)
  torch.random.manual_seed(0)
  np.random.seed(0)
  # net = PolicyNet()
  # muesli stuff
  net = Net()
  optimizer = optim.SGD(net.parameters(), lr=3e-4, weight_decay=3e-5, momentum=0.8)
  episodes = []

  if os.path.isfile("/tmp/policynet.safetensors"): load_state_dict(net, safe_load("/tmp/policynet.safetensors"))
  # optim = Adam(get_parameters(net))

  ast_strs = load_worlds()

  # select a world
  all_feats, all_acts, all_rews = [], [], []
  for epoch in range(num_epochs):
  # while 1:
    # Tensor.no_grad, Tensor.training = True, False
    # lin = ast_str_to_lin(random.choice(ast_strs))
    # lin = ast_str_to_lin(ast_strs[0])  # debug single ast
    state = State(ast_strs[0])
    # rawbufs = bufs_from_lin(lin)
    # tm = last_tm = base_tm = time_linearizer(lin, rawbufs)

    # take actions
    # feats, acts, rews = [], [], []
    features, policies, selected_actions, selected_action_features = [], [], [], []
    sampled_infos = []
    action_stop = False
    while not state.terminal:
      feat = state.feature()
      features.append(feat)
      # probs = net(Tensor([feat])).exp()[0].numpy()

      # rp_root = net.representation.inference(feat)
      rp_root = feat # no representation for now
      p_root, v_root = net.prediction.inference(np.array(rp_root))
      p_root = state.get_masked_probs(p_root)

      policies.append(p_root)
      actions, exadvs = [], []
      num_sampled_actions = 2
      simulation_depth = 1
      for i in range(num_sampled_actions):
        plan_state = state
        # todo could also skip action here if step doesnt work
        # action = np.random.choice(np.arange(len(p_root)), p=p_root)
        root_act = np.random.choice(len(p_root), p=p_root)

        rp = rp_root
        qs = []
        act = root_act
        for t in range(simulation_depth):
          try:
            plan_state.feature() # check for fails
            plan_state = plan_state.step(act)
          except Exception:
            print("STILL fails?")
            break

          action_feature = plan_state.feature() # todo currently state feature
          # rp = net.dynamics.inference(rp, action_feature) # todo we could just replace with real model here?
          rp = action_feature  # todo currently regular feature is used for action feature
          p, v = net.prediction.inference(np.asarray(rp))
          qs.append(-v if t % 2 == 0 else v)
          if act == 0: # final action
            print("final action")
            break
          masked_st = time.perf_counter()
          p = plan_state.get_masked_probs(p)
          print(f'masked probs time {time.perf_counter()-masked_st}')
          act = np.random.choice(len(p), p=p)
        else:
          # failed so continue to next sampled action
          continue
        actions.append(root_act) # only add when we have a working
        q = np.mean(qs)
        exadvs.append(np.exp(np.clip(q - v_root, -C, C)))
        # actions.append(act)

      exadv_sum = np.sum(exadvs)
      zs = []
      for exadv in exadvs:
        z = (1 + exadv_sum - exadv) / num_sampled_actions
        zs.append(z)
      sampled_infos.append({'a': actions, 'q': qs, 'exadv': exadvs, 'z': zs})

      # Select action with generated distribution, and then make a transition by that action
      selected_action = np.random.choice(np.arange(len(p_root)), p=p_root)

      state = state.step(selected_action)
      selected_actions.append(selected_action)
      # selected_action_features.append(state.action_feature(selected_action))
      selected_action_features.append(state.feature()) # todo currently the action feature is simply the next state
      print('State step', state.steps)
    # state.play(selected_action)

    reward = state.terminal_reward()
    # result_distribution[reward] += 1
    episodes.append({
      'feature': features, 'action': selected_actions,
      'action_feature': selected_action_features, 'policy': policies,
      'reward': reward,
      'sampled_info': sampled_infos})

    # if g % num_games_one_epoch == 0:
    #   print('game ', end='')
    # print(g, ' ', end='')

    # Training of neural net
    if (epoch + 1) % gen_kernels_per_training_step == 0:
      # Show the result distributiuon of generated episodes
      # print('generated = ', sorted(result_distribution.items()))
      net = train(episodes, net, optimizer)
    # OLDDDDDDDDDDD
    # assert len(feats) == len(acts) and len(acts) == len(rews)
    # # print(rews)
    # print(f"***** EPISODE {len(rews)} steps, {sum(rews):5.2f} reward, {base_tm * 1e6:12.2f} -> {tm * 1e6:12.2f} : {lin.colored_shape()}")
    # all_feats += feats
    # all_acts += acts
    # # rewards to go
    # for i in range(len(rews) - 2, -1, -1): rews[i] += rews[i + 1]
    # all_rews += rews
    #
    # BS = 32
    # if len(all_feats) >= BS:
    #   Tensor.no_grad, Tensor.training = False, True
    #   x = Tensor(all_feats[:BS])
    #   mask = np.zeros((BS, len(actions) + 1), dtype=np.float32)
    #   mask[range(BS), all_acts[:BS]] = all_rews[:BS]
    #   loss = -(net(x) * Tensor(mask)).mean()
    #   optim.zero_grad()
    #   loss.backward()
    #   optim.step()
    #   all_feats = all_feats[BS:]
    #   all_acts = all_acts[BS:]
    #   all_rews = all_rews[BS:]