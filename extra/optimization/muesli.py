import os
os.environ["GPU"] = '1'
import time

import numpy as np
import math, random

import torch.random
from torch import optim
from tinygrad.tensor import Tensor

Tensor.manual_seed(0)
random.seed(0)
torch.random.manual_seed(0)
np.random.seed(0)

from extra.optimization.muesli_net import Net, State
from tinygrad.nn.state import safe_load, load_state_dict
from extra.optimization.helpers import load_worlds


def gen_target(ep):
  '''Generate inputs and targets for training'''
  # path, reward, observation, action, policy
  ep_length = len(ep['feature'])
  turn_idx = np.random.randint(ep_length)

  x = ep['feature'][turn_idx]
  ps, rs, acts, axs = [], [], [], []
  sas, seas, szs = [], [], []
  states = []
  for t in range(turn_idx, turn_idx + same_episode_length_K + 1):
    if t < ep_length:
      p = ep['policy'][t]
      a = ep['action'][t]
      ax = ep['action_feature'][t]
      sa = ep['sampled_info'][t]['a']
      sea = ep['sampled_info'][t]['exadv']
      sz = ep['sampled_info'][t]['z']
      _state = ep['state'][t]
    else:  # state after finishing game
      p = np.zeros_like(ep['policy'][-1])
      # random action selection
      # a = np.random.randint(State.action_length())
      probs = np.ones(State.action_length())/State.action_length()
      act = state.get_valid_action(probs)
      ax = state.step(act).feature()
      # ax = state.action_feature()
      sa = np.random.randint(state.action_length(), size=len(sa))
      sea = np.ones_like(sea)
      sz = np.ones_like(sz)
    rs.append(ep['reward']) # if t % 2 == 0 else -ep['reward']])
    acts.append([a])
    axs.append(ax)
    ps.append(p)
    sas.append(sa)
    seas.append(sea)
    szs.append(sz)
    states.append(_state) # todo

  return x, rs, acts, axs, ps, sas, seas, szs, states


# @profile
def train(episodes, net:Net, optimizer):
  '''Train neural net'''
  pg_loss_sum, cmpo_loss_sum, v_loss_sum = 0, 0, 0
  net.train()
  for _ in range(training_steps):
    targets = [gen_target(episodes[np.random.randint(len(episodes))]) for _ in range(batch_size)]
    x, r, a, ax, p_prior, sa, sea, sz, states = zip(*targets)
    x = torch.from_numpy(np.array(x, dtype=np.float32))
    r = torch.from_numpy(np.array(r))
    a = torch.from_numpy(np.array(a))
    ax = torch.from_numpy(np.array(ax))
    p_prior = torch.from_numpy(np.array(p_prior))
    sa = torch.from_numpy(np.array(sa))
    sea = torch.from_numpy(np.array(sea))
    sz = torch.from_numpy(np.array(sz))

    # Compute losses for k (+ current) steps
    ps, vs = [], []
    # rp = net.representation(x)
    rp = x # no representation for now
    for t in range(same_episode_length_K + 1):
      p, v = net.prediction(rp)
      ps.append(p)
      vs.append(v)
      # rp = net.dynamics(rp, ax[:, t])
      rp = torch.tensor([s[t].feature() for s in states])

    cmpo_loss, v_loss = 0, 0
    for t in range(same_episode_length_K, -1, -1): # todo bit weird?? is just [1,0]
      cmpo_loss += -torch.mean(sea[:, t] / sz[:, t] * torch.log(ps[t].gather(1, sa[:, t])), dim=1).sum()
      v_loss += torch.sum(((vs[t] - r[:, t]) ** 2) / 2)

    p_selected = ps[0].gather(1, a[:, 0])
    p_selected_prior = p_prior[:, 0].gather(1, a[:, 0])
    clipped_rho = torch.clamp(p_selected.detach() / p_selected_prior, 0, 1)
    pg_loss = torch.sum(-clipped_rho * torch.log(p_selected) * (r[:, 0] - vs[0]))

    pg_loss_sum += pg_loss.item()
    cmpo_loss_sum += cmpo_loss.item() / (same_episode_length_K + 1)
    v_loss_sum += v_loss.item() / (same_episode_length_K + 1)

    optimizer.zero_grad()
    print(f'pg_loss,cmpo_loss,v_loss {pg_loss.item()/batch_size:.2f},{cmpo_loss.item()/batch_size:.2f},{v_loss.item()/batch_size:.2f}')
    assert math.isfinite(pg_loss + cmpo_loss + v_loss), f'nan pg_loss,cmpo_loss,v_loss {pg_loss.item(),cmpo_loss.item(),v_loss.item()}'
    (pg_loss + cmpo_loss + v_loss).backward()
    optimizer.step()

  data_count = training_steps * batch_size
  print('pg_loss %f cmpo_loss %f v_loss %f' % (pg_loss_sum / data_count, cmpo_loss_sum / data_count, v_loss_sum / data_count))
  return net



if __name__ == "__main__":
  batch_size = 32
  training_steps = 4
  num_epochs = 100
  start_train_epoch = 0
  episodes_per_training_step = 1
  same_episode_length_K = 1
  C = 1
  num_sampled_actions = 3
  simulation_depth = 1
  # DENSE_REWARD = True
  # net = PolicyNet()
  # muesli stuff
  net = Net()
  # optimizer = optim.SGD(net.parameters(), lr=3e-4, weight_decay=3e-5, momentum=0.8)
  optimizer = optim.Adam(net.parameters(), lr=3e-5, weight_decay=3e-5)
  episodes = []

  if os.path.isfile("/tmp/policynet.safetensors"): load_state_dict(net, safe_load("/tmp/policynet.safetensors"))
  # optim = Adam(get_parameters(net))

  ast_strs = load_worlds()

  # select a world
  all_feats, all_acts, all_rews = [], [], []

  for epoch in range(num_epochs):
    net.eval()
    ast_num = epoch%1
    print('astnum', ast_num)
    state = State(ast_strs[ast_num])
    # state = State(ast_strs[0])
    # assert not math.isinf(state.tm)
    # print(f"base time {state.base_tm*1000:.1f} ms")
    features, policies, selected_actions, selected_action_features = [], [], [], []
    states = []
    sampled_infos = []
    action_stop = False
    with torch.no_grad():

      while not state.terminal:
        feat = state.feature()
        features.append(feat)
        states.append(state)
        # probs = net(Tensor([feat])).exp()[0].numpy()

        # rp_root = net.representation.inference(feat)
        rp_root = feat # no representation for now
        p_root, v_root = net.prediction.inference(np.asarray(rp_root))
        # p_root = state.get_masked_probs(p_root)
        # print('p_root', p_root, p_root.sum())
        policies.append(p_root)
        actions, exadvs = [], []
        for i in range(num_sampled_actions):
          root_act = state.get_valid_action(p_root)

          plan_state = state
          # todo could also skip action here if step doesnt work
          # action = np.random.choice(np.arange(len(p_root)), p=p_root)
          # root_act = np.random.choice(len(p_root), p=p_root)

          rp = rp_root
          qs = []
          act = root_act
          failed = False
          for t in range(simulation_depth):
            if plan_state.terminal:
              break
            try:
              # action_featureplan_state.feature() # check for fails
              plan_state = plan_state.step(act)
            except AssertionError as e:
              print("STILL fails?")
              failed = True
              break
            action_feature = plan_state.feature() # todo currently state feature
            # rp = net.dynamics.inference(rp, action_feature) # todo we could just replace with real model here?
            rp = action_feature  # todo currently regular feature is used for action feature
            p, v = net.prediction.inference(np.asarray(rp))
            qs.append(v)
            if act == 0: # final action
              break
            masked_st = time.perf_counter()
            act = plan_state.get_valid_action(p)
            # p = plan_state.get_masked_probs(p)
            # print(f'masked probs time {time.perf_counter()-masked_st}')
            # act = np.random.choice(len(p), p=p)
          if failed:
            continue
          actions.append(root_act) # only add when we have a working step
          q = np.mean(qs)
          exadvs.append(np.exp(np.clip(q - v_root, -C, C)))
        assert len(actions) > 0
        exadv_sum = np.sum(exadvs)
        zs = []
        for exadv in exadvs:
          z = (1 + exadv_sum - exadv) / num_sampled_actions
          zs.append(z)
        sampled_infos.append({'a': actions, 'q': qs, 'exadv': exadvs, 'z': zs})

        # Select action with generated distribution, and then make a transition by that action
        # selected_action = np.random.choice(np.arange(len(p_root)), p=p_root)
        selected_action = state.get_valid_action(p_root)

        state = state.step(selected_action)
        selected_actions.append(selected_action)
        # selected_action_features.append(state.action_feature(selected_action))
        selected_action_features.append(state.feature()) # todo currently the action feature is simply the next state
        # print('State step', state.steps)
      # state.play(selected_action)
      # print("steps taken", state.steps)
      # print(f"Final ast {ast_num} color", state.lin.colors())
      # print(f"prediction reward {net.prediction.value(torch.tensor(features[-1]).float()).cpu().item():.2f}", )
      reward = state.terminal_reward()
      # result_distribution[reward] += 1
      assert len(features) > 0
      episodes.append({
        'feature': features, 'action': selected_actions,
        'action_feature': selected_action_features, 'policy': policies,
        'reward': reward,
        'sampled_info': sampled_infos, 'state':states})
      print(f'*** episode reward {reward:.5f} mean last 10 episodes {np.mean([e["reward"] for e in episodes[-10:]]):.3f}')

    # Training of neural net
    if (epoch) % episodes_per_training_step == 0 and epoch > start_train_epoch:
      print("train")
      net = train(episodes, net, optimizer)
      print("done train")