import copy
from typing import Any, List, TYPE_CHECKING, Tuple, Union

import numpy as np
import torch
from easydict import EasyDict

import extra.optimization.ptree_ez as tree
from extra.optimization.minimax import MinMaxStatsList
from extra.optimization.muesli_funcs import to_scalar

if TYPE_CHECKING:
  from extra.optimization.muesli_new import Target

from scipy.stats import entropy

def select_action(visit_counts: np.ndarray,
                  temperature: float = 1,
                  deterministic: bool = True) -> Tuple[np.int64, np.ndarray]:
    """
    Overview:
        Select action from visit counts of the root node.
    Arguments:
        - visit_counts (:obj:`np.ndarray`): The visit counts of the root node.
        - temperature (:obj:`float`): The temperature used to adjust the sampling distribution.
        - deterministic (:obj:`bool`):  Whether to enable deterministic mode in action selection. True means to \
            select the argmax result, False indicates to sample action from the distribution.
    Returns:
        - action_pos (:obj:`np.int64`): The selected action position (index).
        - visit_count_distribution_entropy (:obj:`np.ndarray`): The entropy of the visit count distribution.
    """
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i in visit_counts]
    action_probs = [x / sum(action_probs) for x in action_probs]

    if deterministic:
        action_pos = np.argmax([v for v in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    visit_count_distribution_entropy = entropy(action_probs, base=2)
    return action_pos, visit_count_distribution_entropy

def to_detach_cpu_numpy(data_list: Union[torch.Tensor, List[torch.Tensor]]) -> Union[np.ndarray, List[np.ndarray]]:
  """
  Overview:
      convert the data or data list to detach cpu numpy.
  Arguments:
      - data_list (:obj:`Union[torch.Tensor, List[torch.Tensor]]`): the data or data list
  Returns:
      - output_data_list (:obj:`Union[np.ndarray,List[np.ndarray]]`): the output data or data list
  """
  if isinstance(data_list, torch.Tensor):
    return data_list.detach().cpu().numpy()
  elif isinstance(data_list, list) and all(isinstance(data, torch.Tensor) for data in data_list):
    output_data_list = []
    for data in data_list:
      output_data_list.append(data.detach().cpu().numpy())
    return output_data_list
  else:
    raise TypeError("The type of input must be torch.Tensor or List[torch.Tensor]")


class EfficientZeroMCTSPtree(object):
  """
  Overview:
      MCTSPtree for EfficientZero. The core ``batch_traverse`` and ``batch_backpropagate`` function is implemented in python.
  Interfaces:
      __init__, roots, search
  """

  # the default_config for EfficientZeroMCTSPtree.
  config = dict(
    # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
    root_dirichlet_alpha=0.3,
    # (float) The noise weight at the root node of the search tree.
    root_noise_weight=0.25,
    # (int) The base constant used in the PUCT formula for balancing exploration and exploitation during tree search.
    pb_c_base=19652,
    # (float) The initialization constant used in the PUCT formula for balancing exploration and exploitation during tree search.
    pb_c_init=1.25,
    # (float) The maximum change in value allowed during the backup step of the search tree update.
    value_delta_max=0.01,
  )

  @classmethod
  def default_config(cls: type) -> EasyDict:
    cfg = EasyDict(copy.deepcopy(cls.config))
    cfg.cfg_type = cls.__name__ + 'Dict'
    return cfg

  def __init__(self, cfg: EasyDict = None) -> None:
    """
    Overview:
        Use the default configuration mechanism. If a user passes in a cfg with a key that matches an existing key
        in the default configuration, the user-provided value will override the default configuration. Otherwise,
        the default configuration will be used.
    """
    default_config = self.default_config()
    default_config.update(cfg)
    self._cfg = default_config
    # self.inverse_scalar_transform_handle = InverseScalarTransform(
    #     self._cfg.model.support_scale, self._cfg.device, self._cfg.model.categorical_distribution
    # )

  @classmethod
  def roots(cls: int, root_num: int, legal_actions: List[Any]) -> tree.Roots:
    """
    Overview:
        The initialization of CRoots with root num and legal action lists.
    Arguments:
        - root_num: the number of the current root.
        - legal_action_list: the vector of the legal action of this root.
    """
    # import lzero.mcts.ptree.ptree_ez as ptree
    return tree.Roots(root_num, legal_actions)

  def search(
    self,
    roots: Any,
    model: 'Target',
    latent_state_roots: List[Any],
    reward_hidden_state_roots: List[Any] = None, #todo
    to_play: Union[int, List[Any]] = -1
  ) -> None:
    """
    Overview:
        Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference.
        Use the python ctree.
    Arguments:
        - roots (:obj:`Any`): a batch of expanded root nodes
        - latent_state_roots (:obj:`list`): the hidden states of the roots
        - reward_hidden_state_roots (:obj:`list`): the value prefix hidden states in LSTM of the roots
        - to_play (:obj:`list`): the to_play list used in in self-play-mode board games
    """
    with torch.no_grad():
      model.eval()

      # preparation some constant
      batch_size = roots.num
      pb_c_base, pb_c_init, discount_factor = self._cfg.pb_c_base, self._cfg.pb_c_init, self._cfg.discount_factor

      # the data storage of latent states: storing the latent state of all the nodes in one search.
      latent_state_batch_in_search_path = [latent_state_roots]

      # the data storage of value prefix hidden states in LSTM
      # reward_hidden_state_c_batch = [reward_hidden_state_roots[0]]
      # reward_hidden_state_h_batch = [reward_hidden_state_roots[1]]

      # minimax value storage
      min_max_stats_lst = MinMaxStatsList(batch_size)

      for simulation_index in range(self._cfg.num_simulations):
        # In each simulation, we expanded a new node, so in one search, we have ``num_simulations`` num of nodes at most.

        latent_states = []
        # hidden_states_c_reward = []
        # hidden_states_h_reward = []

        # prepare a result wrapper to transport results between python and c++ parts
        results = tree.SearchResults(num=batch_size)

        # latent_state_index_in_search_path: the first index of leaf node states in latent_state_batch_in_search_path, i.e. is current_latent_state_index in one the search.
        # latent_state_index_in_batch: the second index of leaf node states in latent_state_batch_in_search_path, i.e. the index in the batch, whose maximum is ``batch_size``.
        # e.g. the latent state of the leaf node in (x, y) is latent_state_batch_in_search_path[x, y], where x is current_latent_state_index, y is batch_index.
        # The index of value prefix hidden state of the leaf node are in the same manner.
        """
        MCTS stage 1: Selection
            Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
        """
        latent_state_index_in_search_path, latent_state_index_in_batch, last_actions, virtual_to_play = tree.batch_traverse(
          roots, pb_c_base, pb_c_init, discount_factor, min_max_stats_lst, results, copy.deepcopy(to_play)
        )
        # obtain the search horizon for leaf nodes
        search_lens = results.search_lens

        # obtain the latent state for leaf node
        for ix, iy in zip(latent_state_index_in_search_path, latent_state_index_in_batch):
          latent_states.append(latent_state_batch_in_search_path[ix][iy])
          # hidden_states_c_reward.append(reward_hidden_state_c_batch[ix][0][iy])
          # hidden_states_h_reward.append(reward_hidden_state_h_batch[ix][0][iy])

        latent_states = torch.from_numpy(np.asarray(latent_states)).to(self._cfg.device).float()
        # hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(self._cfg.device
        #                                                                                  ).unsqueeze(0)
        # hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(self._cfg.device
        #                                                                                  ).unsqueeze(0)
        # .long() is only for discrete action
        last_actions = np.asarray(last_actions)
        if len(last_actions.shape) == 1:
          last_actions = last_actions.reshape(-1, 1)
        last_actions = torch.from_numpy(last_actions).to(self._cfg.device).long()
        """
        MCTS stage 2: Expansion
            At the final time-step l of the simulation, the next_latent_state and reward/value_prefix are computed by the dynamics function.
            Then we calculate the policy_logits and value for the leaf node (next_latent_state) by the prediction function. (aka. evaluation)
        MCTS stage 3: Backup
            At the end of the simulation, the statistics along the trajectory are updated.
        """
        latent_state, r_logits = model.dynamics_network(latent_states, last_actions)
        policy_logits, v_logits = model.prediction_network(latent_state)
        # hs, r_logits= self.dynamics_network(hs, action_traj[:, i - 1])
        # network_output = model.recurrent_inference(
        #     latent_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions
        # )

        [
            latent_state, policy_logits, value#,
            #network_output.value_prefix
        ] = to_detach_cpu_numpy(
            [
                latent_state,
                policy_logits,
                to_scalar(v_logits),
                # self.inverse_scalar_transform_handle(network_output.value_prefix),
            ]
        )
        # network_output.reward_hidden_state = (
        #     network_output.reward_hidden_state[0].detach().cpu().numpy(),
        #     network_output.reward_hidden_state[1].detach().cpu().numpy()
        # )

        latent_state_batch_in_search_path.append(latent_state)
        # reward_latent_state_batch = network_output.reward_hidden_state
        # tolist() is to be compatible with cpp datatype.
        # value_batch = network_output.value.reshape(-1).tolist()
        value_batch = value.reshape(-1).tolist()
        # value_prefix_batch = network_output.value_prefix.reshape(-1).tolist()
        policy_logits_batch = policy_logits.tolist()

        # reset the hidden states in LSTM every ``lstm_horizon_len`` steps in one search.
        # which enable the model only need to predict the value prefix in a range (e.g.: [s0,...,s5]).
        # assert self._cfg.lstm_horizon_len > 0
        # reset_idx = (np.array(search_lens) % self._cfg.lstm_horizon_len == 0)
        # reward_latent_state_batch[0][:, reset_idx, :] = 0
        # reward_latent_state_batch[1][:, reset_idx, :] = 0
        is_reset_list = [False] * len(search_lens)
        # is_reset_list = reset_idx.astype(np.int32).tolist()
        # reward_hidden_state_c_batch.append(reward_latent_state_batch[0])
        # reward_hidden_state_h_batch.append(reward_latent_state_batch[1])

        # In ``batch_backpropagate()``, we first expand the leaf node using ``the policy_logits`` and
        # ``reward`` predicted by the model, then perform backpropagation along the search path to update the
        # statistics.

        # NOTE: simulation_index + 1 is very important, which is the depth of the current leaf node.
        current_latent_state_index = simulation_index + 1
        value_prefix_batch = [0] * batch_size # todo we dont use this
        tree.batch_backpropagate(
          current_latent_state_index, discount_factor, value_prefix_batch, value_batch, policy_logits_batch,
          min_max_stats_lst, results, is_reset_list, virtual_to_play
        )