from itertools import islice
import networkx as nx
import numpy as np


class Path:

    def __init__(self, path_id, node_list, length, best_modulation=None):
        self.path_id = path_id
        self.node_list = node_list
        self.length = length
        self.best_modulation = best_modulation
        self.hops = len(node_list) - 1


class Service:

    def __init__(self, service_id, source, source_id, destination=None, destination_id=None, arrival_time=None,
                 holding_time=None, bit_rate=None, best_modulation=None, service_class=None, number_slots=None):
        self.service_id = service_id
        self.arrival_time = arrival_time
        self.holding_time = holding_time
        self.source = source
        self.source_id = source_id
        self.destination = destination
        self.destination_id = destination_id
        self.bit_rate = bit_rate
        self.service_class = service_class
        self.best_modulation = best_modulation
        self.number_slots = number_slots
        self.route = None
        self.initial_slot = None
        self.accepted = False

    def __str__(self):
        msg = '{'
        msg += '' if self.bit_rate is None else f'br: {self.bit_rate}, '
        msg += '' if self.service_class is None else f'cl: {self.service_class}, '
        return f'Serv. {self.service_id} ({self.source} -> {self.destination})' + msg


def start_environment(env, steps):
    done = True
    for i in range(steps):
        if done:
            env.reset()
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
    return env


def get_k_shortest_paths(G, source, target, k, weight=None):
    """
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    """

    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def get_k_shortest_paths_disjoint(G,source,target,k,weight=None):
    # metodo parecido al get shortest path pero solo se busca las rutas disjuntas
    return list(islice(nx.edge_disjoint_paths(G,source,target),k))

def mix_disjoint_edge(G,s,t,k):
  A=get_k_shortest_paths(G,s,t,k)
  A1=get_k_shortest_paths_disjoint(G,s,t,k)
  C=[]
  for i in A:
    for j in A1:
      if(i==j):
        C.append(i)
  return list(C)  

def mix_disjoint_edge(G,s,t,k):
  A=get_k_shortest_paths(G,s,t,k)
  A1=get_k_shortest_paths_disjoint(G,s,t,k)
  C=[]
  for i in A:
    for j in A1:
      if(i==j):# rutas disjuntas
        C.append(i)
      elif j not in A:# agregar las rutas restantes de 
        C.append(j)
      elif i not in A1:
        C.append(i)
  return list(C) 

def get_path_weight(graph, path, weight='length'):
    return np.sum([graph[path[i]][path[i + 1]][weight] for i in range(len(path) - 1)])


def random_policy(env):
    return env.action_space.sample()


def evaluate_heuristic(env, heuristic, n_eval_episodes=10,
                       render=False, callback=None, reward_threshold=None,
                       return_episode_rewards=False):
    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action = heuristic(env)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, 'Mean reward below threshold: ' \
                                               '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
