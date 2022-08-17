import gym
import numpy as np

from .rmsa_env import RMSAEnv
from .optical_network_env import OpticalNetworkEnv


class DeepRMSAEnv(RMSAEnv):

    def __init__(self, topology=None, j=1,
                 episode_length=1000,
                 mean_service_holding_time=25.0,
                 mean_service_inter_arrival_time=.1,
                 num_spectrum_resources=200,
                 node_request_probabilities=None,
                 seed=None,
                 k_paths=5,
                 allow_rejection=True):
        super().__init__(topology=topology,
                         episode_length=episode_length,
                         load=mean_service_holding_time / mean_service_inter_arrival_time,
                         mean_service_holding_time=mean_service_holding_time,
                         num_spectrum_resources=num_spectrum_resources,
                         node_request_probabilities=node_request_probabilities,
                         seed=seed,
                         k_paths=k_paths,
                         allow_rejection=allow_rejection,
                         reset=False)
        
        self.j = j
        shape = 1 + 2 * self.topology.number_of_nodes() + (2 * self.j + 3) * self.k_paths
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.uint8, shape=(shape,))
        self.action_space = gym.spaces.Discrete(self.k_paths * self.j + self.reject_action)
        self.action_space.seed(self.rand_seed)
        self.observation_space.seed(self.rand_seed)
        self.reset(only_counters=False)

    def step(self, action: int):
        #print(action)
        if self.link is None:
                                 
            if action[1] < self.k_paths * self.j:  # action is for assigning a path
                path, block = self._get_path_block_id(action)
    
                initial_indices, lengths = self.get_available_blocks(path)
                if block < len(initial_indices):
                    return super().step([path, initial_indices[block]])
                else:
                    return super().step([self.k_paths, self.num_spectrum_resources])
            else:
                return super().step([self.k_paths, self.num_spectrum_resources])
        else:
            if action[0] < self.k_paths * self.j:
                path, block = self._get_path_block_id(action)
    
                initial_indices, lengths = self.get_available_blocks(path)
                if block < len(initial_indices):
                    return super().step([path, initial_indices[block]])
                else:
                    return super().step([self.k_paths, self.num_spectrum_resources])
            else:
                return super().step([self.k_paths, self.num_spectrum_resources])
    def observation(self):
        # observation space defined as in https://github.com/xiaoliangchenUCD/DeepRMSA/blob/eb2f2442acc25574e9efb4104ea245e9e05d9821/DeepRMSA_Agent.py#L384
        source_destination_tau = np.zeros((2, self.topology.number_of_nodes()))
        min_node = min(self.service.source_id, self.service.destination_id)
        max_node = max(self.service.source_id, self.service.destination_id)
        source_destination_tau[0, min_node] = 1
        source_destination_tau[1, max_node] = 1
        spectrum_obs = np.full((self.k_paths, 2 * self.j + 3), fill_value=-1.)
        for idp, path in enumerate(self.k_shortest_paths[self.service.source, self.service.destination]):
            available_slots = self.get_available_slots(path)
            num_slots = self.get_number_slots(path)
            initial_indices, lengths = self.get_available_blocks(idp)

            for idb, (initial_index, length) in enumerate(zip(initial_indices, lengths)):
                # initial slot index
                spectrum_obs[idp, idb * 2 + 0] = 2 * (initial_index - .5 * self.num_spectrum_resources) / self.num_spectrum_resources

                # number of contiguous FS available
                spectrum_obs[idp, idb * 2 + 1] = (length - 8) / 8
            spectrum_obs[idp, self.j * 2] = (num_slots - 5.5) / 3.5 # number of FSs necessary

            idx, values, lengths = DeepRMSAEnv.rle(available_slots)

            av_indices = np.argwhere(values == 1) # getting indices which have value 1
            spectrum_obs[idp, self.j * 2 + 1] = 2 * (np.sum(available_slots) - .5 * self.num_spectrum_resources) / self.num_spectrum_resources # total number available FSs
            spectrum_obs[idp, self.j * 2 + 2] = (np.mean(lengths[av_indices]) - 4) / 4 # avg. number of FS blocks available
        bit_rate_obs = np.zeros((1, 1))
        bit_rate_obs[0, 0] = self.service.bit_rate / 100

        return np.concatenate((bit_rate_obs, source_destination_tau.reshape((1, np.prod(source_destination_tau.shape))),
                               spectrum_obs.reshape((1, np.prod(spectrum_obs.shape)))), axis=1)\
            .reshape(self.observation_space.shape)

    def reward(self):
        return 1 if self.service.accepted else 0

    def reset(self, only_counters=True):
        return super().reset(only_counters=only_counters)

    def _get_path_block_id(self, action: int) -> (int, int):
        if self.link is None:# and self.allow_rejection is not False:
            path = action[1] // self.j
            block = action[1] % self.j
        else:
            path = action[1] // self.j
            block = action[1] % self.j
        return path, block


def shortest_path_first_fit(env: DeepRMSAEnv) -> int:
    if not env.allow_rejection:
        return 0
    else:
        initial_indices, lengths = env.get_available_blocks(0)
        if len(initial_indices) > 0:  # if there are available slots
            return 0
        else:
            return env.k_paths * env.j


def shortest_available_path_first_fit(env: DeepRMSAEnv) -> int:
    for idp, path in enumerate(env.k_shortest_paths[env.service.source, env.service.destination]):
        initial_indices, lengths = env.get_available_blocks(idp)
        if len(initial_indices) > 0: # if there are available slots
            return idp * env.j # this path uses the first one
    return env.k_paths * env.j


def Path_protecttion_shortest_available_path_first_fit(env: DeepRMSAEnv) -> int:
    for idp,path in enumerate(env.k_shortest_paths[env.service.source,env.service.destination]):
        Prot_path=env.k_shortest_paths[env.service.source,env.service.destination][path+1]# le indico el protection path
        initial_indices, lengths = env.get_available_blocks(idp)# los  fs disponibles del working path
        num_slots = env.get_number_slots(path)# se obtieine el numero de slot necesarias para envar el working path
        if path == Prot_path:
            Prot_path=env.k_shortest_paths[env.service.source,env.service.destination][path-1]
        initial_indices_2, lengths_2=env.get_available_blocks(env.k_shortest_paths[env.service.source, env.service.destination].index(Prot_path))#
        num_slots_2=env.get_number_slots(Prot_path)
        for initial_slot in range(0, env.topology.graph['available_slots'] - num_slots - num_slots_2):
            if len(initial_indices)>0 and env.is_path_free(path, initial_slot, num_slots) :
                print("okis")
                return idp * env.j
            elif len(initial_indices_2) >0 and env.is_path_free(Prot_path, initial_slot, num_slots_2):
                print("NO-okis")
                idp=env.k_shortest_paths[env.service.source, env.service.destination].index(Prot_path)
                return env.k_shortest_paths[env.service.source, env.service.destination].index(Prot_path) * env.j 
            else:
                return 0
    #return  env.k_paths * env.j 
    
    
def one_plus_one__available_path(env: DeepRMSAEnv) -> int:
    for idp,path in enumerate(env.k_shortest_paths[env.service.source,env.service.destination]):
        Prot_path=env.k_shortest_paths[env.service.source,env.service.destination][path+1]# le indico el protection path
        initial_indices, lengths = env.get_available_blocks(idp)# los  fs disponibles del working path
        num_slots = env.get_number_slots(path)# se obtieine el numero de slot necesarias para envar el working path
        if path == Prot_path:
            Prot_path=env.k_shortest_paths[env.service.source,env.service.destination][path-1]
        initial_indices_2, lengths_2=env.get_available_blocks(env.k_shortest_paths[env.service.source, env.service.destination].index(Prot_path))#
        num_slots_2=env.get_number_slots(Prot_path)
        for initial_slot in range(0, env.topology.graph['num_spectrum_resources'] - num_slots - num_slots_2):
            if len(initial_indices)>0 and env.is_path_free(path, initial_slot, num_slots) and  len(initial_indices_2) >0 and env.is_path_free(Prot_path, initial_slot, num_slots_2):
                return [idp * env.j , env.k_shortest_paths[env.service.source, env.service.destination].index(Prot_path) * env.j ]
            else:
                return 0
    