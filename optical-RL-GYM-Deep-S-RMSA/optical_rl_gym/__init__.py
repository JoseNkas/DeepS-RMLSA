from gym.envs.registration import register

register(
    id='RBMLSA-v0',
    entry_point='optical_rl_gym.envs:RBMLSAEnv',
)

register(
    id='DeepRBMLSA-v0',
    entry_point='optical_rl_gym.envs:DeepRBMLSAEnv',
)

register(
    id='RMSCA-v0',
    entry_point='optical_rl_gym.envs:RMSCAEnv',
)

register(
    id='DeepRMSA-v0',
    entry_point='optical_rl_gym.envs:DeepRMSAEnv',
)

register(
    id='RWA-v0',
    entry_point='optical_rl_gym.envs:RWAEnv',
)

register(
    id='QoSConstrainedRA-v0',
    entry_point='optical_rl_gym.envs:QoSConstrainedRA',
)

register(
    id='RMSA-v0',
    entry_point='optical_rl_gym.envs:RMSAEnv',
)
register(
    id='SRMSA-v0',
    entry_point='optical_rl_gym.envs:SRMSAEnv',
)