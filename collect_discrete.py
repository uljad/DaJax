# Standard library imports
import os
from functools import partial
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

# Third-party imports
import chex
import distrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from flax import struct
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.serialization import from_bytes
from flax.training.train_state import TrainState as BaseTrainState
from jax.lib import xla_bridge

# Brax imports
from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper

# Gymnax imports
import gymnax
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import (
    FlattenObservationWrapper,
    GymnaxWrapper,
    LogWrapper
)

#loggers
from utils.loggers import save_config, load_config

# Platform check
print(xla_bridge.get_backend().platform)


class TrainState(BaseTrainState):
    update_count: int

@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info
    
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def collect_policy_data_manual_reset(config,out, checkpoint_id, collect_random=False):

    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    network = ActorCritic(env.action_space(env_params).n, activation=config["ACTIVATION"])

    config["seed"] = 101
    rng = jax.random.PRNGKey(config["seed"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    initial_params = network.init(_rng, init_x)
    network_params = from_bytes(initial_params,out)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    init_obsv, init_state = env.reset(_rng, env_params)
    TIMESTEPS = 10000
    
    if not collect_random:
        runner_state = (rng, network_params, init_obsv, init_state, env_params)
    if collect_random:
        runner_state = (rng, initial_params, init_obsv, init_state, env_params)

    def _env_step(runner_state, run_data):
        rng, params, obsv, env_state, env_params= runner_state
        old_obs = obsv
        pi, _ = network.apply(params, obsv)
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed = _rng)
        rng, _rng = jax.random.split(rng)
        obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
        rng, _rng = jax.random.split(rng)
        init_obsv, init_state = env.reset(_rng, env_params)
        rng, _rng = jax.random.split(rng)
        state = jax.tree_map(
            lambda x, y: jnp.where(done, x, y), init_state, env_state
        )
        obs = jax.lax.select(done, init_obsv, obsv)
        runner_state = (rng, params, obs, state, env_params)
        return runner_state, (action, old_obs, obs, reward, done)
    
    runner_state, run_data = jax.lax.scan(_env_step, runner_state, None, length=TIMESTEPS)

    actions = run_data[0]
    old_obs = run_data[1]
    observations = run_data[2]
    rewards = run_data[3]
    dones = run_data[4]

    session_name = f"{config['ENV_NAME']}_timesteps_{TIMESTEPS}_{str(checkpoint_id)}"
    print(session_name)
    actions_arr = jnp.expand_dims(actions, axis=1)
    old_obs_arr = old_obs
    observations_arr = observations
    dones_arr = dones
    rewards_arr = rewards

    '''
    Plotting the actions, observations and done flags
    '''
    FINAL_INDEX = 2200
    # Createa figure and a set of subplots
    fig, axs = plt.subplots(4, 1,figsize=(10, 5)) # 1 row, 2 columns
    # Plot for actions
    axs[0].plot(actions[0:FINAL_INDEX])
    axs[0].set_title('Actions')  # Subtitle for the first plot
    # Plot for observations
    for i in range(1):
        axs[1].plot(observations[0:FINAL_INDEX, i], label=f'{i}')
    axs[1].set_title('Observations') # Subtitle for the second plot
    axs[1].legend()
    axs[2].set_title('Done Flags')
    axs[2].plot(dones[0:-1])
    axs[3].set_title('Rewards')
    axs[3].plot(rewards[0:FINAL_INDEX])
    plt.tight_layout()
    plt.savefig(f"data/media/{session_name}.png")

    '''
    Saving to a CSV for the World Model Training
    '''
    df1 = pd.DataFrame(actions_arr, columns=[f'Action_{i}' for i in range(actions_arr.shape[1])])
    df1_2 = pd.DataFrame(old_obs_arr, columns=[f'Old{i}' for i in range(old_obs_arr.shape[1])])
    df2 = pd.DataFrame(observations_arr, columns=[f'Observation_{i}' for i in range(observations_arr.shape[1])])
    df3 = pd.DataFrame(dones_arr, columns=['Done'])
    df4 = pd.DataFrame(rewards_arr, columns=['Reward'])
    # Concatenate DataFrames horizontally
    final_df = pd.concat([df1, df1_2, df2, df3, df4], axis=1)
    final_df.to_csv(f"data/expert_data/{session_name}.csv", index=False)

if __name__=="__main__":

    wandb.login()
    api = wandb.Api()
    
    # put the wandb run_name here to use your own artifact
    run_name = ""
    artifact_name = ""
     
    for id in range(0,210,50):
        if not run_name or not artifact_name:
            print("========================Running Example============================")
            ENV_NAME = "CartPole-v1"
            config = load_config(f"configs/config_{ENV_NAME}.json")
            with open(f"artifacts/{ENV_NAME}/checkpoint:v{id}/checkpoint.msgpack", "rb") as infile:
                byte_data = infile.read()
        else:
            print("========================You need to specify run and artifact name============================")
            artifact_path = "/".join(run_name.split("/")[:-1])+"/"+artifact_name+":v"+str(id)
            run = api.run(run_name)
            config = run.config
            save_config(config, f"configs/config_{config['ENV_NAME']}.json")
            artifact = api.artifact(artifact_path)
            artifact_dir = artifact.download()
            with open(os.path.join(artifact_dir,"checkpoint.msgpack"), "rb") as infile:
                byte_data = infile.read()
            
        out = byte_data
        if id == 0:
            collect_policy_data_manual_reset(config,out,id, True)
        else:
            collect_policy_data_manual_reset(config,out,id, False)
        
    wandb.finish()