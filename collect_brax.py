# Standard library imports
import os
import pprint
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
from flax.linen import nn
from flax.linen.initializers import constant, orthogonal
from flax.serialization import from_bytes
from flax.training.train_state import TrainState as BaseTrainState
from jax.lib import xla_bridge

# Brax imports
from brax import envs
from brax.envs.wrappers.training import AutoResetWrapper, EpisodeWrapper

# Gymnax imports
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper

# Platform check
print(xla_bridge.get_backend().platform)

class TrainState(BaseTrainState):
    update_count: int

class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend, terminate_when_unhealthy=False)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        # env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )

class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        # print("key in clip action", key.shape)
        # print("actio nin clip action", action.shape)
        return self._env.step(key, state, action, params)

def log_eval(log_dict):
    wandb.log(log_dict)
    
class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState

class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state, reward, done, info

@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState

class NormalizeVecReward(GymnaxWrapper):

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        return_val = (state.return_val * self.gamma * (1 - done) + reward)
 
        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info

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
        print("key in log warapper", key.shape)
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
        print("key in log warapper", key.shape)
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
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
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

    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)

    network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
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
        action = jnp.clip(action, -1.0, 1.0)
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

    print("from run data")
    print(actions.shape)
    print(old_obs.shape)
    print(observations.shape)
    print(dones.shape)
    
    session_name = f"brax_{config['ENV_NAME']}_timesteps_{TIMESTEPS}_{str(checkpoint_id)}"
    # if collect_random:
    #     session_name = f"brax_{config['ENV_NAME']}_timesteps_{TIMESTEPS}_random_{str(checkpoint_id)}"
    print(session_name)
    actions_arr = actions
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
    # Show the plot
    plt.savefig(f"data/media/{session_name}.png")
    # plt.show()

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
    
    run_name = "path"

    for id in range(0,242,5):
        print(id)
        artifact_name = "path"+str(id)
        run = api.run(run_name)
        config = run.config
        pprint.pprint(config)
        artifact = api.artifact(artifact_name)
        artifact_dir = artifact.download()
        with open(os.path.join(artifact_dir,"checkpoint.msgpack"), "rb") as infile:
            byte_data = infile.read()
        out = byte_data
        if id == 0:
            collect_policy_data_manual_reset(config,out,id, True)
        else:
            collect_policy_data_manual_reset(config,out,id, False)
    wandb.finish()
