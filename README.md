# 🏗️ DaJax: Data Collection in the JAX ecosystem🏗️
 
<p align="center">
    <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
</p>

<h3 align="center">
    <strong>From Policy Weights to Datasets!</strong>
</h3>

## 📊 Data Collection

DaJax provides robust tools for collecting and managing data from reinforcement learning environments.

 **Tools to help you or your LLM of choice** to write better and more helpful data collection scripts:

<p align="center">
  <img src="animation.gif" alt="Hopper Data" width="480" height="360">
</p>
<p align="center">
  <em>Hopper Rollouts for 1.1 episodes at various policy checkpoint from randomly initialized policy</em>
</p>

### Use

To get policy rollouts using an Actor-Critic network:

[`collect_brax.py`](collect_brax.py) for environments based on the [Brax Physics Enginge](https://github.com/google/brax/tree/main)

[`collect_discrete.py`](collect_discrete.py) for discrete (categorical) action space on environments using [Gymnax](https://github.com/RobertTLange/gymnax/tree/main)

[`collect_continuous.py`](collect_continuous.py) for a continuous (multivariate Gaussian) action space on environments using [Gymnax](https://github.com/RobertTLange/gymnax/tree/main)





### Utils Integration
The collectors leverage utility functions for:
- Efficient JAX-based data buffering and storage
- Vectorized environment stepping
- Batched policy evaluation


This modular design allows for flexible data collection while maintaining JAX's performance benefits and functional programming paradigm.

### Outputs

- **Episodes** associated to a particular checkpoint during the policy training in the same file
- **Corresponding image** to see that the terminations, observations and actions are aligned

### Examples

The files in the main directory save the results here:

[**CSV files**](data/expert_data/) - each row is a tuple of  $(a_t, o_t, o_{t+1}, \text{Done}, r_{t+1})$ where:
- $a_t$ is the action at time $t$
- $o_t$ is the observation at time $t$
- $o_{t+1}$ is the next observation
- $\text{Done}$ is the terminal state flag
- $r_{t+1}$ is the reward received
 

[**Verification Media**](data/media/) - media to check that the data makes sense



## 🔩 Installation

You can install DaJax and its dependencies using pip. Install all dependencies at once using:

### Basic Installation
```bash
pip install -r setup/requirements-base.txt
```

### For CUDA Support
```bash
pip install -r setup/docker/requirements-cuda.txt
```

### For CPU-only
```bash
pip install -r setup/docker/requirements-cpu.txt
```

## 🐳 Running Via Docker

1. **Build the Docker container** with the provided script:
```
cd setup/docker && ./build.sh
```
2. **Add your [WandB key](https://wandb.ai/authorize)** to the `lstm/setup/docker` folder:

```
echo <wandb_key> > setup/docker/wandb_key
```
## 🐍 Running Via Conda

```
conda env create -f setup/environment.yml
```



👼 just add a `wandb_key` file without any extensions containing the key from the link above. the `.gitignore` is set up to ignore it and ensure the privacy of your key and your data. 

## 📝 To be added

1. **Dataset Format Conversions**:
   - [ ] D4RL format conversion utilities
   - [ ] Minari format conversion support
   - [ ] One-step dynamics training data format

2. **Would be nice**:
   - [ ] Documentation for format conversion workflows
   - [ ] Weights and Biases Integration

 😬 I will reorganize soon once I receive more feedback regarding the best ways people like to use such tools. 

##  Acknowledgement

These tools are based on the following:

🚀 **[Jax Ecosystem](https://github.com/jax-ml/jax_)** 

💪 **[Gymnax](https://github.com/RobertTLange/gymnax)** 

🌟 **[PureJaxRL](https://github.com/luchris429/purejaxrl/tree/main)** 

## Citation

If you use DaJax in your research, please cite:

```bibtex
@software{dajax2024,
  author       = {Uljad Berdica},
  title        = {DaJax: Data Collection in the JAX ecosystem},
  year         = {2024},
  publisher    = {GitHub},
  url          = {https://github.com/rodrigodelazcano/DaJax},
  description  = {A JAX-based library for collecting and managing reinforcement learning datasets}
}
```
