# 🏗️ DaJax: Data Collection in the JAX ecosystem🏗️
 
<p align="center">
    <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
</p>

<h3 align="center">
    <strong>From Policy Weights to Datasets!</strong>
</h3>

## 📊 Data Collection

DaJax provides robust tools for collecting and managing data from reinforcement learning environments. The collection system is built around two main components:

### Collectors
The `collect_` modules provide specialized collectors for different types of data:

- **Trajectory Collection**: Gathers complete sequences of states, actions, and rewards from environment rollouts
- **State-Action Pairs**: Captures specific (state, action) tuples during policy execution
- **Custom Metrics**: Allows collection of user-defined metrics and observations during environment interaction

### Utils Integration
The collectors leverage utility functions for:
- Efficient JAX-based data buffering and storage
- Vectorized environment stepping
- Batched policy evaluation
- Memory-efficient data management

This modular design allows for flexible data collection while maintaining JAX's performance benefits and functional programming paradigm.

### Outputs

- **Episodes** associated to a particular checkpoint during the policy training in the same file
- **Corresponding image** to see that the terminations, observations and actions are aligned

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

👼 just add a `wandb_key` file without any extensions containing the key from the link above. the `.gitignore` is set up to ignore it and ensure the privacy of your key and your data. 

## 📝 To be added

1. **Dataset Format Conversions**:
   - [ ] D4RL format conversion utilities
   - [ ] Minari format conversion support
   - [ ] One-step dynamics training data format

2. **Would be nice**:
   - [ ] Documentation for format conversion workflows

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
