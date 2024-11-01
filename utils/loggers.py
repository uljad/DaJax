import os
import json
from typing import Dict, Any
import wandb
from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)

def save_checkpoint(state, step, ensemble_id=0):
    with open("checkpoint.msgpack", "wb") as outfile:
        outfile.write(msgpack_serialize(to_state_dict(state)))
    artifact = wandb.Artifact(
        f'{wandb.run.name}-checkpoint', type='model'
    )
    artifact.add_file("checkpoint.msgpack")
    wandb.log_artifact(artifact, aliases=["latest", f"step_{step}_ensemble_id+{ensemble_id}"])


def load_checkpoint(api, artifact_name, state):
    wandb.init()
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()

    with open(os.path.join(artifact_dir,"checkpoint.msgpack"), "rb") as infile:
        byte_data = infile.read()
    return from_bytes(state, byte_data)


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration dictionary to a JSON file.
    
    Args:
        config: Dictionary containing configuration settings
        filepath: Path where to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration dictionary from a JSON file.
    
    Args:
        filepath: Path to the JSON configuration file
    
    Returns:
        Dictionary containing configuration settings
    """
    with open(filepath, 'r') as f:
        return json.load(f) 