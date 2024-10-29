import wandb
from flax.serialization import (
    to_state_dict, msgpack_serialize, from_bytes
)
import os 


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


