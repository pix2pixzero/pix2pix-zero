import os
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


"""
This function takes in a task name and returns the direction in the embedding space that transforms class A to class B for the given task.

Parameters:
task_name (str): name of the task for which direction is to be constructed.

Returns:
torch.Tensor: A tensor representing the direction in the embedding space that transforms class A to class B.

Examples:
>>> construct_direction("cat2dog")
"""
def construct_direction(task_name):
    (src, dst) = task_name.split("2")
    emb_dir = f"assets/embeddings_sd_1.4"
    embs_a = torch.load(os.path.join(emb_dir, f"{src}.pt"), map_location=device)
    embs_b = torch.load(os.path.join(emb_dir, f"{dst}.pt"), map_location=device)
    return (embs_b.mean(0)-embs_a.mean(0)).unsqueeze(0)
