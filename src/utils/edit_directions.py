import os
import torch


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
    if task_name=="cat2dog":    
        emb_dir = f"assets/embeddings_sd_1.4"
        embs_a = torch.load(os.path.join(emb_dir, f"cat.pt"))
        embs_b = torch.load(os.path.join(emb_dir, f"dog.pt"))
        return (embs_b.mean(0)-embs_a.mean(0)).unsqueeze(0)
    elif task_name=="dog2cat":    
        emb_dir = f"assets/embeddings_sd_1.4"
        embs_a = torch.load(os.path.join(emb_dir, f"dog.pt"))
        embs_b = torch.load(os.path.join(emb_dir, f"cat.pt"))
        return (embs_b.mean(0)-embs_a.mean(0)).unsqueeze(0)
    else:
        raise NotImplementedError
