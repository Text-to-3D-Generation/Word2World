import torch
import torch.nn as nn
from torch.optim import Optimizer

class OptimizerManager:
    def __init__(self, optimizer):
        self.optimizer = optimizer  # Placeholder for the optimizer instance

    def _modify_optimizer_params(self, modify_fn):
        """
        Generalized function to modify optimizer parameters (pruning or densifying).

        Args:
            modify_fn (callable): Function that takes (param_tensor, stored_state, group_name)
            and modifies them accordingly.

        Returns:
            dict: A dictionary of updated parameters.
        """
        updated_params = {}
        for group in self.optimizer.param_groups:
            param_tensor = group["params"][0]
            #print(group['name'])
            stored_state = self.optimizer.state.get(param_tensor, None)
            new_param_tensor, new_stored_state = modify_fn(param_tensor, stored_state, group)
            if stored_state is not None:
                del self.optimizer.state[param_tensor]
                self.optimizer.state[new_param_tensor] = new_stored_state
            group["params"][0] = new_param_tensor
            updated_params[group["name"]] = new_param_tensor

        return updated_params

    def prune_optimizer(self, mask):
        """
        Prune optimizer parameters based on a boolean mask indicating which parameters to KEEP.
        """
        def prune_fn(param_tensor, stored_state, group):
            new_param_tensor = nn.Parameter(param_tensor[mask].requires_grad_(True))
            if stored_state:
                for key in stored_state.keys():
                    if stored_state[key].ndim > 0:
                        stored_state[key] = stored_state[key][mask]
            return new_param_tensor, stored_state
        return self._modify_optimizer_params(prune_fn)

    def densify_on_optimizer(self, new_params_dict):
        """
        Extend optimizer parameters by concatenating new tensors.
        """
        def densify_fn(param_tensor, stored_state, group):
            extension_tensor = new_params_dict[group["name"]]
            print(group["name"])
            print(param_tensor.shape)
            print(extension_tensor.shape)
            new_param_tensor = nn.Parameter(torch.cat((param_tensor, extension_tensor), dim=0).requires_grad_(True))
            if stored_state:
                for key in stored_state.keys():
                    if stored_state[key].ndim > 0:
                        stored_state[key] = torch.cat((stored_state[key], torch.zeros_like(extension_tensor)), dim=0)
            return new_param_tensor, stored_state
        return self._modify_optimizer_params(densify_fn)