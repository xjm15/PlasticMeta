import logging
import torch
from torch import nn
from torch.nn import functional as F
import copy

from models.learner import SequencingLearner
from models.utility_manager import UtilityManager
from utils.meta_optimizer import CustomAdam

logger = logging.getLogger('experiment')


class MetaLearner(nn.Module):

    def __init__(self, args, config):
        super(MetaLearner, self).__init__()

        self.update_lr = args["update_lr"]
        self.meta_lr = args["meta_lr"]
        self.utility_management = args["utility_management"]
        self.device = torch.device(f'cuda:{args["rank"] % args["gpus"]}' if torch.cuda.is_available() else 'cpu')

        self.net = SequencingLearner(config).to(self.device)
        self.optimizers = []

        forward_meta_weights = self.net.get_forward_meta_parameters()
        if len(forward_meta_weights) > 0:
            self.optimizer_meta = CustomAdam(forward_meta_weights, lr=self.meta_lr)
            self.optimizers.append(self.optimizer_meta)
        else:
            logger.warning("Zero meta parameters in the forward pass")

        if self.utility_management:
            param_list = list(self.net.parameters())
            managed_indices = [0, 1, 8, 9, 12, 13]
            param_list = list(self.net.parameters())
            original_managed_keys = [param_list[i] for i in managed_indices]
            managed_params_initial = [copy.deepcopy(p) for p in original_managed_keys]
            managed_params = [param_list[i] for i in managed_indices]

            self.utility_manager = UtilityManager(
                net_params=managed_params_initial,
                original_params_keys=original_managed_keys,
                hidden_activation='relu',
                opt=self.optimizer_meta,
                replacement_rate=args.get("replacement_rate", 1e-5),
                decay_rate=args.get("decay_rate", 0.99),
                maturity_threshold=args.get("maturity_threshold", 20),
                device=self.device,
                utility_type='adaptation',
                accumulate=False
            )
            self.managed_indices = managed_indices
        self.log_model()

    def log_model(self):
        for name, param in self.net.named_parameters():
            if param.meta:
                logger.info(f"Weight in meta-optimizer = {name} {param.shape}")
            if hasattr(param, 'adaptation') and param.adaptation:
                logger.debug(f"Weight for adaptation = {name} {param.shape}")

    def optimizer_zero_grad(self):
        for opti in self.optimizers:
            opti.zero_grad()

    def optimizer_step(self):
        for opti in self.optimizers:
            opti.step()

    def inner_update(self, net, vars, grad, adaptation_lr):
        adaptation_weight_counter = 0
        new_weights = []

        for p in vars:
            if hasattr(p, 'adaptation') and p.adaptation:
                g = grad[adaptation_weight_counter]
                temp_weight = p - adaptation_lr * g

                temp_weight.adaptation = p.adaptation
                temp_weight.meta = p.meta

                new_weights.append(temp_weight)
                adaptation_weight_counter += 1
            else:
                new_weights.append(p)

        return new_weights

    def clip_grad(self, grad, norm=10):
        grad_clipped = []
        for g in grad:
            g = (g * (g < norm).float()) + ((g > norm).float()) * norm
            g = (g * (g > -norm).float()) - ((g < -norm).float()) * norm
            grad_clipped.append(g)
        return grad_clipped

    def forward(self, x_traj, y_traj, x_rand, y_rand):
        mean, log_std = self.net(x_traj[0, 0, :, :].unsqueeze(0), vars=None)
        outputs = torch.tanh(mean)
        loss = F.mse_loss(outputs, y_traj[0, 0, :].unsqueeze(0))

        grad = self.clip_grad(torch.autograd.grad(loss, self.net.get_adaptation_parameters(), create_graph=True))
        fast_weights = self.inner_update(self.net, self.net.parameters(), grad, self.update_lr)

        with torch.no_grad():
            mean, log_std = self.net(x_rand[0], vars=None)
            outputs = torch.tanh(mean)
            first_loss = F.mse_loss(outputs, y_rand[0])

        for k in range(1, len(x_traj)):
            mean, log_std = self.net(x_traj[k, 0, :, :].unsqueeze(0), fast_weights)
            outputs = torch.tanh(mean)
            loss = F.mse_loss(outputs, y_traj[k, 0, :].unsqueeze(0))

            grad = self.clip_grad(torch.autograd.grad(loss, self.net.get_adaptation_parameters(fast_weights),
                                                      create_graph=True))

            fast_weights = self.inner_update(self.net, fast_weights, grad, self.update_lr)

        mean, log_std, features = self.net.utility_forward(x_rand[0], fast_weights)
        outputs = torch.tanh(mean)
        final_meta_loss = F.mse_loss(outputs, y_rand[0])

        self.optimizer_zero_grad()
        final_meta_loss.backward()
        self.optimizer_step()

        if self.utility_management:
            managed_params_current = [p for i, p in enumerate(fast_weights) if i in self.managed_indices]
            self.utility_manager.update_parameters(managed_params_current)
            new_managed_params = self.utility_manager.manage_utility_and_regenerate(features=features)

            for i, p_idx in enumerate(self.managed_indices):
                self.net.vars[p_idx].data = new_managed_params[i].data

        return [first_loss.detach(), final_meta_loss.detach()], features