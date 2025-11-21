import sys
import torch
from math import sqrt
import torch.nn.functional as F
import copy


class UtilityManager(object):
    def __init__(
            self,
            net_params,
            original_params_keys,
            hidden_activation,
            opt,
            decay_rate=0.99,
            replacement_rate=1e-4,
            init='kaiming',
            device="cpu",
            maturity_threshold=20,
            utility_type='adaptation',
            accumulate=False,
    ):
        super(UtilityManager, self).__init__()
        self.device = torch.device(device)
        self.net_params = net_params
        self.original_params_keys = original_params_keys
        self.num_layers_to_manage = len(self.net_params) // 2
        self.accumulate = accumulate

        self.opt = opt
        self.opt_type = 'sgd'
        if 'Adam' in type(self.opt).__name__:  # Generic check for Adam based on name
            self.opt_type = 'adam'

        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.utility_type = utility_type


        self.util = [torch.zeros(self.net_params[i * 2].shape[0]).to(self.device) for i in
                     range(self.num_layers_to_manage)]
        self.bias_corrected_util = \
            [torch.zeros(self.net_params[i * 2].shape[0]).to(self.device) for i in range(self.num_layers_to_manage)]
        self.ages = [torch.zeros(self.net_params[i * 2].shape[0]).to(self.device) for i in
                     range(self.num_layers_to_manage)]
        self.mean_feature_act = [torch.zeros(self.net_params[i * 2].shape[0]).to(self.device) for i in
                                 range(self.num_layers_to_manage)]
        self.accumulated_num_features_to_replace = [0 for i in range(self.num_layers_to_manage)]

        self.bounds = self._compute_bounds(hidden_activation=hidden_activation, init=init)

    def update_parameters(self, net_params):
        self.net_params = [copy.deepcopy(net_params[i]).to(self.device) for i in range(len(net_params))]

    def _compute_bounds(self, hidden_activation, init='kaiming'):
        if hidden_activation in ['swish', 'elu']:
            hidden_activation = 'relu'

        bounds = []
        for i in range(self.num_layers_to_manage):
            input_dim = self.net_params[i * 2].shape[1]
            if init == 'xavier':
                gain = torch.nn.init.calculate_gain(nonlinearity=hidden_activation)
                bound = gain * sqrt(6 / (input_dim + self.net_params[i * 2].shape[0]))
            elif init == 'lecun':
                bound = sqrt(3 / input_dim)
            else:
                gain = torch.nn.init.calculate_gain(nonlinearity=hidden_activation)
                bound = gain * sqrt(3 / input_dim)
            bounds.append(bound)

        return bounds

    def _update_utility(self, layer_idx, features):
        with torch.no_grad():
            self.util[layer_idx] *= self.decay_rate
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

            current_weight = self.net_params[layer_idx * 2]
            input_weight_mag = current_weight.abs().mean(dim=1)

            if self.utility_type == 'adaptation':
                new_util = 1 / (input_weight_mag + 1e-8)
            else:
                new_util = 0

            self.util[layer_idx] += (1 - self.decay_rate) * new_util

            self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction

    def assess_plasticity(self, features):
        features_to_replace = [torch.empty(0, dtype=torch.long).to(self.device) for _ in
                               range(self.num_layers_to_manage)]
        num_features_to_replace = [0 for _ in range(self.num_layers_to_manage)]

        if self.replacement_rate == 0:
            return features_to_replace, num_features_to_replace

        for i in range(self.num_layers_to_manage):
            self.ages[i] += 1

            self._update_utility(layer_idx=i, features=features[i])

            eligible_feature_indices = torch.where(self.ages[i] > self.maturity_threshold)[0]

            if eligible_feature_indices.shape[0] == 0:
                continue

            num_to_replace_float = self.replacement_rate * eligible_feature_indices.shape[0]
            self.accumulated_num_features_to_replace[i] += num_to_replace_float

            if self.accumulate:
                num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
                self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace
            else:
                if num_to_replace_float < 1:
                    if torch.rand(1) <= num_to_replace_float:
                        num_new_features_to_replace = 1
                    else:
                        num_new_features_to_replace = 0
                else:
                    num_new_features_to_replace = int(num_to_replace_float)

            if num_new_features_to_replace == 0:
                continue

            new_features_to_replace = torch.topk(-self.bias_corrected_util[i][eligible_feature_indices],
                                                 num_new_features_to_replace)[1]
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            self.util[i][new_features_to_replace] = 0
            self.mean_feature_act[i][new_features_to_replace] = 0.
            self.ages[i][new_features_to_replace] = 0

            features_to_replace[i] = new_features_to_replace
            num_features_to_replace[i] = num_new_features_to_replace

        return features_to_replace, num_features_to_replace

    def generate_new_features(self, features_to_replace, num_features_to_replace):
        with torch.no_grad():
            for i in range(self.num_layers_to_manage):
                if num_features_to_replace[i] == 0:
                    continue

                current_weight = self.net_params[i * 2]
                current_bias = self.net_params[i * 2 + 1]

                current_weight[features_to_replace[i], :] *= 0.0
                current_weight[features_to_replace[i], :] += \
                    torch.empty(num_features_to_replace[i], current_weight.shape[1]).uniform_(
                        -self.bounds[i], self.bounds[i]).to(self.device)

                current_bias[features_to_replace[i]] *= 0

        return self.net_params

    def update_optimizer_state(self, features_to_replace, num_features_to_replace):
        if self.opt_type == 'adam':
            for i in range(self.num_layers_to_manage):
                if num_features_to_replace[i] == 0:
                    continue

                w_param = self.net_params[i * 2]
                b_param = self.net_params[i * 2 + 1]

                if w_param not in self.opt.state:
                    continue

                self.opt.state[w_param]['exp_avg'][features_to_replace[i], :] = 0.0
                self.opt.state[w_param]['exp_avg_sq'][features_to_replace[i], :] = 0.0
                if 'step' in self.opt.state[w_param]:
                    self.opt.state[w_param]['step'][features_to_replace[i], :] = 0

                if b_param not in self.opt.state:
                    continue
                self.opt.state[b_param]['exp_avg'][features_to_replace[i]] = 0.0
                self.opt.state[b_param]['exp_avg_sq'][features_to_replace[i]] = 0.0
                if 'step' in self.opt.state[b_param]:
                    self.opt.state[b_param]['step'][features_to_replace[i]] = 0


    def manage_utility_and_regenerate(self, features):
        features_to_replace, num_features_to_replace = self.assess_plasticity(features=features)
        new_params = self.generate_new_features(features_to_replace, num_features_to_replace)
        self.update_optimizer_state(features_to_replace, num_features_to_replace)
        return new_params