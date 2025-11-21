import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from metatorch import nn as mnn  # Renamed import

logger = logging.getLogger("experiment")


class SequencingLearner(nn.Module):
    def __init__(self, learner_configuration):
        super(SequencingLearner, self).__init__()

        self.config = learner_configuration
        self.vars = nn.ParameterList()

        self.vars = self._parse_config(self.config, nn.ParameterList())
        self.context_backbone = None

        for con in self.config:
            if 'Encoding' in con["name"]:
                max_len = 50
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, con["d_model"], 2).float() * (-np.log(10000.0) / con["d_model"]))
                pe = torch.zeros(max_len, con["d_model"])
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe)
            if 'Attention' in con["name"]:
                self.num_heads = con['config']['num_heads']
                self.d_k = con['config']['d_model'] // con['config']['num_heads']

    def _parse_config(self, config, vars_list):

        for i, info_dict in enumerate(config):

            if info_dict["name"] == 'linear':
                param_config = info_dict["config"]
                w, b = mnn.linear(param_config["out"], param_config["in"], info_dict["adaptation"],
                                  info_dict["meta"])

                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] == 'doublelinear':
                param_config = info_dict["config"]
                w, b = mnn.linear(param_config["out"], param_config["in"], info_dict["adaptation"],
                                  info_dict["meta"])

                vars_list.append(w)
                vars_list.append(b)

                w, b = mnn.linear(param_config["out"], param_config["in"], False,
                                  info_dict["meta"])

                vars_list.append(w)
                vars_list.append(b)


            elif info_dict["name"] == 'MultiHeadAttention':
                param_config = info_dict["config"]
                q_w, q_b, k_w, k_b, v_w, v_b, fc_w, fc_b = mnn.attention(
                    param_config["d_model"], param_config["num_heads"], info_dict["adaptation"],
                    info_dict["meta"])
                vars_list.append(q_w)
                vars_list.append(q_b)
                vars_list.append(k_w)
                vars_list.append(k_b)
                vars_list.append(v_w)
                vars_list.append(v_b)
                vars_list.append(fc_w)
                vars_list.append(fc_b)

            elif info_dict["name"] == 'layer_norm':
                w, b = mnn.layer_norm(info_dict["d_model"], info_dict["adaptation"],
                                      info_dict["meta"])
                self.d_model = info_dict["d_model"]
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] in ['PositionalEncoding', 'relu', 'flatten']:
                continue
            else:
                raise NotImplementedError
        return vars_list

    def forward(self, x, vars=None, config=None):
        x = x.float()
        if vars is None:
            vars = self.vars

        if config is None:
            config = self.config

        idx = 0
        y = None

        for layer_counter, info_dict in enumerate(config):
            name = info_dict["name"]

            if name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2

            elif name == 'doublelinear':
                w1, b1 = vars[idx], vars[idx + 1]
                x1 = F.linear(x, w1, b1)
                idx += 2
                w2, b2 = vars[idx], vars[idx + 1]
                x2 = F.linear(x, w2, b2)
                idx += 2
                if info_dict['output_num'] == 2:
                    x2 = torch.clamp(x2, min=-20, max=2)
                    return x1, x2
                else:
                    x = torch.cat((x1, x2), dim=0)

            elif name == 'relu':
                x = F.relu(x)

            elif name == 'PositionalEncoding':
                x = x + self.pe[:x.size(1), :]

            elif name == 'MultiHeadAttention':
                q, k, v = x, x, x
                bs = q.size(0)

                w, b = vars[idx], vars[idx + 1]
                q = F.linear(q, w, b).view(bs, -1, self.num_heads, self.d_k)
                idx += 2
                w, b = vars[idx], vars[idx + 1]
                k = F.linear(k, w, b).view(bs, -1, self.num_heads, self.d_k)
                idx += 2
                w, b = vars[idx], vars[idx + 1]
                v = F.linear(v, w, b).view(bs, -1, self.num_heads, self.d_k)
                idx += 2

                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
                scores = F.softmax(scores, dim=-1)

                y = torch.matmul(scores, v).transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)

                w, b = vars[idx], vars[idx + 1]
                y = F.linear(y, w, b)
                idx += 2

            elif name == 'layer_norm':
                if info_dict['flag'] == 1:
                    x = x + y
                    gamma, beta = vars[idx], vars[idx + 1]
                    x = F.layer_norm(x, (self.d_model,), gamma, beta)
                    idx += 2
                    y = x
                elif info_dict['flag'] == 2:
                    x = x + y
                    gamma, beta = vars[idx], vars[idx + 1]
                    x = F.layer_norm(x, (self.d_model,), gamma, beta)
                    idx += 2
                    x = x.mean(dim=1)

            else:
                raise NotImplementedError
        assert idx == len(vars)
        return x

    def utility_forward(self, x, vars=None, config=None):
        outputs = []
        x = x.float()
        if vars is None:
            vars = self.vars

        if config is None:
            config = self.config

        idx = 0
        y = None

        for layer_counter, info_dict in enumerate(config):
            name = info_dict["name"]

            if name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                outputs.append(x.view(-1, x.size(-1)))

            elif name == 'doublelinear':
                w1, b1 = vars[idx], vars[idx + 1]
                x1 = F.linear(x, w1, b1)
                idx += 2
                w2, b2 = vars[idx], vars[idx + 1]
                x2 = F.linear(x, w2, b2)
                idx += 2
                if info_dict['output_num'] == 2:
                    x2 = torch.clamp(x2, min=-20, max=2)
                    return x1, x2, outputs
                else:
                    x = torch.cat((x1, x2), dim=0)

            elif name == 'relu':
                x = F.relu(x)

            elif name == 'PositionalEncoding':
                x = x + self.pe[:x.size(1), :]

            elif name == 'MultiHeadAttention':
                q, k, v = x, x, x
                bs = q.size(0)

                w, b = vars[idx], vars[idx + 1]
                q = F.linear(q, w, b).view(bs, -1, self.num_heads, self.d_k)
                idx += 2
                w, b = vars[idx], vars[idx + 1]
                k = F.linear(k, w, b).view(bs, -1, self.num_heads, self.d_k)
                idx += 2
                w, b = vars[idx], vars[idx + 1]
                v = F.linear(v, w, b).view(bs, -1, self.num_heads, self.d_k)
                idx += 2

                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
                scores = F.softmax(scores, dim=-1)

                y = torch.matmul(scores, v).transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)

                w, b = vars[idx], vars[idx + 1]
                y = F.linear(y, w, b)
                idx += 2
                outputs.append(y.view(-1, y.size(-1)))

            elif name == 'layer_norm':
                if info_dict['flag'] == 1:
                    x = x + y
                    gamma, beta = vars[idx], vars[idx + 1]
                    x = F.layer_norm(x, (self.d_model,), gamma, beta)
                    idx += 2
                    y = x
                elif info_dict['flag'] == 2:
                    x = x + y
                    gamma, beta = vars[idx], vars[idx + 1]
                    x = F.layer_norm(x, (self.d_model,), gamma, beta)
                    idx += 2
                    x = x.mean(dim=1)

            else:
                raise NotImplementedError
        assert idx == len(vars)

    def get_adaptation_parameters(self, vars=None):
        if vars is None:
            vars = self.vars
        return list(filter(lambda x: x.adaptation, list(vars)))

    def get_no_adaptation_parameters(self, vars=None):
        if vars is None:
            vars = self.vars
        return list(filter(lambda x: hasattr(x, 'adaptation') and not x.adaptation, list(vars)))

    def get_forward_meta_parameters(self):
        return list(filter(lambda x: x.meta, list(self.vars)))

    def parameters(self):
        return self.vars