import numpy as np


class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset, input_dimension=6, output_dimension=6, width=300, d_ff=128):

        if dataset == "transformer":
            hidden_size = width
            return [
                {"name": 'linear', "adaptation": False, "meta": True,
                 "config": {"out": hidden_size, "in": input_dimension}},

                {"name": 'PositionalEncoding', "adaptation": False, "meta": True, "d_model": hidden_size},

                {"name": 'MultiHeadAttention', "adaptation": False, "meta": True,
                 "config": {"d_model": hidden_size, "num_heads": 4}},

                {"name": 'layer_norm', "adaptation": False, "meta": True, "flag": 1,
                 "d_model": hidden_size},

                {"name": 'linear', "adaptation": False, "meta": True,
                 "config": {"out": d_ff, "in": hidden_size}},

                {"name": 'relu'},

                {"name": 'linear', "adaptation": False, "meta": True,
                 "config": {"out": hidden_size, "in": d_ff}},

                {"name": 'layer_norm', "adaptation": False, "meta": True, "flag": 2,
                 "d_model": hidden_size},

                {"name": 'doublelinear', "adaptation": True, "meta": True, "output_num": 2,
                 "config": {"out": output_dimension, "in": hidden_size}},
            ]

        else:
            raise NotImplementedError