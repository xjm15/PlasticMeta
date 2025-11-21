import argparse
import sys


class Parser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PlasticMeta for Autonomous Driving')
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument('--gpus', type=int, default=1)
        self.parser.add_argument('--seed', nargs='+', default=[42])
        self.parser.add_argument('--rank', type=int, default=0)
        self.parser.add_argument('--name', type=str, default="PlasticMeta_AD")

        self.parser.add_argument('--meta_lr', type=float, default=1e-3)
        self.parser.add_argument('--update_lr', type=float, default=1e-2)
        self.parser.add_argument('--update_step', type=int, default=10)
        self.parser.add_argument('--step', type=int, default=500000)

        self.parser.add_argument('--input_dimension', type=int, default=259)
        self.parser.add_argument('--output_dimension', type=int, default=2)
        self.parser.add_argument('--width', type=int, default=128)
        self.parser.add_argument('--d_ff', type=int, default=512)

        self.parser.add_argument('--utility_management', dest='utility_management', action='store_true',
                                 help='Enable Neuron Utility Management (PlasticMeta++)')
        self.parser.add_argument('--no_utility_management', dest='utility_management', action='store_false')
        self.parser.set_defaults(utility_management=True)

        self.parser.add_argument('--plasticity_lr', type=float, default=1e-3)
        self.parser.add_argument('--replacement_rate', type=float, default=1e-5)
        self.parser.add_argument('--decay_rate', type=float, default=0.99)
        self.parser.add_argument('--maturity_threshold', type=int, default=20)

        self.parser.add_argument('--no_sigmoid', dest='no_sigmoid', action='store_true')
        self.parser.add_argument('--no_save', dest='no_save', action='store_true')
        self.parser.add_argument('--model_path', type=str, default=None)

    def parse_known_args(self):
        return self.parser.parse_known_args()