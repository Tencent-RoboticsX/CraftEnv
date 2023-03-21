REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .mlp_agent import MLPAgent
REGISTRY["mlp"] = MLPAgent