from torch import nn


def get_loss():
    return nn.CrossEntropyLoss()
