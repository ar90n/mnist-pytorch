import os
from pathlib import Path

from mnist_pytorch import io
from mnist_pytorch.network import Network, extract_result
from mnist_pytorch.train import train

root = Path("../input/digit-recognizer")
train_x, train_y = io.load_train_data(root)

test = io.load_test_data(root)

net = Network()
losses = train(net, train_x, train_y)

net_out = net(test)
result = extract_result(net_out)
io.save_result(result)
