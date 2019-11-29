import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def train(net, x, y, epochs=5, batch_size=50, lr=0.001):
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    loss_log = []
    for e in range(epochs):
        for i in range(0, x.shape[0], batch_size):
            x_mini = x[i : i + batch_size]
            y_mini = y[i : i + batch_size]

            optimizer.zero_grad()
            net_out = net(Variable(x_mini))

            loss = loss_func(net_out, Variable(y_mini))
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                loss_log.append(loss.item())

        print("Epoch: {} - Loss: {:.6f}".format(e + 1, loss.item()))
    return loss_log

