import torch
import torch.nn as nn

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Define a forward hook to calculate FLOPS
class FLOPSCounter:
    def __init__(self):
        self.total_flops = 0

    def hook(self, module, input, output):
        input_size = input[0].size()
        output_size = output.size()

        # Calculate the number of FLOPS for a fully connected layer
        flops = input_size[0] * input_size[1] * output_size[1]

        # Add the FLOPS to the total count
        self.total_flops += flops

# Create an instance of the neural network
net = Net()

# Create an instance of the FLOPS counter
flops_counter = FLOPSCounter()

# Register the FLOPS counter to the FC layers
handle1 = net.fc1.register_forward_hook(flops_counter.hook)
handle2 = net.fc2.register_forward_hook(flops_counter.hook)

# Define some input data
x = torch.randn(1, 10)

# Forward pass through the neural network
y = net(x)

# Print the total FLOPS
print("Total FLOPS:", flops_counter.total_flops)

# Deregister the forward hooks
handle1.remove()
handle2.remove()
