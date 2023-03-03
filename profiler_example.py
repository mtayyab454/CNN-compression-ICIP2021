import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

model = models.resnet18()
input_size = (1, 3, 224, 224) # batch_size, channels, height, width
input_tensor = torch.rand(input_size)

def measure_flops(model):
    input = torch.randn(1, 3, 224, 224).to(list(model.parameters())[0].device)
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_flops=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                model(input)
    org_flops = 0
    for ka in prof.key_averages():
        org_flops += ka.flops
        print(org_flops)

    return org_flops

measure_flops(model)