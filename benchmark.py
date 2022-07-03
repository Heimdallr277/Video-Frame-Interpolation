import os
import time
import numpy as np
import torch
from models.VFI import VFINet

torch.backends.cudnn.benchmark = True
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

img1 = torch.randn(1, 3, 640, 448).cuda()
img2 = torch.randn(1, 3, 640, 448).cuda()


model = VFINet()
model = torch.nn.DataParallel(model).cuda().eval()
load_name = os.path.join('checkpoint', 'VFI_FFN', 'checkpoint.pth')
model.load_state_dict(torch.load(load_name)['state_dict'])



start = time.time()
for x in range(1000):
    output_t = model(img1,img2)    
end = time.time()
print('Time elapsed: {:.3f} ms'.format(end-start))

model = model.train()
print('Number of parameters: {:.2f} M'.format(count_parameters(model) / 1e6))