import torch
from generator import Generator
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from tools import *

DOMAIN_A = 'Blond_Hair'
DOMAIN_B = 'Black_Hair'

device= 'gpu' if torch.cuda.is_available() else 'cpu'
path = './generatorB_model.mdl'

model = torch.load(path, map_location=device)

if isinstance(model, torch.nn.DataParallel):
    model = model.module
else:
    model = model


test_A, test_B = getCeleb('Male', -1, DOMAIN_A, DOMAIN_B, True)
A = torch.FloatTensor(read_images( test_A[14:15], 64))



img_list = []

fake = model(A).detach().cpu()
img_list.append(vutils.make_grid(fake, padding=10, normalize=True))

plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(np.transpose(read_images(test_A[14:15])[0], (1,2,0)))
plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
