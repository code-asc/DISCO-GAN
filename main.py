import os
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from generator import Generator
from discriminator import Discriminator
from progressbar import ETA, Bar, Percentage, ProgressBar

from tools import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_gan_loss(dis_fake, criterion):
    global device
    labels_gen = torch.ones([dis_fake.size()[0], 1]).to(device)
    gen_loss = criterion( dis_fake, labels_gen)
    return gen_loss

def get_dis_loss(dis_real, dis_fake, criterion):
    global device
    labels_dis_real = torch.ones( [dis_real.size()[0], 1]).to(device)
    labels_dis_fake = torch.zeros([dis_fake.size()[0], 1]).to(device)
    dis_loss = criterion( dis_real, labels_dis_real ) + criterion( dis_fake, labels_dis_fake)
    return dis_loss


EPOCH = 50
BATCH_SIZE = 64
DOMAIN_A = 'Blond_Hair'
DOMAIN_B = 'Black_Hair'
IMAGE_SIZE = 64
LEARNING_RATE = 0.0002
UPDATE_INTERVAL = 3
LOG_INTERVAL = 50

data_A, data_B = getCeleb('Male', -1, DOMAIN_A, DOMAIN_B)
test_A, test_B = getCeleb('Male', -1, DOMAIN_A, DOMAIN_B, True)

generator_A = Generator()
generator_B = Generator()
discriminator_A = Discriminator()
discriminator_B = Discriminator()

generator_A = generator_A.to(device)
generator_B = generator_B.to(device)
discriminator_A = discriminator_A.to(device)
discriminator_B = discriminator_B.to(device)

if device == 'cuda':
    generator_A= torch.nn.DataParallel(generator_A)
    generator_B = torch.nn.DataParallel(generator_B)
    discriminator_A = torch.nn.DataParallel(discriminator_A)
    discriminator_B = torch.nn.DataParallel(discriminator_B)


chained_gen_params = chain(generator_A.parameters(), generator_B.parameters())
chained_dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())

optim_gen = torch.optim.Adam(chained_gen_params, lr=LEARNING_RATE, betas=(0.5,0.999), weight_decay=0.00001)
optim_dis = torch.optim.Adam(chained_dis_params, lr=LEARNING_RATE, betas=(0.5,0.999), weight_decay=0.00001)

data_size = min(len(data_A), len(data_B))
n_batches = ( data_size // BATCH_SIZE)

recon_criterion = nn.MSELoss()
gan_criterion = nn.BCELoss()
feat_criterion = nn.HingeEmbeddingLoss()
iters = 0

for epoch in range(EPOCH):
    ############################# DATA SHUFFLE START ################################
    data_A_idx = [i for i in range(len(data_A))]
    data_B_idx = [i for i in range(len(data_B))]
    np.random.shuffle(data_A_idx)
    np.random.shuffle(data_B_idx)

    data_A = np.array(data_A)[np.array(data_A_idx)]
    data_B = np.array(data_B)[np.array(data_B_idx)]
    ############################# DATA SHUFFLE STOP ################################
    widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval=n_batches, widgets=widgets)
    pbar.start()

    for i in range(n_batches):
         pbar.update(i)
         generator_A.zero_grad()
         generator_B.zero_grad()
         discriminator_A.zero_grad()
         discriminator_B.zero_grad()

         A_subset_image = data_A[ i * BATCH_SIZE: (i+1) * BATCH_SIZE ]
         B_subset_image = data_B[ i * BATCH_SIZE: (i+1) * BATCH_SIZE ]

         A = torch.FloatTensor(read_images( A_subset_image, IMAGE_SIZE))
         B = torch.FloatTensor(read_images( B_subset_image, IMAGE_SIZE))

         AB = generator_B(A) # Domain A to Domain B
         ABA = generator_A(AB) # Reconstruct domain A again

         BA = generator_A(B) # Domain B to Domain A
         BAB = generator_B(BA) # Reconstruct domain B again

         recon_loss_A = recon_criterion(ABA, A) # Calculate reconstruct of domain A
         recon_loss_B = recon_criterion(BAB, B) # Calculate reconstruct of domain B

         ############################# L(D_A) ################################
         A_dis_real = discriminator_A(A)
         A_dis_fake = discriminator_A(BA)
         ############################# END ################################

         ############################# L(D_B) ################################
         B_dis_real = discriminator_B(B)
         B_dis_fake = discriminator_B(AB)
         ############################# END ################################

         dis_loss_A = get_dis_loss(A_dis_real, A_dis_fake, gan_criterion)
         dis_loss_B = get_dis_loss(B_dis_real, B_dis_fake, gan_criterion)

         gen_loss_A = get_gan_loss(A_dis_fake, gan_criterion)
         gen_loss_B = get_gan_loss(B_dis_fake, gan_criterion)


         gen_loss = gen_loss_A + gen_loss_B + recon_loss_A + recon_loss_B
         dis_loss = dis_loss_A + dis_loss_B

         if iters % UPDATE_INTERVAL == 0:
            dis_loss.backward()
            optim_dis.step()

         else:
             gen_loss.backward()
             optim_gen.step()

         print('Epoch [{:5d}/{:5d}] | d_total_loss: {:6.4f} | g_loss: {:6.4f}'.format(epoch+1, EPOCH, dis_loss.item(), gen_loss.item()))
         iters+=1

torch.save(generator_A.state_dict(), './generatorA_state.mdl')
print('Model A state saved....')
torch.save(generator_A, './generatorA_model.mdl')
print('Model A saved....')

torch.save(generator_B.state_dict(), './generatorB_state.mdl')
print('Model B state saved....')
torch.save(generator_B, './generatorB_model.mdl')
print('Model B saved....')
