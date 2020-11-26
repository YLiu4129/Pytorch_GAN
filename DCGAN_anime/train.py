import time
from Setting import Train_arguments
from get_dataset import dataset
from model import Generator, Discriminator
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import show_demo, get_optimizer
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'now we are using {device}....')
    args = Train_arguments().parse()
    print(args)
    trans = transforms.Compose([
        transforms.Resize(args.reshape_size),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    )
    data_path = args.data_root
    data = dataset(data_path, transform=trans)
    trainSet = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    D = Discriminator(args.img_channel, args.out_channels).to(device)
    G = Generator(args.noise_channel, args.out_channels, args.img_channel).to(device)

    optim_D, optim_G = get_optimizer(args, D, G)

    G.train()
    D.train()

    loss_fn = nn.BCELoss()

    fixed_noise = torch.randn(args.batch_size, args.noise_channel, 1, 1).to(device)

    writer_real = SummaryWriter(f'test_real')
    writer_fake = SummaryWriter(f'test_fake')
    total_iters = 0  # the total number of training iterations
    print('Start Training...')

    for epoch in range(args.epochs):
        # timer for entire epoch
        epoch_start_time = time.time()

        for batch_idx, data in enumerate(trainSet):
            data = Variable(data).to(device)
            batch_size = data.shape[0]
            # train discriminator: max logD(x)+log(1-D(G(z)))
            # G(z) is the generated image
            D.zero_grad()
            # make D not super confident for its prediction by mutiplying 0.9 and max the logD(x) to 0.9
            label = (torch.ones(batch_size)).to(device)
            output = D(data).reshape(-1)
            loss_D_real = loss_fn(output, label)
            # compute the mean of confidence of D
            D_x = output.mean().item()

            # make fake image and put it into G
            noise = Variable(torch.randn(batch_size, args.noise_channel, 1, 1)).to(device)
            fake = G(noise)

            # make G not super confident for its prediction by mutiplying 0.1 and min the D(G(z)) to 0.1
            label = (torch.zeros(batch_size)).to(device)
            # detach(): we don't want to do back proprogation for the fake image
            output = D(fake.detach()).reshape(-1)
            loss_D_fake = loss_fn(output, label)

            loss_D = loss_D_real + loss_D_fake

            loss_D.backward()
            optim_D.step()

            # train generator : max(D(G(z))) because we want G to fool D, so set label(score) to 1

            G.zero_grad()
            label = torch.ones(batch_size).to(device)
            # we dont want to use detach() becasue we want to min the loss of G
            output = D(fake).reshape(-1)
            loss_G = loss_fn(output, label)
            loss_G.backward()
            optim_G.step()


            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{args.epochs}] Batch {batch_idx}/{len(trainSet)}\
                      Loss D : {loss_D:.4f}, Loss G : {loss_G:.4f}, D(x) : {D_x : .4f}')


                with torch.no_grad():
                    fake = G(fixed_noise)
                    img_grid_real = torchvision.utils.make_grid(data[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                    writer_real.add_image('real image', img_grid_real)
                    writer_fake.add_image('fake image', img_grid_fake)

        now = time.time()
        print(f'Time pass: {now - epoch_start_time}')