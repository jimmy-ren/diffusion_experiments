import sys
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import time
from main_network import*

#some config
num_epochs = 400
batch_size = 100
learning_rate = 0.0001
n_diff_steps = 100
alpha_min = 0.8
alpha_max = 0.9999
alpha_schedule = torch.linspace(alpha_max, alpha_min, n_diff_steps)
#dataset = 'MNIST'
dataset = 'CIFAR10'

label_type = 'image'
#label_type = 'noise'

if dataset == 'MNIST':
    t = transforms.Compose([
        transforms.Pad(padding=2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    # MNIST dataset, images are 1x28x28 pad to 1x32x32
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=t,
                                              download=True)
elif dataset == 'CIFAR10':
    t = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # CIFAR10 dataset, images are 3x32x32
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=t,
                                               download=True)
else:
    sys.exit('Invalid dataset')

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
if dataset == 'MNIST':
    model = Unet(im_channels=1, enable_attention=False)
elif dataset == 'CIFAR10':
    model = Unet(im_channels=3, enable_attention=True)
else:
    model = Unet()

# load weights
#FILE = f'./save/{dataset}/model_cont_predimg_epoch_200_v2_down_to_4x4_no_dropout.pth'
#model.load_state_dict(torch.load(FILE))
model = model.to(device)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
loss_sum_in_step_loop = 0
loss_sum_in_epoch_loop = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    start_time = time.time()  # Record start time
    for i, (images, labels) in enumerate(train_loader):
        # size [batch_size, c, h, w]
        image_batch = images
        #tmp = images[0,0,:,:]
        #image_batch = image_batch[:,0:1,:,:]
        #images = images[:,0:1,:,:]

        # steps in [1, 100]
        random_step = torch.randint(low=1, high=n_diff_steps + 1, size=(1,))

        noisy_image_batch, ori_noise = noisify(image_batch, alpha_schedule, random_step)

        '''
        for i in range(6):
            plt.subplot(2, 3, i+1)
            tmp = noisy_image_batch[i,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.imshow(tmp)
        plt.show()
        '''

        random_step = random_step.to(device)
        noisy_image_batch = noisy_image_batch.to(device)

        # forward
        outputs = model(noisy_image_batch, random_step)

        if label_type == 'image':
            images = images.to(device)
            loss = criterion(outputs, images)
            FILE = f'./save/{dataset}/model_cont_predimg_epoch_{epoch + 1}.pth'
        elif label_type == 'noise':
            ori_noise = ori_noise.to(device)
            loss = criterion(outputs, ori_noise)
            FILE = f'./save/{dataset}/model_cont_prednoise_epoch_{epoch + 1}.pth'
        else:
            # something is wrong
            sys.exit()

        '''
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.imshow(noisy_image_batch[i][0], cmap='gray')
        '''

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_sum_in_step_loop += loss.item()
        loss_sum_in_epoch_loop += loss.item()
        if (i + 1) % 100 == 0:
            loss_in_step_loop = loss_sum_in_step_loop / 100
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, avg loss={loss_in_step_loop:.4f}')
            loss_sum_in_step_loop = 0


    # save checkpoint for each epoch
    torch.save(model.state_dict(), FILE)
    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    loss_in_epoch_loop = loss_sum_in_epoch_loop / (i + 1)
    print(f'Elapsed time for one epoch: {elapsed_time:.6f} seconds, epoch avg loss={loss_in_epoch_loop:.4f}')
    loss_sum_in_epoch_loop = 0


