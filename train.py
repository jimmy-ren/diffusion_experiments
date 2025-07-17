import sys

import torchvision
import torchvision.transforms as transforms
import time
from main_network import*

#some config
num_epochs = 200
batch_size = 100
learning_rate = 0.0006
n_diff_steps = 100
alpha_min = 0.8
alpha_max = 0.9999
alpha_schedule = torch.linspace(alpha_max, alpha_min, n_diff_steps)
#alpha = 0.99


label_type = 'image'
#label_type = 'noise'

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),
                                          download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),
                                         download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = Unet()
model = model.to(device)

# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    start_time = time.time()  # Record start time
    for i, (images, labels) in enumerate(train_loader):
        # size [batch_size, 1, 28, 28]
        image_batch = images
        # steps in [1, 100]
        random_step = torch.randint(low=1, high=n_diff_steps + 1, size=(1,))

        noisy_image_batch, ori_noise = noisify(image_batch, alpha_schedule, random_step)
        random_step = random_step.to(device)
        noisy_image_batch = noisy_image_batch.to(device)

        # forward
        outputs = model(noisy_image_batch, random_step)

        if label_type == 'image':
            images = images.to(device)
            loss = criterion(outputs, images)
            FILE = f'./save/model_cont_predimg_epoch_{epoch + 1}.pth'
        elif label_type == 'noise':
            ori_noise = ori_noise.to(device)
            loss = criterion(outputs, ori_noise)
            FILE = f'./save/model_cont_prednoise_epoch_{epoch + 1}.pth'
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

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss={loss.item():.4f}')

    # save checkpoint for each epoch
    torch.save(model.state_dict(), FILE)
    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time for one epoch: {elapsed_time:.6f} seconds")

