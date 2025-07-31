import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import time
from main_network import*

num_epochs = 100
batch_size = 100
n_diff_steps = 100
learning_rate = 0.0006
alpha_min = 0.95
alpha_max = 0.99999
alpha_schedule = torch.linspace(alpha_max, alpha_min, n_diff_steps)
# construct the transition matrices
Q_set = torch.zeros([n_diff_steps, 2, 2])
Q_set[:, 0, 0] = alpha_schedule
Q_set[:, 1, 1] = alpha_schedule
Q_set[:, 0, 1] = 1 - alpha_schedule
Q_set[:, 1, 0] = 1 - alpha_schedule
Q_bar_set = cumulative_matrix_mul(Q_set)

composed = torchvision.transforms.Compose([transforms.ToTensor(), BinaryOneHotTransform()])

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=composed,
                                          download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),
                                         download=True)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = Unet(im_channels=2)   # 2 dim one-hot encoding for discrete MNIST
model = model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    start_time = time.time()  # Record start time
    for i, (images, labels) in enumerate(train_loader):
        # size [batch_size, 1, 28, 28]
        image_batch = images

        '''
        # test visualizing noisy data
        nstep = 5
        noisy_image_batch = noisify_discrete(image_batch, Q, nstep)
        for i in range(18):
            plt.subplot(6, 3, i+1)
            if(i < 6):
                plt.imshow(image_batch[i][0], cmap='gray')
            elif(i >= 6 and i < 12):
                plt.imshow(noisy_image_batch[i-6][0], cmap='gray')
            else:
                plt.imshow(noisy_image_batch[i-12][1], cmap='gray')

        plt.show()
        '''

        random_step = torch.randint(low=1, high=n_diff_steps + 1, size=(1,))
        noisy_image_batch = noisify_discrete(image_batch, Q_bar_set, random_step.item()-1)
        random_step = random_step.to(device)

        noisy_image_batch = noisy_image_batch.to(device)
        train_labels = image_batch[:, 1, :, :].long()
        train_labels = train_labels.to(device)

        # forward
        outputs = model(noisy_image_batch, random_step)

        loss = criterion(outputs, train_labels)

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss={loss.item():.4f}')

    # save checkpoint for each epoch
    FILE = f'./save/model_discrete_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), FILE)
    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time for one epoch: {elapsed_time:.6f} seconds")

