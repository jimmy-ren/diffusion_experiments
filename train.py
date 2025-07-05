import torchvision
import torchvision.transforms as transforms
import time
from main_network import*

num_epochs = 10
batch_size = 100
learning_rate = 0.001
alpha = 0.99
n_diff_steps = 100

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
alpha = torch.tensor(alpha, requires_grad=False)
for epoch in range(num_epochs):
    start_time = time.time()  # Record start time
    for i, (images, labels) in enumerate(train_loader):
        # size [batch_size, 1, 28, 28]
        image_batch = images
        random_step = torch.randint(low=1, high=n_diff_steps + 1, size=(1,))
        noisy_image_batch = noisify(image_batch, alpha, random_step)
        random_step = random_step.to(device)

        '''
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.imshow(noisy_image_batch[i][0], cmap='gray')
        '''
        # sys.exit()

        noisy_image_batch = noisy_image_batch.to(device)
        images = images.to(device)

        # forward
        outputs = model(noisy_image_batch, random_step)
        loss = criterion(outputs, images)

        # backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss={loss.item():.4f}')

    # save checkpoint for each epoch
    FILE = f'./save/model_cont_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), FILE)
    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Elapsed time for one epoch: {elapsed_time:.6f} seconds")

