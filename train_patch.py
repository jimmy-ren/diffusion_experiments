import sys
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import time
from main_vae_network import *
from main_network import *

#some config
num_epochs = 100
batch_size = 100
learning_rate = 0.0002
n_diff_steps = 100
# alpha_min 0.8 for 100 steps, 0.98 for 1000 steps
alpha_min = 0.8
alpha_max = 0.9999
alpha_schedule = torch.linspace(alpha_max, alpha_min, n_diff_steps)
alphas_cumprod = torch.cumprod(alpha_schedule, dim=0)

dataset = 'MNIST'
#dataset = 'CIFAR10'

label_type = 'image'
#label_type = 'noise'

print(f'total epochs: {num_epochs}, diff steps: {n_diff_steps}')
print(f'dataset: {dataset}, batch_size: {batch_size}, label type: {label_type}, learning rate: {learning_rate}')

if dataset == 'MNIST':
    t = transforms.Compose([
        #transforms.Pad(padding=2, fill=0, padding_mode='constant'),
        #transforms.CenterCrop((8,8)),
        #transforms.RandomCrop(size=(32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    # MNIST dataset, images are 1x28x28 pad to 1x32x32
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=t,
                                              download=True)

    # Specify the target class (e.g., '3')
    target_class = 3
    # Get the indices of samples belonging to the target class
    indices = (train_dataset.targets == target_class).nonzero().flatten()

    # Create a subset with only the target class
    single_class_dataset = torch.utils.data.Subset(train_dataset, indices)

    kernel_size = 7
    rows_batch, cols_batch, grid_h, grid_w, num_patches = get_patch_coordinates_batch(batch_size, 28, 7)
    rows_flat = rows_batch.reshape(-1, 1)
    cols_flat = cols_batch.reshape(-1, 1)
    rows_cols = torch.cat([rows_flat, cols_flat], dim=1)

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
#train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Create a DataLoader for the single-class dataset
train_loader = torch.utils.data.DataLoader(single_class_dataset, batch_size=batch_size, shuffle=True)

# device
#torch.cuda.empty_cache()
device = torch.device('cuda:1' if torch.cuda.is_available else 'cpu')
torch.cuda.set_device(device)
dn = torch.cuda.get_device_name(device)
print('using device:', dn)

if dataset == 'MNIST':
    #model = VAENet(im_channels=1, enable_attention=False)
    model = Unet(im_channels=1, attention_mode='OFF', down_sample=[False, False, False], p_emb_dim=64)
elif dataset == 'CIFAR10':
    model = VAENet(im_channels=3, enable_attention=False)
else:
    model = VAENet()

# model parameters count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

model = model.to(device)

# loss and optimizer
criterion = nn.MSELoss()        # denoising loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
loss_sum_in_step_loop = 0
loss_sum_in_epoch_loop = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    start_time = time.time()  # Record start time
    grad_norm = 0
    for i, (images, labels) in enumerate(train_loader):
        # size [batch_size, c, h, w]
        image_batch = images
        #tmp = images[0,0,:,:]
        #image_batch = image_batch[:,0:1,:,:]
        #images = images[:,0:1,:,:]

        # steps in [1, 100]
        current_batch_sz = image_batch.shape[0]
        random_step = torch.randint(low=1, high=n_diff_steps + 1, size=(current_batch_sz,))

        noisy_image_batch, ori_noise = noisify(image_batch, alphas_cumprod, random_step)

        '''
        for i in range(100):
            plt.subplot(10, 10, i+1)
            tmp = noisy_image_batch[i,:,:,:]
            tmp = torch.permute(tmp,(1,2,0))
            plt.axis('off')
            plt.imshow(tmp)
        plt.show()
        '''
        # unfolding the data
        unfold = torch.nn.Unfold(kernel_size=(kernel_size, kernel_size))
        input_unfold = unfold(noisy_image_batch)
        input_unfold = torch.permute(input_unfold, (0, 2, 1))
        input_unfold = input_unfold.reshape(-1, 1, kernel_size, kernel_size)

        dup_n = int(input_unfold.shape[0] / current_batch_sz)
        random_step_unfold = torch.repeat_interleave(random_step, repeats=dup_n)

        permutation = torch.randperm(input_unfold.shape[0])
        permutation = permutation[0:1000]
        input_unfold = input_unfold[permutation, :, :, :]
        random_step_unfold = random_step_unfold[permutation]

        input_unfold = input_unfold.to(device)
        random_step_unfold = random_step_unfold.to(device)

        rows_cols_perm = rows_cols[permutation, :]
        rows_cols_perm = rows_cols_perm.to(device)

        # forward
        outputs = model(input_unfold, random_step_unfold, rows_cols_perm)

        if label_type == 'image':
            gt = images
            FILE = f'./save/{dataset}/model_cont_patch_predimg_epoch_{epoch + 1}.pth'
        elif label_type == 'noise':
            gt = ori_noise
            FILE = f'./save/{dataset}/model_cont_patch_prednoise_epoch_{epoch + 1}.pth'
        else:
            # something is wrong
            sys.exit()

        gt_unfold = unfold(gt)
        gt_unfold = torch.permute(gt_unfold, (0, 2, 1))
        gt_unfold = gt_unfold.reshape(-1, 1, kernel_size, kernel_size)
        gt_unfold = gt_unfold[permutation, :, :, :]

        gt_unfold = gt_unfold.to(device)
        #start_pos = int((kernel_size - 1) / 2)
        #loss = criterion(outputs[:,:,start_pos:start_pos+1,start_pos:start_pos+1], gt_unfold[:,:,start_pos:start_pos+1,start_pos:start_pos+1])
        loss = criterion(outputs, gt_unfold)

        '''
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.imshow(noisy_image_batch[i][0], cmap='gray')
        '''

        # backward
        loss.backward()

        # monitor the gradients
        gn = get_gradient_norm(model)
        # get the recent maximum gn
        if gn > grad_norm:
            grad_norm = gn

        optimizer.step()
        optimizer.zero_grad()

        loss_sum_in_step_loop += loss.item()
        loss_sum_in_epoch_loop += loss.item()
        if (i + 1) % 100 == 0:
            loss_in_step_loop = loss_sum_in_step_loop / 100
            print(
                f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, avg loss={loss_in_step_loop:.4f}, Gradient norm: {grad_norm:.2f}')
            loss_sum_in_step_loop = 0
            grad_norm = 0

    if (epoch + 1) % 10 == 0:
        # save checkpoint every 100 epochs
        torch.save(model.state_dict(), FILE)

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    loss_in_epoch_loop = loss_sum_in_epoch_loop / (i + 1)
    print(f'Elapsed time for one epoch: {elapsed_time:.6f} seconds, epoch avg loss={loss_in_epoch_loop:.4f}')
    loss_sum_in_epoch_loop = 0



