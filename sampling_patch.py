import sys
import torch
from main_network import *
from naive_kpn import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

batch_size = 9
n_diff_steps = 100
# alpha_min 0.8 for 100 steps, 0.98 for 1000 steps
alpha_min = 0.8
alpha_max = 0.9999
alpha_schedule = torch.linspace(alpha_max, alpha_min, n_diff_steps)

dataset = 'MNIST'
#dataset = 'CIFAR10'

label_type = 'image'
FILE = f'./save/{dataset}/model_cont_patch_predimg_epoch_500.pth'

#label_type = 'noise'
#FILE = f'./save/{dataset}/model_cont_prednoise_1000steps_epoch_2200_v10_2.pth'

# device
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
torch.cuda.set_device(device)
dn = torch.cuda.get_device_name(device)
print('using device:', dn)

if dataset == 'MNIST':
    kernel_size = 7
    #model = Unet(im_channels=3, out_channels=1, attention_mode='OFF', down_sample=[False, False, False], p_emb_dim=64)
    model = NaiveKPN(im_channels=3, out_channels=1, input_size=kernel_size, kernel_size=kernel_size, p_emb_dim=64,
                     enable_dropout=True)
    rows_batch, cols_batch, grid_h, grid_w, num_patches = get_patch_coordinates_batch(batch_size, 28+kernel_size-1, 7)
    rows_flat = rows_batch.reshape(-1, 1)
    cols_flat = cols_batch.reshape(-1, 1)
    rows_cols = torch.cat([rows_flat, cols_flat], dim=1)

    # for oracle test
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])
    # MNIST dataset, images are 1x28x28 pad to 1x32x32
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=t,
                                               download=True)

    # Specify the target class (e.g., '3')
    target_class = 5
    # Get the indices of samples belonging to the target class
    indices = (train_dataset.targets == target_class).nonzero().flatten()
    # Create a subset with only the target class
    single_class_dataset = torch.utils.data.Subset(train_dataset, indices)
    train_loader = torch.utils.data.DataLoader(single_class_dataset, batch_size=1, shuffle=True)
    for i, (images, labels) in enumerate(train_loader):
        # size [batch_size, c, h, w]
        oracle_img = images
        break
    oracle_img = F.interpolate(oracle_img, size=kernel_size, mode='area')

    plt.subplot(1, 2, 1)
    plt.imshow(torch.permute(images[0,:,:,:], (1,2,0)), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(torch.permute(oracle_img[0,:,:,:], (1,2,0)), cmap='gray')
    plt.show()

elif dataset == 'CIFAR10':
    model = Unet(im_channels=3, attention_mode='ON')
else:
    model = Unet()
model.load_state_dict(torch.load(FILE, map_location=torch.device('cpu')))

# model parameters count
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")

model.eval()
model.to(device)
rows_cols = rows_cols.to(device)

# inference
alpha_schedule = alpha_schedule.to(device)
global_context_x0 = None # dummy
with torch.no_grad():
    for loop in range(1):
        if dataset == 'MNIST':
            inputs = torch.randn([batch_size, 1, 28+kernel_size-1, 28+kernel_size-1])
        elif dataset == 'CIFAR10':
            inputs = torch.randn([batch_size, 3, 32, 32])
        else:
            sys.exit()
        inputs = inputs.to(device)

        for i in range(n_diff_steps):
            #model_step = torch.full((batch_size,), n_diff_steps - i).to(device)
            #model_step = torch.tensor([n_diff_steps - i]).to(device)
            start_pos = int((kernel_size - 1) / 2)
            # unfolding the data
            unfold = torch.nn.Unfold(kernel_size=(kernel_size, kernel_size))
            input_unfold = unfold(inputs)
            input_unfold = torch.permute(input_unfold, (0, 2, 1))
            input_unfold = input_unfold.reshape(-1, 1, kernel_size, kernel_size)

            model_step = torch.full((input_unfold.shape[0],), n_diff_steps - i).to(device)

            dup_n = int(input_unfold.shape[0] / batch_size)
            in_for_global_n = inputs[:,:,start_pos:28+start_pos,start_pos:28+start_pos]
            global_context_n = F.interpolate(in_for_global_n, size=kernel_size, mode='area')
            global_context_n = torch.repeat_interleave(global_context_n, repeats=dup_n, dim=0)
            global_context_n = global_context_n.to(device)

            global_context_x0 = oracle_img
            global_context_x0 = torch.repeat_interleave(global_context_x0, repeats=input_unfold.shape[0], dim=0)
            global_context_x0 = global_context_x0.to(device)
            combined_input = torch.cat([input_unfold, global_context_n, global_context_x0], dim=1)
            #if i == 0:
            #    combined_input = torch.cat([input_unfold, global_context_n, global_context_n], dim=1)
            #else:
            #    combined_input = torch.cat([input_unfold, global_context_n, global_context_x0], dim=1)

            #outputs = model(combined_input, model_step, rows_cols)
            # KPN
            _, outputs = model(combined_input, model_step, rows_cols)
            #in_for_global_x0 = outputs[:, :, start_pos:start_pos + 1, start_pos:start_pos + 1]
            in_for_global_x0 = outputs
            in_for_global_x0 = in_for_global_x0.reshape(batch_size, 1, 28, 28)
            global_context_x0 = F.interpolate(in_for_global_x0, size=kernel_size, mode='area')
            global_context_x0 = torch.repeat_interleave(global_context_x0, repeats=dup_n, dim=0)

            next_time_step = torch.tensor([n_diff_steps - i - 1]).to(device)
            #inputs, _ = noisify(outputs, alpha_schedule, next_time_step)
            #inputs = reverse_proc_sampling(input_unfold, outputs, alpha_schedule, next_time_step, label_type)
            inputs = reverse_proc_sampling(in_for_global_n, in_for_global_x0, alpha_schedule, next_time_step, label_type)

            # shape back to the original input size
            #inputs = inputs[:,:,start_pos:start_pos+1,start_pos:start_pos+1]
            #inputs = inputs.reshape(batch_size, 1, 28, 28)

            if label_type == 'noise' and i >= n_diff_steps - 1:
                # get x0 from the predicted noise
                tmp = torch.prod(alpha_schedule[0:n_diff_steps - i - 1])
                alpha_t_bar = torch.prod(alpha_schedule[0:n_diff_steps - i])
                outputs = (inputs - torch.sqrt(1 - alpha_t_bar) * outputs) / torch.sqrt(alpha_t_bar)

            print(f'{i+1:02d}', end=" ")
            if (i+1) % 25 == 0:
                print()


            for j in range(9):
                plt.subplot(3, 3, j + 1)
                if dataset == 'MNIST':
                    plt.imshow(inputs[j][0].detach().cpu(), cmap='gray')
                else:
                    im_c = inputs[j, :, :, :].detach().cpu()
                    im_c = torch.permute(im_c, (1, 2, 0))
                    im_c = im_c * 0.5 + 0.5
                    im_c[im_c < 0] = 0.0
                    im_c[im_c > 1] = 1.0
                    plt.imshow(im_c)
                plt.grid(False)
                plt.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(f'./plots/continuous_patch_diff_{i}.png')
            #plt.show()


            pad_size = int((kernel_size - 1) / 2)
            inputs = F.pad(inputs, pad=(pad_size, pad_size, pad_size, pad_size), mode='replicate')


        #inputs = inputs.detach().cpu()
        #outputs = outputs[:, :, start_pos:start_pos + 1, start_pos:start_pos + 1]
        outputs = outputs.reshape(batch_size, 1, 28, 28)
        outputs = outputs.detach().cpu()
        #tmp = outputs[0,0,:,:]
        outputs = outputs * 0.5 + 0.5

        max = torch.max(outputs)
        min = torch.min(outputs)
        outputs[outputs < 0] = 0.0
        outputs[outputs > 1] = 1.0

        # save generated images
        #for i in range(batch_size):
        #    file_id = batch_size * loop + i + 1
        #    print(f'saving file {file_id}')
        #    save_image(outputs[i, :, :, :], f'./data/cifar10/generated_1000steps_prednoise_2200epoch_v10_2b/{file_id}.png')


        for i in range(9):
            plt.subplot(3, 3, i + 1)
            if dataset == 'MNIST':
                plt.imshow(outputs[i][0], cmap='gray')
            else:
                im_c = outputs[i, :, :, :]
                im_c = torch.permute(im_c, (1, 2, 0))
                plt.imshow(im_c)
                #plt.imshow(outputs[i][0], cmap='gray')
            plt.grid(False)
            plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


