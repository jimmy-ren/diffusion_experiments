import sys
import torch
from main_network import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image

batch_size = 2000
n_diff_steps = 1000
# alpha_min 0.8 for 100 steps, 0.98 for 1000 steps
alpha_min = 0.98
alpha_max = 0.9999
alpha_schedule = torch.linspace(alpha_max, alpha_min, n_diff_steps)

#dataset = 'MNIST'
dataset = 'CIFAR10'

#label_type = 'image'
#FILE = f'./save/{dataset}/model_cont_predimg_1000steps_epoch_1600_v2.pth'

label_type = 'noise'
FILE = f'./save/{dataset}/model_cont_prednoise_1000steps_epoch_1000_v12.pth'

# device
device = torch.device('cuda:1' if torch.cuda.is_available else 'cpu')
torch.cuda.set_device(device)
dn = torch.cuda.get_device_name(device)
print('using device:', dn)

if dataset == 'MNIST':
    model = Unet(im_channels=1, attention_mode='OFF')
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

# inference
alpha_schedule = alpha_schedule.to(device)
with torch.no_grad():
    for loop in range(25):
        if dataset == 'MNIST':
            inputs = torch.randn([batch_size, 1, 32, 32])
        elif dataset == 'CIFAR10':
            inputs = torch.randn([batch_size, 3, 32, 32])
        else:
            sys.exit()
        inputs = inputs.to(device)

        for i in range(n_diff_steps):
            model_step = torch.full((batch_size,), n_diff_steps - i).to(device)
            #model_step = torch.tensor([n_diff_steps - i]).to(device)
            outputs = model(inputs, model_step)

            next_time_step = torch.tensor([n_diff_steps - i - 1]).to(device)
            #inputs, _ = noisify(outputs, alpha_schedule, next_time_step)
            inputs = reverse_proc_sampling(inputs, outputs, alpha_schedule, next_time_step, label_type)
            #inputs = 1 * inputs1 + 0 * inputs2

            if label_type == 'noise' and i >= n_diff_steps - 1:
                # get x0 from the predicted noise
                tmp = torch.prod(alpha_schedule[0:n_diff_steps - i - 1])
                alpha_t_bar = torch.prod(alpha_schedule[0:n_diff_steps - i])
                outputs = (inputs - torch.sqrt(1 - alpha_t_bar) * outputs) / torch.sqrt(alpha_t_bar)

            print(f'{i+1:02d}', end=" ")
            if (i+1) % 25 == 0:
                print()
            '''
            for j in range(100):
                plt.subplot(10, 10, j + 1)
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
            plt.savefig(f'./plots/continuous_diff_{i}.png')
            #plt.show()
            '''

        #inputs = inputs.detach().cpu()
        outputs = outputs.detach().cpu()
        #tmp = outputs[0,0,:,:]
        outputs = outputs * 0.5 + 0.5

        max = torch.max(outputs)
        min = torch.min(outputs)
        outputs[outputs < 0] = 0.0
        outputs[outputs > 1] = 1.0

        # save generated images
        for i in range(batch_size):
            file_id = batch_size * loop + i + 1
            print(f'saving file {file_id}')
            save_image(outputs[i, :, :, :], f'./data/cifar10/generated_1000steps_prednoise_1000epoch_v12/{file_id}.png')

        '''
        for i in range(100):
            plt.subplot(10, 10, i + 1)
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
        '''

