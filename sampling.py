from main_network import *
import matplotlib.pyplot as plt

n_diff_steps = 100
alpha = 0.99

label_type = 'image'
FILE = './save/model_cont_predimg_epoch_10.pth'

#label_type = 'noise'
#FILE = './save/model_cont_prednoise_epoch_30.pth'

# device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = Unet()
model.load_state_dict(torch.load(FILE))
model.to(device)

# inference
alpha = torch.tensor(alpha).to(device)
with torch.no_grad():
    inputs = torch.randn([100, 1, 28, 28])
    inputs = inputs.to(device)

    for i in range(n_diff_steps):
        #model_step = torch.full((1,), n_diff_steps - i).to(device)
        model_step = torch.tensor([n_diff_steps - i]).to(device)
        outputs = model(inputs, model_step)

        next_time_step = torch.tensor([n_diff_steps - i - 1]).to(device)
        #inputs, _ = noisify(outputs, alpha, next_time_step)
        inputs = reverse_proc_sampling(inputs, outputs, alpha, next_time_step, label_type)
        #inputs = 1 * inputs1 + 0 * inputs2

        '''
        for j in range(100):
            plt.subplot(10, 10, j + 1)
            plt.imshow(inputs[j][0].detach().cpu(), cmap='gray')
            plt.grid(False)
            plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'./plots/continuous_diff_{i}.png')
        plt.show()
        '''

    inputs = inputs.detach().cpu()
    outputs = outputs.detach().cpu()

    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.imshow(inputs[i][0], cmap='gray')
        plt.grid(False)
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()