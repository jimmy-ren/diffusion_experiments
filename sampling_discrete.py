from main_network import *
import matplotlib.pyplot as plt

n_diff_steps = 100
Q = torch.tensor([0.99, 0.01, 0.01, 0.99])
Q = torch.reshape(Q, [2, 2])
FILE = './save/model_discrete_epoch_10.pth'

# device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = Unet(im_channels=2)
model.load_state_dict(torch.load(FILE))
model.to(device)

# inference
with torch.no_grad():
    prob = torch.ones([100, 1, 28, 28])
    prob = prob - 0.5
    inputs = prob_to_onehot(prob)
    inputs = inputs.to(device)
    '''
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(rand_samples[i][0], cmap='gray')
    '''
    for i in range(n_diff_steps):
        model_step = torch.full((1,), n_diff_steps - i).to(device)
        outputs = model(inputs, model_step)

        # convert the output to probability
        outputs = nn.Softmax(dim=1)(outputs)

        outputs = outputs.to('cpu')
        inputs = inputs.to('cpu')
        inputs = reverse_proc_sampling_discrete(inputs, outputs, Q, n_diff_steps - i - 1)
        inputs = inputs.to(device)

        '''
        for j in range(100):
            plt.subplot(10, 10, j+1)
            plt.imshow(inputs[j][0].detach().cpu(), cmap='gray')
            plt.grid(False)
            plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'./plots/discrete_diff_{i}.png')
        #plt.show()
        '''


    inputs = inputs.detach().cpu()
    outputs = outputs.detach().cpu()
    outputs = model_output_to_onehot(outputs)

    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.imshow(outputs[i][0], cmap='gray')
        plt.grid(False)
        plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()