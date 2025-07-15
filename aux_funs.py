import torch

# imgs nchw
# alpha [0 1], time_step >= 0
def noisify(imgs, alpha, time_step):
    rand_n = torch.randn(imgs.shape)
    # make sure rand_n is stored on the same device as imgs
    rand_n = rand_n.to(imgs.device)

    alpha_bar = torch.pow(alpha, time_step)
    #alpha_bar = alpha_bar[:, None, None, None]
    out = imgs * torch.sqrt(alpha_bar) + rand_n * torch.sqrt(1 - alpha_bar)
    return out, rand_n

# model_outputs n2hw
def model_output_to_onehot(model_outputs):
    ret = torch.zeros(model_outputs.shape)
    condition = model_outputs[:,0,:,:] > model_outputs[:,1,:,:]
    ret[:,0,:,:] = torch.where(condition, 1, 0)
    condition = torch.logical_not(condition)
    ret[:,1,:,:] = torch.where(condition, 1, 0)
    return ret

# prob_imgs n2hw
def prob_to_onehot(prob_imgs):
    prob = prob_imgs[:,0:1,:,:]
    c1 = torch.bernoulli(prob)
    c2 = torch.ones(c1.shape)
    condition = c1 > 0
    c2 = torch.where(condition, 0, c2)
    ret = torch.cat([c1, c2], dim=1)
    return ret

# imgs nchw, one-hot encoding
# Q is a mxm state transition matrix
def noisify_discrete(imgs, Q, time_step, return_onehot=True):
    Q = torch.matrix_power(Q, time_step)
    Q = torch.reshape(Q, [Q.shape[0], Q.shape[1], 1, 1])
    imgs = imgs.permute(0, 3, 2, 1) # to nwhc
    Qr = Q.repeat([1, 1, imgs.shape[0], imgs.shape[1]]) # repeat batch_size, w times
    Qr = Qr.permute([2, 3, 0, 1])
    ret = torch.matmul(imgs, Qr)
    # back to nchw
    ret = ret.permute([0, 3, 2, 1])

    if return_onehot:
        # for forward process output should be noisy discrete images (one-hot encoding), not probabilities
        ret = prob_to_onehot(ret)
    return ret

# x_t, x_0, nchw
def reverse_proc_sampling(x_t, x_0, alpha, time_step, label_type='image'):
    # from time step: t
    # to time step: t-1
    from_ts = time_step + 1
    to_ts = time_step
    alpha_bar_from_ts = torch.pow(alpha, from_ts)
    alpha_bar_to_ts = torch.pow(alpha, to_ts)
    if label_type == 'image':
        mean = (torch.sqrt(alpha) * (1 - alpha_bar_to_ts) * x_t + \
                torch.sqrt(alpha_bar_to_ts) * (1 - alpha) * x_0) / (1 - alpha_bar_from_ts)
    elif label_type == 'noise':
        mean = (1 / torch.sqrt(alpha)) * x_t - (1 - alpha) / (torch.sqrt(1 - alpha_bar_from_ts) * (torch.sqrt(alpha))) * x_0
    else:
        # something is wrong
        mean = x_0
    variance = (1 - alpha) * (1 - alpha_bar_to_ts) / (1 - alpha_bar_from_ts)
    rand_n = torch.randn(x_0.shape)
    # make sure rand_n is stored on the same device as x_0
    rand_n = rand_n.to(x_0.device)

    ret = mean + torch.sqrt(variance) * rand_n
    return ret

# x_t, x_0 nchw, one-hot encoding
# Q is a mxm state transition matrix
def reverse_proc_sampling_discrete(x_t, x_0, Q, time_step):
    # transpose Q
    Qt = torch.transpose(Q, 0, 1)
    p1 = noisify_discrete(x_t, Qt, 1, return_onehot=False)
    p2 = noisify_discrete(x_0, Q, time_step, return_onehot=False)
    ret = torch.mul(p1, p2)
    # normalize
    sum = torch.sum(ret, dim=1, keepdim=True)
    ret = ret / sum
    ret = prob_to_onehot(ret)
    return ret

class BinaryOneHotTransform():
    def __call__(self, sample):
        i = torch.ones([1, sample.shape[1], sample.shape[2]])
        j = torch.zeros([1, sample.shape[1], sample.shape[2]])
        condition = sample == 0
        # Set elements satisfying the condition to a new value (e.g., 0)
        i = torch.where(condition, 0, i)
        j = torch.where(condition, 1, j)
        ret = torch.cat([i, j], dim=0)
        return ret

def get_time_embedding(time_steps: torch.Tensor, t_emb_dim: int) -> torch.Tensor:
    """
    Transform a scalar time-step into a vector representation of size t_emb_dim.

    :param time_steps: 1D tensor of size -> (Batch,)
    :param t_emb_dim: Embedding Dimension -> for ex: 128 (scalar value)

    :return tensor of size -> (B, t_emb_dim)
    """
    assert t_emb_dim % 2 == 0, "time embedding must be divisible by 2."

    factor = 2 * torch.arange(start=0,
                              end=t_emb_dim // 2,
                              dtype=torch.float32,
                              device=time_steps.device
                              ) / (t_emb_dim)

    factor = 10000 ** factor

    t_emb = time_steps[:, None]  # B -> (B, 1)
    t_emb = t_emb / factor  # (B, 1) -> (B, t_emb_dim//2)
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=1)  # (B , t_emb_dim)

    return t_emb