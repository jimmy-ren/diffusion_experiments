from aux_modules import *

class NaiveKPN(nn.Module):

    def __init__(self,
                 im_channels: int = 1,  # RGB
                 down_ch: list = [32, 64, 128, 256],
                 mid_ch: list = [256, 256, 256],
                 up_ch: list[int] = [256, 128, 64, 16],
                 down_sample: list[bool] = [False, False, False],
                 t_emb_dim: int = 128,
                 num_downc_layers: int = 2,
                 num_midc_layers: int = 2,
                 num_upc_layers: int = 2,
                 enable_attention: bool = False,
                 input_size = 7,
                 kernel_size = 7
                 ):
        super(NaiveKPN, self).__init__()

        self.im_channels = im_channels
        self.down_ch = down_ch
        self.mid_ch = mid_ch
        self.up_ch = up_ch
        self.t_emb_dim = t_emb_dim
        self.down_sample = down_sample
        self.num_downc_layers = num_downc_layers
        self.num_midc_layers = num_midc_layers
        self.num_upc_layers = num_upc_layers
        self.enable_attention = enable_attention
        self.input_size = input_size
        self.kernel_size = kernel_size

        self.up_sample = list(reversed(self.down_sample))

        # Initial Convolution
        self.cv1 = nn.Conv2d(self.im_channels, self.down_ch[0], kernel_size=3, padding=1)

        # Initial Time Embedding Projection
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )

        # DownC Blocks
        self.downs = nn.ModuleList([
            DownC(
                self.down_ch[i],
                self.down_ch[i + 1],
                self.t_emb_dim,
                self.num_downc_layers,
                self.down_sample[i],
                self.enable_attention,
                dropout=False
            ) for i in range(len(self.down_ch) - 1)
        ])

        # MidC Block
        self.mids = nn.ModuleList([
            MidC(
                self.mid_ch[i],
                self.mid_ch[i+1],
                self.t_emb_dim,
                self.num_midc_layers,
                self.enable_attention,
                dropout=False
            ) for i in range(len(self.mid_ch) - 1)
        ])

        # UpC Block
        self.ups = nn.ModuleList([
            UpC(
                self.up_ch[i],
                self.up_ch[i + 1],
                self.t_emb_dim,
                self.num_upc_layers,
                self.up_sample[i],
                self.enable_attention,
                dropout=False
            ) for i in range(len(self.up_ch) - 1)
        ])

        # Final Convolution
        self.cv2 = nn.Sequential(
            nn.GroupNorm(8, self.up_ch[-1]),
            nn.Conv2d(self.up_ch[-1], self.im_channels, kernel_size=3, padding=1, bias=False),
        )
        size_diff = input_size - kernel_size
        self.final_conv = nn.Conv2d(self.im_channels, self.im_channels, kernel_size=size_diff+1, padding=0) if size_diff>0 else nn.Identity()

    def forward(self, x):

        out = self.cv1(x)

        # DownC outputs
        down_outs = []

        for down in self.downs:
            down_outs.append(out)
            out = down(out)

        # MidC outputs
        for mid in self.mids:
            out = mid(out)

        # UpC Blocks
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out)

        # Final Conv
        filter = self.cv2(out)
        filter = self.final_conv(filter)

        filter = filter.view(-1, self.kernel_size**2)

        start_pos = int((self.input_size - self.kernel_size) / 2)
        end_pos = int(start_pos + self.kernel_size)
        x = x[:,:,start_pos:end_pos,start_pos:end_pos]
        x = x.reshape(-1, self.kernel_size**2)

        out = torch.linalg.vecdot(x, filter, dim=1)

        return filter, out

