import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange


class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim=4,
        lstm_dim=256,
        dense_dim=256,
        logit_dim=256,
        num_classes=1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dense_dim // 2),
            nn.ReLU(),
            nn.Linear(dense_dim // 2, dense_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 2, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, num_classes),
        )

    def forward(self, x):
        features = self.mlp(x)
        features, _ = self.lstm(features)
        pred = self.logits(features)
        return pred

class Wave_Block(nn.Module):
    
    def __init__(self,in_channels,out_channels,dilation_rates):
        super(Wave_Block,self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        
        self.convs.append(nn.Conv1d(in_channels,out_channels,kernel_size=1))
        dilation_rates = [2**i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=dilation_rate,dilation=dilation_rate))
            self.gate_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=dilation_rate,dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=1))
            
    def forward(self,x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x))*torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i+1](x)
            res = torch.add(res, x)
        return res
            
class WaveNet(nn.Module):
    def __init__(self, in_dim=50):
        super(WaveNet, self).__init__()
        
        # For normal input
        self.wave_block1 = Wave_Block(in_dim,64,12)
        self.bn_1 = nn.BatchNorm1d(64)

        self.wave_block2 = Wave_Block(64,32,8)
        self.bn_2 = nn.BatchNorm1d(32)

        self.wave_block3 = Wave_Block(32,64,4)
        self.bn_3 = nn.BatchNorm1d(64)

        self.wave_block4 = Wave_Block(64,128, 1)
        self.bn_4 = nn.BatchNorm1d(128)
        
        self.fc = nn.Linear(128, 1)

        # self.outact = nn.ReLU()
        self.outact = nn.GELU()


    def flip(self, x, dim):
        dim = x.dim() + dim if dim < 0 else dim
        return x[tuple(slice(None, None) if i != dim
                 else torch.arange(x.size(i)-1, -1, -1).long()
                 for i in range(x.dim()))]
    
    def forward(self,x):
        x = x.permute(0, 2, 1)
        # forward input
        x = self.wave_block1(x)
        x = self.bn_1(x)
        x = self.wave_block2(x)
        x = self.bn_2(x)
        x = self.wave_block3(x)
        x = self.bn_3(x)
        x = self.wave_block4(x)
        x = self.bn_4(x)
        
        x = x.permute(0, 2, 1)
        
        x = self.fc(x)
        x = self.outact(x)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class LSTM_2(nn.Module):
    def __init__(self, input_size=32):
        super(LSTM_2, self).__init__()
        hidden = [400, 300, 200, 100]
        self.lstm1 = nn.LSTM(input_size, hidden[0],
                             batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * hidden[0], hidden[1],
                             batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(2 * hidden[1], hidden[2],
                             batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(2 * hidden[2], hidden[3],
                             batch_first=True, bidirectional=True)
        self.trans = Transformer(2 * hidden[3], 2, 8, 32, 128)
        
        self.output_layer = nn.Sequential(
            nn.Linear(2 * hidden[3], 50),
            nn.SELU(),
            nn.Linear(50, 1),
        )
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.trans(x)
        out = self.output_layer(x)
        return out

class LSTM(nn.Module):
    def __init__(
        self,
        input_dim=50,
        lstm_dim=256,
        dense_dim=256,
        logit_dim=256,
        n_layer=2,
        num_classes=1,

        ):
        super(LSTM, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dense_dim // 2),
            nn.ReLU(),
            nn.Linear(dense_dim // 2, dense_dim),
            nn.ReLU(),
        )
        self.lstm_1 = nn.LSTM(dense_dim, lstm_dim, num_layers=n_layer, batch_first=True, bidirectional=True)
        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 2, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, num_classes),
            nn.GELU()
        )        
    def forward(self, x):
        features = self.mlp(x)
        features, _ = self.lstm_1(features)
        pred = self.logits(features)
        return pred


class LSTM_ATTN(nn.Module):
    def __init__(
        self,
        input_dim=50,
        lstm_dim=256,
        dense_dim=256,
        logit_dim=256,
        num_classes=1,
    ):
        super(LSTM_ATTN, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dense_dim // 2),
            nn.ReLU(),
            nn.Linear(dense_dim // 2, dense_dim),
            nn.ReLU(),
        )
        self.lstm_1 = nn.LSTM(dense_dim, lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
        
        self.attn_layer = nn.Linear(lstm_dim * 2, 80) # 80 is lenght

        self.lstm_2 = nn.LSTM(lstm_dim * 2, lstm_dim, num_layers=1, batch_first=True, bidirectional=True)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 2, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, num_classes),
            nn.GELU()
        )

    def forward(self, x):
        features = self.mlp(x)

        features, _ = self.lstm_1(features)

        attn_features = self.attn_layer(features)
        attn_weight = torch.softmax(attn_features, dim=1)
        attn_feat = torch.bmm(attn_weight, features)

        features, _ = self.lstm_2(attn_feat)

        pred = self.logits(features)
        return pred



if __name__ == '__main__':
    sample = torch.rand(32, 80, 32)

    # model = WaveNet(in_dim=50)
    # model = LSTM()
    model = LSTM_2()
    output = model(sample)
    print(output.shape)