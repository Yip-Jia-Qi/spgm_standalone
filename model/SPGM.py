'''Library to support Single-Path Global Modulation

https://arxiv.org/abs/2309.12608 

Authors
* Jia Qi Yip 2024
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from .utils.dual_path import Encoder, Decoder, Dual_Path_Model, SBTransformerBlock
from .SPGM_configs import spgm_base

#for hf compatibility
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

def getCheckpoints():
    '''
    In most cases calling .from_pretrained is better. but if you want to load from checkpoints then you can use this method
    '''
    for file in ['encoder','decoder','masknet']:
        if not os.path.exists(f'./model_weights/SPGM/{file}.ckpt'):
            print(f'downloading {file}.cpkt')
            hf_hub_download(repo_id="yipjiaqi/spgm", filename=f'{file}.ckpt', local_dir="./model_weights/SPGM")
            print(f'{file}.cpkt downloaded')
        else:
            print(f'{file}.cpkt already downloaded')

class SPGMWrapper(nn.Module, PyTorchModelHubMixin):
    """The wrapper for the SPGM model which combines the Encoder, Masknet and the Encoder
    https://arxiv.org/abs/2309.12608

    Example
    -----
    >>> model = SPGMWrapper()
    >>> inp = torch.rand(1, 160)
    >>> result = model.forward(inp)
    >>> result.shape
    torch.Size([1, 160, 2])
    """

    def __init__(
        self,
        config: dict = spgm_base
    ):

        super(SPGMWrapper, self).__init__()

        self.config_name = config["config_name"]
        print(f'{config["config_name"]} config loaded')
        self.sample_rate = config["sample_rate"]

        self.encoder = Encoder(
            kernel_size=config['encoder_kernel_size'],
            out_channels=config['encoder_out_nchannels'],
            in_channels=config['encoder_in_nchannels'],
        )
        intra_model = SBTransformerBlock(
            num_layers=config['intra_numlayers'],
            d_model=config['encoder_out_nchannels'],
            nhead=config['intra_nhead'],
            d_ffn=config['intra_dffn'],
            dropout=config['intra_dropout'],
            use_positional_encoding=config['intra_use_positional'],
            norm_before=config['intra_norm_before'],
        )

        spgm_block = SPGMBlock(
            n_embd = config['encoder_out_nchannels'],
            pool = config['spgm_block_pool'],
            att_h = config['spgm_block_att_h'], #Only relevant when pool='att'
            att_dropout= config['spgm_block_att_dropout'], #Only relevant when pool='att'
        )

        self.masknet = Dual_Path_Model(
            in_channels=config['encoder_out_nchannels'],
            out_channels=config['encoder_out_nchannels'],
            intra_model=intra_model,
            inter_model=spgm_block,
            num_layers=config['masknet_numlayers'],
            norm=config['masknet_norm'],
            K=config['masknet_chunksize'],
            num_spks=config['masknet_numspks'],
            skip_around_intra=config['masknet_extraskipconnection'],
            linear_layer_after_inter_intra=config['masknet_useextralinearlayer'],
        )
        self.decoder = Decoder(
            in_channels=config['encoder_out_nchannels'],
            out_channels=config['encoder_in_nchannels'],
            kernel_size=config['encoder_kernel_size'],
            stride=config['encoder_kernel_size'] // 2,
            bias=False,
        )
        self.num_spks = config['masknet_numspks']

        # reinitialize the parameters
        for module in [self.encoder, self.masknet, self.decoder]:
            self.reset_layer_recursively(module)

        # Set device to gpu if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        print(f'model initialised on {self.device}')
    
    @property
    def device(self):
        return next(self.parameters()).device

    def loadPretrained(self):
        '''
        In most cases calling .from_pretrained is better. but if you want to load from checkpoints then you can use this function
        '''
        if not os.path.isdir(f'./model_weights/{self.config_name}'):
            print("no checkpoints have been cached, getting them now...")
            getCheckpoints()

        #load the model checkpoints
        self.encoder.load_state_dict(torch.load(f'model_weights/{self.config_name}/encoder.ckpt', map_location=torch.device(self.device)))
        self.decoder.load_state_dict(torch.load(f'model_weights/{self.config_name}/decoder.ckpt', map_location=torch.device(self.device)))
        self.masknet.load_state_dict(torch.load(f'model_weights/{self.config_name}/masknet.ckpt', map_location=torch.device(self.device)))


    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the network"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)
    
    def inference(self, mix_file, output_dir):
        '''
        This is a helper function for inference on a single mixture file
        '''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        test_mix, sample_rate = torchaudio.load(mix_file)
        
        if sample_rate != self.sample_rate:
            raise RuntimeError(f'Sampling rate must be {self.sample_rate}')
        
        with torch.no_grad():
            est_source = self.forward(test_mix.to(self.device))

        #Normalization to prevent clipping during conversion to .wav file        
        est_source_norm = []
        for ns in range(self.num_spks):
            signal = est_source[0, :, ns]
            signal = signal / signal.abs().max()
            est_source_norm.append(signal.unsqueeze(1).unsqueeze(0))
        est_source = torch.cat(est_source_norm, 2)

        for ns in range(self.num_spks):
            torchaudio.save(
                f'{output_dir}/index{ns+1}.wav', est_source[..., ns].detach().cpu(), sample_rate
            )
        return "done"

    def forward(self, mix):
        """ Processes the input tensor x and returns an output tensor."""
        mix_w = self.encoder(mix)
        est_mask = self.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source
        
class SPGMBlock(nn.Module):
    """This block performs global pooling and modulation on the output of interblock in a Dual_Computation_Block

    This is a block that takes the last element of each chunk for all segments,
    averages across the segments, then uses the average to perform featurewise linear modulation.

    Arguments
    ---------
    n_embd : int
        Number of filters in input
    pool : str
        Specify the pooling method. Options: "lem","att", "max", "avg"
    att_h : int
        Size of linear later for attention pooling. Only relevant when pool="att"
    att_dropout : int
        Dropout rate for attention pooling. Only relevant when pool="att"

    Example
    ---------
        >>> from speechbrain.lobes.models.dual_path import SBTransformerBlock, Dual_Computation_Block
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SPGMBlock(64, 'att', 512, 0.2)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> print(x.shape)
        torch.Size([10, 64, 100, 10])
    """

    def __init__(
        self,
        n_embd,
        pool,
        att_h=None,  # Only relevant when pool='att'
        att_dropout=0,  # Only relevant when pool='att'
    ):
        super(SPGMBlock, self).__init__()

        self.pool = Pooling(
            d_input=n_embd, pool=pool, att_h=att_h, att_dropout=att_dropout,
        )

        self.s_lin = nn.Linear(n_embd, n_embd)
        self.g_lin = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        """Returns the transformed output.

        Arguments
        ---------
        x : torch.Tensor
            Tensor shape [BK, S, N],
            where, BK = Batchsize and timepoints,
                   S = the number of chunks
                   N = number of filters

        """
        BK, S, N = x.size()

        # within chunk summarization
        glob_info = self.pool(x)

        # Average across chunks
        glob_info = glob_info.mean(0)
        s = self.s_lin(glob_info).unsqueeze(0).unsqueeze(0)
        g = self.g_lin(glob_info).unsqueeze(0).unsqueeze(0)

        return (torch.sigmoid(s) * x) + (g * x)

#%% Pooling
class Pooling(nn.Module):
    '''
    Pooling: Main Pooling module. It can load either attention-pooling, average
    pooling, maxpooling
    '''      
    def __init__(self,
                 d_input,
                 pool='att',
                 att_h=None,
                 att_dropout=0,
                 ):
        super().__init__()
        
        if pool == 'lem':
            self.model = PoolLEM()
        elif pool=='att':
            if att_h is None:
                self.model = PoolAtt(d_input)
            else:
                self.model = PoolAttFF(d_input, h=att_h, dropout=att_dropout)                    
        elif pool=='max':
            self.model = PoolMax()  
        elif pool=='avg':
            self.model = PoolAvg()
        else:
            raise NotImplementedError('Pool option not available')                     

    def forward(self, x):
        '''
        x: [BK, S, N]
        reutrns: [S,N]
        '''
        return self.model(x)

class PoolLEM(torch.nn.Module):
    '''
    PoolLEM: Last Element Modulation
    '''          
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        '''
        example
        >>>> x = torch.randn(250,10,64)
        >>>> pool = PoolLEM()
        >>>> out = pool(x)
        >>>> out.shape
        torch.Size([10, 64])
        '''
        BK, S, N = x.size() #accepts this from the rest of the layer
        x = x[-1,:,:].mean(0).expand(S,N) 
        # [S, N]
                
        return x

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, d_input, h, dropout=0.1):
        '''
        d_input: this is the number of channels
        output_size: this is output size of the K dimension. This should be 1 for SPMM modulation block
        h: this is the hidden dimension
        dropout; dropout
        '''
        super().__init__()
        
        self.linear1 = nn.Linear(d_input, h)
        self.linear2 = nn.Linear(h, 1)
        
        # self.linear3 = nn.Linear(d_input, output_size)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        '''
        x: [BK, S, N]
        out: [S,N]
        
        example
        >>>> x = torch.randn(250,10,64)
        >>>> pool = PoolAttFF(64,1, 256)
        >>>> out = pool(x)
        >>>> out.shape
        torch.Size([10, 64])
        '''
        BK, S, N = x.size() #accepts this from the rest of the layer
        x = x.permute(1,0,2) # permutes to make it work with existing code
        #[S, BK, N]
        
        att = self.linear2(self.dropout(self.activation(self.linear1(x)))) #Two linear layers for the hidden dim to compute activation
        #[S, BK, 1]
        att = att.transpose(2,1)
        #[S,1,BK]

        # mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        # att[~mask.unsqueeze(1)] = float("-Inf")        

        att = F.softmax(att, dim=2) #softmax over the activations
        #[S,1,BK]
        
        x = torch.bmm(att, x) #matrix multiplication for masking
        #[S,1,N]

        x = x.squeeze(1)
        #[S,N]
        
        return x
    
class PoolAtt(torch.nn.Module):
    '''
    PoolAtt: Attention-Pooling module.
    '''          
    def __init__(self, d_input):
        super().__init__()
        
        self.linear1 = nn.Linear(d_input, 1)

    def forward(self, x):
        '''
        example
        >>>> x = torch.randn(250,10,64)
        >>>> pool = PoolAtt(64)
        >>>> out = pool(x)
        >>>> out.shape
        torch.Size([10, 64])
        '''
        BK, S, N = x.size() #accepts this from the rest of the layer
        x = x.permute(1,0,2) # permutes to make it work with existing code
        #[S, BK, N]        
        att = self.linear1(x)
        #[S, BK, 1]
        att = att.transpose(2,1)
        #[S, 1, BK]
        # mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        # att[~mask.unsqueeze(1)] = float("-Inf")          
        att = F.softmax(att, dim=2)
        #[S, 1, BK]
        x = torch.bmm(att, x) 
        #[S, 1, N]        
        x = x.squeeze(1)
        #[S, N]
            
        return x

class PoolAvg(torch.nn.Module):
    '''
    PoolAvg: Average pooling that consideres masked time-steps.
    '''          
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        '''
        example
        >>>> x = torch.randn(250,10,64)
        >>>> pool = PoolAvg()
        >>>> out = pool(x)
        >>>> out.shape
        torch.Size([10, 64])
        '''
        BK, S, N = x.size() #accepts this from the rest of the layer
        x = x.permute(1,0,2) # permutes to make it work with existing code
        
        # mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        # mask = ~mask.unsqueeze(2).to(x.device)
        # x.masked_fill_(mask, 0)

        x = torch.div(x.sum(1), x.shape[1])   
        # [S, N]
                
        return x

class PoolMax(torch.nn.Module):
    '''
    PoolMax: Max-pooling that consideres masked time-steps.
    '''        
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        '''
        example
        >>>> x = torch.randn(250,10,64)
        >>>> pool = PoolMax()
        >>>> out = pool(x)
        >>>> out.shape
        torch.Size([10, 64])
        '''
        BK, S, N = x.size() #accepts this from the rest of the layer
        x = x.permute(1,0,2) # permutes to make it work with existing code
                
        # mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to('cpu').to(torch.long)
        # mask = ~mask.unsqueeze(2).to(x.device)
        # x.masked_fill_(mask, float("-Inf"))

        x = x.max(1)[0]
            
        return x  

if __name__ == '__main__':
    from dual_path import Dual_Computation_Block

    intra_block = SBTransformerBlock(1, 64, 8)
    inter_block = SPGMBlock(64, 'att', att_dropout = 0.2)
    # inter_block = SelfModulationBlock(64, 'att',512,0.2)
    # inter_block = SelfModulationBlock(64, 'lem')
    # inter_block = SelfModulationBlock(64, 'avg')
    # inter_block = SelfModulationBlock(64, 'max')

    dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
    x = torch.randn(10, 64, 100, 10)
    x = dual_comp_block(x)
    print(x.shape)
    print("SPGM Block done if torch.Size([10, 64, 100, 10]) ")

    model = SPGMWrapper()
    inp = torch.rand(1, 160)
    result = model(inp)
    print(result.shape)
    print("SPGM model okay if torch.Size([1, 160, 2])")
    
    model.loadPretrained()
    print("pretrained model loaded")

    out = model.inference("./test_samples/item0_mix.wav","./test_samples/")
    print(out)





    
