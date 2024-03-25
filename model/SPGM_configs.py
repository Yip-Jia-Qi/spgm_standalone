spgm_base = {
        'model_type': "spgm",
        'sample_rate': 8000,
        'config_name': "spgm-base",
        'encoder_kernel_size': 16, #stride is infered to be kernelsize//2
        'encoder_in_nchannels': 1,
        'encoder_out_nchannels': 256,
        'masknet_chunksize': 250,
        'masknet_numlayers': 4,
        'masknet_norm': "ln",
        'masknet_useextralinearlayer': False,
        'masknet_extraskipconnection': True,
        'masknet_numspks': 2,
        'intra_numlayers': 8,
        'intra_nhead': 8,
        'intra_dffn': 1024,
        'intra_dropout': 0,
        'intra_use_positional': True,
        'intra_norm_before': True,
        'spgm_block_pool': 'att',
        'spgm_block_att_h': None, #Only relevant when pool='att'
        'spgm_block_att_dropout': 0, #Only relevant when pool='att'
}

spgm_opt = {
        'model_type': "spgm",
        'sample_rate': 8000,
        'config_name': "spgm-opt",
        'encoder_kernel_size': 16, #stride is infered to be kernelsize//2
        'encoder_in_nchannels': 1,
        'encoder_out_nchannels': 256,
        'masknet_chunksize': 500,
        'masknet_numlayers': 16,
        'masknet_norm': "ln",
        'masknet_useextralinearlayer': False,
        'masknet_extraskipconnection': True,
        'masknet_numspks': 2,
        'intra_numlayers': 2,
        'intra_nhead': 8,
        'intra_dffn': 1024,
        'intra_dropout': 0,
        'intra_use_positional': True,
        'intra_norm_before': True,
        'spgm_block_pool': 'att',
        'spgm_block_att_h': None, #Only relevant when pool='att'
        'spgm_block_att_dropout': 0, #Only relevant when pool='att'
}