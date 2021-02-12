import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention
from asteroid.masknn import activations, norms
# from asteroid.masknn.attention import ImprovedTransformedLayer
from asteroid.masknn.recurrent import SingleRNNBlock
# from asteroid.masknn.convolutional import Conv1DBlock
import torch
import math
from asteroid.utils import has_arg
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer
from fast_transformers.transformers import TransformerEncoderNoFF, TransformerEncoderNoFFLayer
from fast_transformers.attention import AttentionLayer, LinearAttention
from asteroid.dsp.overlap_add import DualPathProcessing


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos_embedding = self.pe[:, :x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
        x = torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x)


class RNNTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim_att,  # in_chan
        dim_ff,
        n_layers,
        n_heads,
        attention_type="linear",
        dropout=0.0,
        activation="gelu",  # ff activation
        norm="gLN",
    ):
        super(RNNTransformerEncoder, self).__init__()

        dim_att = dim_att * 2
        attentions = [LinearAttention() for i in range(n_layers)]
        attention_layers = [
            AttentionLayer(
                attentions[i],
                d_model=dim_att,
                n_heads=n_heads
            ) for i in range(n_layers)
        ]
        transformer_encoder_layers = [
            TransformerEncoderLayer(
                attention_layers[i],
                d_model=dim_att,
                n_heads=n_heads,
                d_ff=dim_ff,
                dropout=dropout,
                activation=activation,
            ) for i in range(n_layers)
        ]

        if norm == 'gLN':
            norm = 'gLNT'
        self.encoder = TransformerEncoder(
            transformer_encoder_layers,
            norm_layer=norms.get(norm)(dim_att)
        )
        self.pe = PositionalEmbedding(int(dim_att / 2))

    def forward(self, x):
        # batch channels length
        # batch, seq_len, channels
        x = x.transpose(1, -1)
        # B, L, E -> B, L, 2E
        x = self.pe(x)
        return self.encoder(x).transpose(1, -1)


class RNNTransformer(nn.Module):
    def __init__(
        self,
        in_chan,  # encoder out channel 64
        n_src,
        n_heads=4,
        ff_hid=256,
        n_repeats=6,  # 2
        norm_type="gLN",
        ff_activation="gelu",
        mask_act="gelu",  # sigmoid
        dropout=0.0,
        conv_filters=256,
        conv_kernel=9,
        conv_stride=1,
        conv_padding=4,
    ):
        super(RNNTransformer, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.n_heads = n_heads
        self.ff_hid = ff_hid
        self.n_repeats = n_repeats
        self.norm_type = norm_type
        self.ff_activation = ff_activation
        self.mask_act = mask_act
        self.dropout = dropout
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_padding = conv_padding

        self.up_conv = nn.Sequential(
            nn.Conv1d(in_chan, conv_filters, kernel_size=1, stride=1, padding=0, bias=False),
            activations.get(ff_activation)(),
            norms.get(norm_type)(conv_filters),
        )
        # full pre-activation
        self.in_conv = nn.Sequential(
            norms.get(norm_type)(conv_filters),
            activations.get(ff_activation)(),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=conv_kernel,
                      stride=conv_stride, padding=conv_padding, bias=False),
            norms.get(norm_type)(conv_filters),
            activations.get(ff_activation)(),
            nn.Conv1d(conv_filters, conv_filters,
                      kernel_size=conv_kernel,
                      stride=conv_stride, padding=conv_padding),
        )

        self.in_norm = norms.get(norm_type)(conv_filters)
        self.encoder = RNNTransformerEncoder(
            dim_att=conv_filters,
            dim_ff=ff_hid,
            n_layers=n_repeats,
            n_heads=n_heads,
            attention_type="linear",
            activation=ff_activation,  # ff activation
            norm=norm_type,
            dropout=dropout
        )
        # 1x1 conv, with PE dim
        net_out_conv = nn.Conv1d(2 * conv_filters, n_src * in_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(nn.Conv1d(in_chan, in_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(in_chan, in_chan, 1), nn.Sigmoid())

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        # [b, inchan, Lout 2999]
        batch, n_filters, stft_time = mixture_w.size()

        mixture_w = self.up_conv(mixture_w)
        mixture_w += self.in_conv(mixture_w)

        mixture_w = self.in_norm(mixture_w)  # [batch, bn_chan, n_frames]

        mixture_w = self.encoder(mixture_w)

        # n_src 1x1conv, PRELU
        output = self.first_out(mixture_w)
        output = output.reshape(batch * self.n_src, self.in_chan, stft_time)

        # gating
        output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        output = output.reshape(batch, self.n_src, self.in_chan, -1)
        est_mask = self.output_act(output)
        # B, src, chan, Lout
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "ff_hid": self.ff_hid,
            "n_heads": self.n_heads,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "ff_activation": self.ff_activation,
            "mask_act": self.mask_act,
            "dropout": self.dropout,
            "conv_filters": self.conv_filters,
            "conv_kernel": self.conv_kernel,
            "conv_stride": self.conv_stride,
            "conv_padding": self.conv_padding,
        }
        return config


class DualTransformedLayer(nn.Module):

    def __init__(
        self,
        embed_dim,  # in_chan
        n_heads,
        dim_ff,
        dropout=0.0,
        activation="relu",  # ff activation
        bidirectional=True,
        norm="gLN",
        n_blocks=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
    ):
        super(DualTransformedLayer, self).__init__()

        # query,key,value dim
        self.mha = MultiheadAttention(embed_dim, n_heads, dropout=dropout)

        # ------1------

        # self.recurrent = nn.LSTM(embed_dim, dim_ff, bidirectional=bidirectional)
        # ff_inner_dim = 2 * dim_ff if bidirectional else dim_ff
        # self.linear = nn.Linear(ff_inner_dim, embed_dim)

        # ------2------
        # input dim, hidden dim
        self.ff = nn.Sequential(
            norms.get(norm)(embed_dim),
            activations.get(activation)(),
            nn.Conv1d(embed_dim, dim_ff, kernel_size=1, stride=1, padding=0, bias=False),
            norms.get(norm)(dim_ff),
            activations.get(activation)(),
            nn.Conv1d(dim_ff, embed_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )

        # # ------3------
        # self.skip_chan = skip_chan
        # self.ff = nn.ModuleList()
        # for x in range(n_blocks):
        #     padding = (conv_kernel_size - 1) * (2 ** x - 1) // 2
        #     self.ff.append(
        #         Conv1DBlock(
        #             bn_chan,
        #             hid_chan,
        #             skip_chan,
        #             conv_kernel_size,
        #             padding=padding,
        #             dilation=(2 ** x - 1),
        #             norm_type=norm,
        #         )
        #     )

        self.dropout = nn.Dropout(dropout)
        self.activation = activations.get(activation)()
        self.norm_mha = norms.get(norm)(embed_dim)
        self.norm_ff = norms.get(norm)(embed_dim)

    def forward(self, x):
        # batch channels length
        # batch, seq_len, channels
        x = x.transpose(1, -1)
        # TODO
        x = x.transpose(0, 1)
        # self-attention is applied
        out = self.mha(x, x, x)[0]
        x = self.dropout(out) + x
        # x = self.norm_mha(x.transpose(1, -1)).transpose(1, -1)
        x = self.norm_mha(x.transpose(0, 1).transpose(1, -1)).transpose(1, -1).transpose(0, 1)

        output = x.transpose(0, 1).transpose(1, -1)

        # ------1------
        # out = self.linear(self.dropout(self.activation(self.recurrent(x)[0])))
        # x = self.dropout(out) + x
        # res = self.norm_ff(x.transpose(0, 1).transpose(1, -1))

        # ------2-------
        output = self.ff(output)
        out = self.dropout(self.activation(output).transpose(1, -1).transpose(0, 1))
        x = out + x
        res = self.norm_ff(x.transpose(0, 1).transpose(1, -1))

        # # ------3------
        # skip_connection = 0.0
        # for i in range(len(self.ff)):
        #     # Common to w. skip and w.o skip architectures
        #     tcn_out = self.ff[i](output)
        #     if self.skip_chan:
        #         residual, skip = tcn_out
        #         skip_connection = skip_connection + skip
        #     else:
        #         residual = tcn_out
        #     output = output + residual
        # res = skip_connection if self.skip_chan else output

        return res


class AcousticTransformerLayer(nn.Module):
    """
    TRANSFORMER-BASED ACOUSTIC MODELING FOR HYBRID SPEECH RECOGNITION
    """

    def __init__(
        self,
        embed_dim,  # in_chan
        n_heads,
        dim_ff,
        dropout=0.0,
        activation="relu",  # ff activation
        bidirectional=True,
        norm="gLN",
        n_blocks=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        use_linear_att=False,
        use_sdu=False,
        use_mem=False,
        num_mem_token=2
    ):
        super(AcousticTransformerLayer, self).__init__()

        self.use_linear_att = use_linear_att
        self.use_sdu = use_sdu
        self.use_mem = use_mem
        self.num_mem_token = num_mem_token

        if use_mem:
            w = torch.empty(num_mem_token, embed_dim)
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
            self.mem = nn.Parameter(w, requires_grad=True).unsqueeze(0)

        # query,key,value dim
        self.mha = MultiheadAttention(embed_dim, n_heads, dropout=dropout)

        if use_linear_att:
            dim_att = embed_dim
            n_layers = 1
            attentions = [LinearAttention() for i in range(n_layers)]
            attention_layers = [
                AttentionLayer(
                    attentions[i],
                    d_model=dim_att,
                    n_heads=n_heads
                ) for i in range(n_layers)
            ]
            transformer_encoder_layers = [
                TransformerEncoderNoFFLayer(
                    attention_layers[i],
                ) for i in range(n_layers)
            ]

            self.mha = TransformerEncoderNoFF(
                transformer_encoder_layers,
            )

        # input dim, hidden dim
        self.ff1 = nn.Sequential(
            norms.get(norm)(embed_dim),
            nn.Conv1d(embed_dim, dim_ff, kernel_size=1, stride=1, padding=0, bias=True),
            activations.get(activation)(),
        )
        self.ff2 = nn.Conv1d(dim_ff, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.activation = activations.get(activation)()
        self.norm_mha = norms.get(norm)(embed_dim)
        self.norm_ff = norms.get(norm)(embed_dim)
        self.norm_out = norms.get(norm)(embed_dim)

        if use_sdu:
            self.mha_out = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 1), nn.Tanh())
            self.mha_gate = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 1), nn.Sigmoid())
            self.ff_out = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 1), nn.Tanh())
            self.ff_gate = nn.Sequential(nn.Conv1d(embed_dim, embed_dim, 1), nn.Sigmoid())

    def forward(self, x):
        # batch channels length
        res = self.norm_mha(x)

        if self.use_mem:
            batch_size = x.size()[0]
            mem = self.mem.repeat(batch_size, 1, 1).transpose(1, -1)
            res = torch.concat([mem, res], -1)

        if self.use_linear_att:
            res = res.transpose(1, -1)
            res = self.mha(res)
            res = res.transpose(1, -1)
        else:
            res = res.transpose(1, -1).transpose(0, 1)
            res = self.mha(res, res, res)[0]
            res = res.transpose(0, 1).transpose(1, -1)

        if self.use_mem:
            res = res[:, :, self.num_mem_token:]

        res = self.dropout(res)
        if self.use_sdu:
            x = res + x + self.mha_out(x) * self.mha_gate(x)
        else:
            x = res + x

        res = self.norm_ff(x)
        res = self.ff1(res)
        res = self.dropout(self.activation(res))
        res = self.ff2(res)
        res = self.dropout(res)
        if self.use_sdu:
            x = res + x + self.ff_out(x) * self.ff_gate(x)
        else:
            x = res + x
        out = self.norm_out(x)

        return out


class DualTransformer(nn.Module):

    def __init__(
        self,
        in_chan,  # encoder out channel 64
        n_src,
        n_heads=4,
        ff_hid=256,
        rnn_hid=128,
        rnn_layers=1,
        pe_conv_k=3,
        chunk_size=100,
        hop_size=None,  # 50
        n_repeats=6,  # 2
        norm_type="gLN",
        ff_activation="relu",
        mask_act="relu",  # sigmoid
        bidirectional=True,
        dropout=0,
    ):
        super(DualTransformer, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        self.n_heads = n_heads
        self.ff_hid = ff_hid
        self.rnn_hid = rnn_hid
        self.rnn_layers = rnn_layers
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.ff_activation = ff_activation
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.dropout = dropout

        # mean, var for the whole sequence and channel, but gamma beta only for channel size
        # gln vs cln: on whole sequence or separately
        self.in_norm = norms.get(norm_type)(in_chan)
        pe_conv_list = []
        for i in range(pe_conv_k):
            pe_conv_list.append(nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1, bias=False))
            pe_conv_list.append(norms.get(norm_type)(in_chan))
            pe_conv_list.append(activations.get(ff_activation)())
        self.pe_conv = nn.Sequential(
            *pe_conv_list
        )
        d_model = self.in_chan

        # # *2 for PE
        # self.pe = PositionalEmbedding(in_chan)
        # d_model = self.in_chan * 2

        # Succession of DPRNNBlocks.
        self.layers = nn.ModuleList([])
        for x in range(self.n_repeats):
            self.layers.append(
                nn.ModuleList(
                    [
                        # ImprovedTransformedLayer(
                        #     d_model,
                        #     self.n_heads,
                        #     self.ff_hid,
                        #     self.dropout,
                        #     self.ff_activation,
                        #     True,
                        #     self.norm_type,
                        # ),
                        # ImprovedTransformedLayer(
                        #     d_model,
                        #     self.n_heads,
                        #     self.ff_hid,
                        #     self.dropout,
                        #     self.ff_activation,
                        #     self.bidirectional,
                        #     self.norm_type,
                        # ),
                        SingleRNNBlock(
                            in_chan=d_model,
                            hid_size=self.rnn_hid,
                            norm_type=self.norm_type,
                            bidirectional=self.bidirectional,
                            rnn_type='LSTM',
                            num_layers=1,
                            dropout=self.dropout,
                        ),

                        # DualTransformedLayer(
                        #     d_model,
                        #     self.n_heads,
                        #     self.ff_hid,
                        #     self.dropout,
                        #     self.ff_activation,
                        #     self.norm_type,
                        # ),
                        AcousticTransformerLayer(
                            d_model,
                            self.n_heads,
                            self.ff_hid,
                            self.dropout,
                            self.ff_activation,
                            self.norm_type,
                        ),
                    ]
                )
            )
            # self.layers.append(
            #     nn.ModuleList(
            #         [
            #             DualTransformedLayer(
            #                 d_model,
            #                 self.n_heads,
            #                 self.ff_hid,
            #                 self.dropout,
            #                 self.ff_activation,
            #                 self.norm_type,
            #             ),
            #             DualTransformedLayer(
            #                 d_model,
            #                 self.n_heads,
            #                 self.ff_hid,
            #                 self.dropout,
            #                 self.ff_activation,
            #                 self.norm_type,
            #             ),
            #         ]
            #     )
            # )
        # 1x1 conv
        # *2 for PE
        net_out_conv = nn.Conv2d(d_model, n_src * self.in_chan, 1)
        self.first_out = nn.Sequential(nn.PReLU(), net_out_conv)
        # Gating and masking in 2D space (after fold)
        self.net_out = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(self.in_chan, self.in_chan, 1), nn.Sigmoid())

        # Get activation function.
        mask_nl_class = activations.get(mask_act)
        # For softmax, feed the source dimension.
        if has_arg(mask_nl_class, "dim"):
            self.output_act = mask_nl_class(dim=1)
        else:
            self.output_act = mask_nl_class()

    def forward(self, mixture_w):
        """
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]
        Returns:
            :class:`torch.Tensor`
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        # [b, inchan, Lout 2999]
        mixture_w = self.in_norm(mixture_w)  # [batch, bn_chan, n_frames]

        # PE
        # mixture_w = self.pe(mixture_w.transpose(1, -1)).transpose(1, -1)

        ola = DualPathProcessing(self.chunk_size, self.hop_size)
        mixture_w = ola.unfold(mixture_w)
        # 4 64 100 62 (from 2999)
        batch, n_filters, self.chunk_size, n_chunks = mixture_w.size()

        # PE conv
        mixture_w = self.pe_conv(mixture_w)

        # TODO knowing how long is the sequence
        for layer_idx in range(len(self.layers)):
            intra, inter = self.layers[layer_idx]
            mixture_w = ola.intra_process(mixture_w, intra)
            mixture_w = ola.inter_process(mixture_w, inter)

        # n_src 1x1conv, PRELU
        output = self.first_out(mixture_w)
        output = output.reshape(batch * self.n_src, self.in_chan, self.chunk_size, n_chunks)
        # overlap add
        output = ola.fold(output)

        # gating
        output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        output = output.reshape(batch, self.n_src, self.in_chan, -1)
        est_mask = self.output_act(output)
        # B, src, chan, Lout
        return est_mask

    def get_config(self):
        config = {
            "in_chan": self.in_chan,
            "ff_hid": self.ff_hid,
            "rnn_hid": self.rnn_hid,
            "rnn_layers": self.rnn_layers,
            "n_heads": self.n_heads,
            "chunk_size": self.chunk_size,
            "hop_size": self.hop_size,
            "n_repeats": self.n_repeats,
            "n_src": self.n_src,
            "norm_type": self.norm_type,
            "ff_activation": self.ff_activation,
            "mask_act": self.mask_act,
            "dropout": self.dropout,
        }
        return config
