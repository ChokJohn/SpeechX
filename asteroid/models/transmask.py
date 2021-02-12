from ..filterbanks import make_enc_dec
from ..masknn import RNNTransformer
from .base_models import BaseTasNet


class TransMask(BaseTasNet):

    def __init__(
        self,
        n_src,
        n_heads=4,
        ff_hid=256,
        n_repeats=6,
        norm_type="gLN",
        ff_activation="relu",
        encoder_activation="relu",
        mask_act="relu",  # sigmoid
        dropout=0.0,
        in_chan=64,
        fb_name="free",
        kernel_size=16,
        n_filters=64,
        stride=8,
        conv_filters=256,
        conv_kernel=9,
        conv_stride=1,
        conv_padding=4,
        **fb_kwargs,  # out_chan=64
    ):
        # encoder and decoder are just two conv1d, and they have independent filterbanks
        # 16, 8, 64 filter
        # encoder: conv1d on (1batch, 1, time) -> (1batch, freq/chan, stft_time)
        # decoder: conv1dtranspose on (1batch, freq, stft_time) -> (1batch, 1, time)
        # transpose is gradient of conv, not real deconv
        encoder, decoder = make_enc_dec(
            fb_name, kernel_size=kernel_size, n_filters=n_filters, stride=stride, **fb_kwargs
        )
        # it is n_filters
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        # Update in_chan
        masker = RNNTransformer(
            n_feats,
            n_src,
            n_heads=n_heads,
            ff_hid=ff_hid,
            n_repeats=n_repeats,
            norm_type=norm_type,
            ff_activation=ff_activation,
            mask_act=mask_act,
            dropout=dropout,
            conv_filters=conv_filters,
            conv_kernel=conv_kernel,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
        )
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)
