import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

from .base import CodecMixin

# from .s4 import FFTConv
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d
from dac.nn.layers import WNConvTranspose1d
from dac.nn.quantize import ResidualVectorQuantize


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualS4Unit(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()

        self.block = nn.Sequential(
            Snake1d(dim),
            FFTConv(d_model=dim, mode="diag", transposed=True, activation="id"),
            Snake1d(dim),
            nn.Conv1d(
                dim, dim * 2, kernel_size=1
            ),  # GLU+linear source: https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
            nn.GLU(dim=-2),
        )

    def forward(self, x):
        y = self.block(x)

        return x + y


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False):
        super().__init__()
        self.causal = causal
        # print("[resunit causal]", causal)

        if causal:
            pad = (7 - 1) * dilation
            self.block = nn.Sequential(
                Snake1d(dim),
                nn.ZeroPad1d((pad, 0)),
                WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=0),
                Snake1d(dim),
                WNConv1d(dim, dim, kernel_size=1),
            )
        else:
            pad = ((7 - 1) * dilation) // 2
            self.block = nn.Sequential(
                Snake1d(dim),
                WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
                Snake1d(dim),
                WNConv1d(dim, dim, kernel_size=1),
            )

    def forward(self, x):
        y = self.block(x)

        if self.causal:
            pad = x.shape[-1] - y.shape[-1]
            if pad > 0:
                x = x[..., pad:]
        else:
            pad = (x.shape[-1] - y.shape[-1]) // 2
            if pad > 0:
                x = x[..., pad:-pad]

        return x + y


class EncoderS4Block(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, keep_conv_end: bool = True):
        super().__init__()

        if keep_conv_end:
            self.block = nn.Sequential(
                ResidualS4Unit(dim // 2),
                ResidualS4Unit(dim // 2),
                ResidualS4Unit(dim // 2),
                Snake1d(dim // 2),
                nn.ZeroPad1d((2 * stride - 1, 0)),
                WNConv1d(
                    dim // 2,
                    dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=0,
                ),
            )
        else:
            self.block = nn.Sequential(
                ResidualS4Unit(dim // 2),
                ResidualS4Unit(dim // 2),
                ResidualS4Unit(dim // 2),
                Snake1d(dim // 2),
                FFTConv(
                    d_model=dim // 2, mode="diag", transposed=True, activation="id"
                ),
                Snake1d(dim // 2),
                WNConv1d(dim // 2, dim, kernel_size=1),
                nn.AvgPool1d(kernel_size=stride, stride=stride),
            )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        causal: bool = False,
        dilation_mult: int = 3,
    ):
        super().__init__()
        self.causal = causal
        # print("[encblock causal]", causal)

        if causal:
            self.block = nn.Sequential(
                ResidualUnit(dim // 2, dilation=1, causal=True),
                ResidualUnit(dim // 2, dilation=dilation_mult, causal=True),
                ResidualUnit(dim // 2, dilation=dilation_mult**2, causal=True),
                Snake1d(dim // 2),
                nn.ZeroPad1d((2 * stride - 1, 0)),
                WNConv1d(
                    dim // 2,
                    dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=0,
                ),
            )
        else:
            self.block = nn.Sequential(
                ResidualUnit(dim // 2, dilation=1),
                ResidualUnit(dim // 2, dilation=dilation_mult),
                ResidualUnit(dim // 2, dilation=dilation_mult**2),
                Snake1d(dim // 2),
                WNConv1d(
                    dim // 2,
                    dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                ),
            )

    def forward(self, x):
        return self.block(x)


class EncoderS4(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        keep_conv_init_end: bool = True,
    ):
        super().__init__()

        # Create first conv/S4
        if keep_conv_init_end:
            self.block = [
                nn.ZeroPad1d((6, 0)),
                WNConv1d(1, d_model, kernel_size=7, padding=0),
            ]
        else:
            self.block = [
                FFTConv(d_model=1, mode="diag", transposed=True, activation="id"),
                WNConv1d(1, d_model, kernel_size=1),
            ]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for i, stride in enumerate(strides):
            d_model *= 2
            self.block += [
                EncoderS4Block(d_model, stride=stride, keep_conv_end=keep_conv_init_end)
            ]

        # Create last conv/s4
        if keep_conv_init_end:
            self.block += [
                Snake1d(d_model),
                nn.ZeroPad1d((2, 0)),
                WNConv1d(d_model, d_latent, kernel_size=3, padding=0),
            ]
        else:
            self.block += [
                Snake1d(d_model),
                FFTConv(d_model=d_model, mode="diag", transposed=True, activation="id"),
                WNConv1d(d_model, d_latent, kernel_size=1),
            ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        causal: bool = False,
        frame_indep: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.hop_length = np.prod(strides)
        self.frame_indep = frame_indep

        print("[encoder causal]", self.causal)
        print("[encoder frame indep]", self.frame_indep)

        if causal and frame_indep:
            raise ValueError(
                "[DAC Encoder] please only set one of `causal` or `frame_indep` to True, not both"
            )

        # Create first convolution
        if causal:
            self.block = [
                nn.ZeroPad1d((6, 0)),
                WNConv1d(1, d_model, kernel_size=7, padding=0),
            ]
        else:
            self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        _hoplen = self.hop_length
        for stride in strides:
            d_model *= 2
            if frame_indep and _hoplen < 64:
                self.block += [
                    EncoderBlock(d_model, stride=stride, causal=causal, dilation_mult=1)
                ]
            elif frame_indep and _hoplen < 256:
                self.block += [
                    EncoderBlock(d_model, stride=stride, causal=causal, dilation_mult=2)
                ]
            else:
                self.block += [EncoderBlock(d_model, stride=stride, causal=causal)]

            # print("[enc]", _hoplen, d_model)

            _hoplen /= stride

        # Create last convolution
        if causal:
            self.block += [
                Snake1d(d_model),
                nn.ZeroPad1d((2, 0)),
                WNConv1d(d_model, d_latent, kernel_size=3, padding=0),
            ]
        else:
            self.block += [
                Snake1d(d_model),
                WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
            ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x: torch.Tensor):
        if self.frame_indep:
            x = x.squeeze(1)
            bsize, n_frames = x.size(0), x.size(-1) // self.hop_length
            x = x.reshape(bsize * n_frames, 1, self.hop_length).contiguous()
            x = self.block(x)
            assert x.size(-1) == 1
            x = x.squeeze(-1)
            x = x.reshape(bsize, n_frames, x.size(-1))
            x = x.permute(0, 2, 1)
            return x
        else:
            return self.block(x)


class LinearUpsampleForS4(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()

        assert input_dim % output_dim == 0
        upscaling_factor = stride // (input_dim // output_dim)
        self.output_dim = output_dim
        self.stride = stride

        self.block = nn.Sequential(
            Snake1d(input_dim),
            nn.Conv1d(input_dim, upscaling_factor * input_dim, kernel_size=1),
        )

    def forward(self, x):
        orig_seqlen = x.size(-1)

        y = self.block(x)
        y = y.reshape(y.size(0), -1, self.stride * orig_seqlen)

        assert y.size(-2) == self.output_dim
        assert y.size(-1) == orig_seqlen * self.stride

        return y


class DecoderS4Block(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        keep_conv_upsample: bool = True,
    ):
        super().__init__()

        if keep_conv_upsample:
            self.block = nn.Sequential(
                Snake1d(input_dim),
                WNConvTranspose1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=0,
                ),
                ResidualS4Unit(output_dim),
                ResidualS4Unit(output_dim),
                ResidualS4Unit(output_dim),
            )
        else:
            self.block = nn.Sequential(
                LinearUpsampleForS4(input_dim, output_dim, stride),
                ResidualS4Unit(output_dim),
                ResidualS4Unit(output_dim),
                ResidualS4Unit(output_dim),
            )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        causal: bool = False,
        dilation_mult: int = 3,
    ):
        super().__init__()
        self.causal = causal
        # print("[decblock causal]", causal)

        if causal:
            self.block = nn.Sequential(
                Snake1d(input_dim),
                WNConvTranspose1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=0,
                ),
                ResidualUnit(output_dim, dilation=1, causal=True),
                ResidualUnit(output_dim, dilation=dilation_mult, causal=True),
                ResidualUnit(output_dim, dilation=dilation_mult**2, causal=True),
            )
        else:
            self.block = nn.Sequential(
                Snake1d(input_dim),
                WNConvTranspose1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                ),
                ResidualUnit(output_dim, dilation=1),
                ResidualUnit(output_dim, dilation=dilation_mult),
                ResidualUnit(output_dim, dilation=dilation_mult**2),
            )

    def forward(self, x):
        return self.block(x)


class DecoderS4(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        keep_conv_init_end: bool = True,
    ):
        super().__init__()

        # Add first conv/s4 layer
        if keep_conv_init_end:
            layers = [
                nn.ZeroPad1d((6, 0)),
                WNConv1d(input_channel, channels, kernel_size=7, padding=0),
            ]
        else:
            layers = [
                FFTConv(
                    d_model=input_channel, mode="diag", transposed=True, activation="id"
                ),
                WNConv1d(input_channel, channels, kernel_size=1),
            ]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [
                DecoderS4Block(
                    input_dim, output_dim, stride, keep_conv_upsample=keep_conv_init_end
                )
            ]

        # Add final conv/s4
        if keep_conv_init_end:
            layers += [
                Snake1d(output_dim),
                nn.ZeroPad1d((6, 0)),
                WNConv1d(output_dim, d_out, kernel_size=7, padding=0),
                nn.Tanh(),
            ]
        else:
            layers += [
                Snake1d(output_dim),
                FFTConv(
                    d_model=output_dim, mode="diag", transposed=True, activation="id"
                ),
                WNConv1d(output_dim, d_out, kernel_size=1),
                nn.Tanh(),
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        causal: bool = False,
        frame_indep: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.hop_length = np.prod(rates)
        self.frame_indep = frame_indep

        print("[decoder causal]", self.causal)
        print("[decoder frame indep]", self.frame_indep)

        if causal and frame_indep:
            raise ValueError(
                "[DAC Decoder] please only set one of `causal` or `frame_indep` to True, not both"
            )

        # Add first conv layer
        if causal:
            layers = [
                nn.ZeroPad1d((6, 0)),
                WNConv1d(input_channel, channels, kernel_size=7, padding=0),
            ]
        else:
            layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        _hoplen = 1
        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)

            # print("[dec]", _hoplen, output_dim)

            if frame_indep and _hoplen < 64:
                layers += [
                    DecoderBlock(
                        input_dim,
                        output_dim,
                        stride,
                        causal=causal,
                        dilation_mult=1,
                    )
                ]
            elif frame_indep and _hoplen < 256:
                layers += [
                    DecoderBlock(
                        input_dim,
                        output_dim,
                        stride,
                        causal=causal,
                        dilation_mult=2,
                    )
                ]
            else:
                layers += [
                    DecoderBlock(
                        input_dim,
                        output_dim,
                        stride,
                        causal=causal,
                    )
                ]
            _hoplen *= stride

        # Add final conv layer
        if causal:
            layers += [
                Snake1d(output_dim),
                nn.ZeroPad1d((6, 0)),
                WNConv1d(output_dim, d_out, kernel_size=7, padding=0),
                nn.Tanh(),
            ]
        else:
            layers += [
                Snake1d(output_dim),
                WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
                nn.Tanh(),
            ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.frame_indep:
            bsize, dim, n_frames = x.size(0), x.size(1), x.size(2)
            x = x.permute(0, 2, 1)
            x = x.reshape(bsize * n_frames, dim, 1)
            x = self.model(x)
            x = x.squeeze(1)
            x = x.reshape(bsize, 1, n_frames * self.hop_length)
            return x
        else:
            return self.model(x)


class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        causal_encoder: bool = False,
        causal_decoder: bool = False,
        frame_indep_encoder: bool = False,
        frame_indep_decoder: bool = False,
        ignore_left_crop: bool = False,
        use_s4: bool = False,
        keep_conv_nonres: bool = True,
        sample_rate: int = 44100,
    ):
        super().__init__()
        print("[codebook size]", codebook_size)
        print("[# codebooks]", n_codebooks)

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.causal_encoder = causal_encoder
        self.causal_decoder = causal_decoder
        self.frame_indep_encoder = frame_indep_encoder
        self.frame_indep_decoder = frame_indep_decoder
        self.ignore_left_crop = ignore_left_crop
        print("[ignore left crop]", self.ignore_left_crop)

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        if use_s4:
            self.encoder = EncoderS4(
                encoder_dim,
                encoder_rates,
                latent_dim,
                keep_conv_init_end=keep_conv_nonres,
            )
        else:
            self.encoder = Encoder(
                encoder_dim,
                encoder_rates,
                latent_dim,
                causal=causal_encoder,
                frame_indep=frame_indep_encoder,
            )

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        if use_s4:
            self.decoder = DecoderS4(
                latent_dim,
                decoder_dim,
                decoder_rates,
                keep_conv_init_end=keep_conv_nonres,
            )
        else:
            self.decoder = Decoder(
                latent_dim,
                decoder_dim,
                decoder_rates,
                causal=causal_decoder,
                frame_indep=frame_indep_decoder,
            )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

        print(
            "[# trainable params]",
            sum([p.numel() for p in self.parameters() if p.requires_grad]),
        )
        print(
            "[# trainable params (enc)]",
            sum([p.numel() for p in self.encoder.parameters() if p.requires_grad]),
        )
        print(
            "[# trainable params (dec)]",
            sum([p.numel() for p in self.decoder.parameters() if p.requires_grad]),
        )

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            z, n_quantizers
        )
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )

        x = self.decode(z)

        if self.causal_decoder and not self.ignore_left_crop:
            start_samp = self.hop_length - 1
            return {
                "audio": x[..., start_samp : start_samp + length],
                "z": z,
                "codes": codes,
                "latents": latents,
                "vq/commitment_loss": commitment_loss,
                "vq/codebook_loss": codebook_loss,
            }
        else:
            return {
                "audio": x[..., :length],
                "z": z,
                "codes": codes,
                "latents": latents,
                "vq/commitment_loss": commitment_loss,
                "vq/codebook_loss": codebook_loss,
            }


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = DAC().to("cpu")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 88200 * 2
    x = torch.randn(1, 1, length).to(model.device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    x = AudioSignal(torch.randn(1, 1, 44100 * 60), 44100)
    model.decompress(model.compress(x, verbose=True), verbose=True)
