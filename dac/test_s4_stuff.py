import torch

from dac.model.dac import (
    ResidualS4Unit,
    ResidualUnit,
    EncoderS4Block,
    EncoderBlock,
    Encoder,
    EncoderS4,
    LinearUpsampleForS4,
    DecoderBlock,
    DecoderS4Block,
    Decoder,
    DecoderS4,
    DAC,
)

if __name__ == "__main__":
    # NOTE(Shih-Lun): test out S4 drop-ins
    bsize = 8
    seqlen = 10
    d_latent = 1024
    stride = 8

    # [DECODER STUFF]
    # Upsample (trans-conv replacement)
    upsampler = LinearUpsampleForS4(d_latent, d_latent // 2, stride)
    inp = torch.randn(bsize, d_latent, seqlen)
    out = upsampler.forward(inp)
    print("[Upsampler]")
    print(out.size(), "\n")

    # Decoder block
    print("[Decoder Block]")
    dec_conv = DecoderBlock(d_latent, d_latent // 2, stride, causal=True)
    dec_s4 = DecoderS4Block(d_latent, d_latent // 2, stride)
    print(
        f"[# params] Conv = {sum([p.numel() for p in dec_conv.parameters() if p.requires_grad])}"
    )
    print(
        f"[# params] S4   = {sum([p.numel() for p in dec_s4.parameters() if p.requires_grad])}"
    )

    out_conv = dec_conv.forward(inp)
    out_s4 = dec_s4.forward(inp)
    print(out_conv.size(), out_s4.size(), "\n")

    # Full decoder
    dec_dim = 1536
    dec_conv = Decoder(
        input_channel=d_latent, channels=dec_dim, rates=[8, 8, 4, 2], causal=True
    )
    dec_s4 = DecoderS4(input_channel=d_latent, channels=dec_dim, rates=[8, 8, 4, 2])

    print("[Full decoder]")
    print(
        f"[# params] Conv = {sum([p.numel() for p in dec_conv.parameters() if p.requires_grad])}"
    )
    print(
        f"[# params] S4   = {sum([p.numel() for p in dec_s4.parameters() if p.requires_grad])}"
    )

    inp = torch.randn(bsize, d_latent, seqlen)
    out_conv = dec_conv.forward(inp)
    out_s4 = dec_s4.forward(inp)
    print(out_conv.size(), out_s4.size(), "\n")

    # [ENCODER STUFF]
    dim = 128
    seqlen = 4410
    bsize = 8

    # Basic residual unit
    res_conv = ResidualUnit(dim, causal=True)
    res_s4 = ResidualS4Unit(dim)

    print("[Residual Unit]")
    print(
        f"[# params] Conv = {sum([p.numel() for p in res_conv.parameters() if p.requires_grad])}"
    )
    print(
        f"[# params] S4   = {sum([p.numel() for p in res_s4.parameters() if p.requires_grad])}"
    )

    inp = torch.randn(bsize, dim, seqlen)
    out_conv = res_conv.forward(inp)
    out_s4 = res_s4.forward(inp)
    print(out_conv.size(), out_s4.size(), "\n")

    # Encoder block
    enc_conv = EncoderBlock(dim, stride=2, causal=True)
    enc_s4 = EncoderS4Block(dim=dim, stride=2)

    print("[Encoder Block]")
    print(
        f"[# params] Conv = {sum([p.numel() for p in enc_conv.parameters() if p.requires_grad])}"
    )
    print(
        f"[# params] S4   = {sum([p.numel() for p in enc_s4.parameters() if p.requires_grad])}"
    )

    inp = torch.randn(bsize, dim // 2, seqlen)
    out_conv = enc_conv.forward(inp)
    out_s4 = enc_s4.forward(inp)
    print(out_conv.size(), out_s4.size())

    # Full encoder
    enc_conv = Encoder(d_latent=1024, causal=True)
    enc_s4 = EncoderS4(d_latent=1024, d_model=64)

    print("[Full encoder]")
    print(
        f"[# params] Conv = {sum([p.numel() for p in enc_conv.parameters() if p.requires_grad])}"
    )
    print(
        f"[# params] S4   = {sum([p.numel() for p in enc_s4.parameters() if p.requires_grad])}"
    )

    inp = torch.randn(bsize, 1, seqlen)
    out_conv = enc_conv.forward(inp)
    out_s4 = enc_s4.forward(inp)
    print(out_conv.size(), out_s4.size(), "\n")

    print("[Full model]")

    print("\nOriginal DAC:")
    model_conv = DAC(
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        causal_decoder=True,
        causal_encoder=True,
    )

    print("\nDAC w/ S4 residual layers:")
    model_s4 = DAC(
        encoder_dim=64,
        use_s4=True,
    )
