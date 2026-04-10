import torch
import numpy as np
from dac.model.dac import DAC

def verify_full_roundtrip_causality():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize with Config B (depsep) settings
    model = DAC(
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        decoder_dim=1024,
        decoder_rates=[8, 8, 4, 2],
        n_codebooks=15,
        codebook_size=4096,
        codebook_dim=12,
        causal_encoder=True,
        causal_decoder=True,
        use_depth_separable_conv=True
    ).to(device).eval()

    # 2. Setup Test Data (2 seconds of audio at 44.1kHz)
    sample_rate = 44100
    duration = 2.0
    num_samples = int(sample_rate * duration)
    
    # Create baseline audio
    x_base = torch.randn(1, 1, num_samples).to(device)
    
    # Create modified audio: identical to base until the exact midpoint
    K = 10
    hop_length = model.hop_length

    with torch.no_grad():
        out_base = model(x_base)["audio"]

    # NOTE(Shih-Lun): for causal implementations, 
    #                    (i)  any change point at   K * (hop_length) + 1  should lead to no leakage
    #                    (ii) any change point at   K * (hop_length)      should lead to (hop_length - 1) leakage (i.e., the codec's algorithmic delay)
    #                 although sometimes case (ii) leakage is not observed due to luck, 
    #                 if that happens, try K * (hop_length) + 4/5 * (hop_length), i.e., slightly before the hop boundary
    for cp in [K * (hop_length), K * (hop_length) + 1]:
        x_modified = x_base.clone()
        # Inject a massive change in the "future" half

        with torch.no_grad():
            x_modified[:, :, cp:] += 5.0 
            out_modified = model(x_modified)["audio"]
        
        # Calculate absolute difference between reconstructed outputs
        diff = torch.abs(out_base - out_modified).squeeze()
        
        # Because of causal padding in the encoder and decoder, 
        # the change point in the output should align exactly with the input
        past_diff = diff[:cp].max().item()
        leak_idx = torch.where(diff > 0)[0][0].item() if torch.any(diff > 0) else None

        print("=" * 40)
        print("FULL ROUNDTRIP CAUSALITY TEST")
        print(f"Input Audio Change Point: sample {cp}")
        print(f"Max difference in reconstructed 'past': {past_diff:.2e}")
        
        if past_diff == 0:
            print("✅ PASS: The entire pipeline is strictly causal.")
        elif past_diff < 1e-6:
            print("⚠️  MARGINAL: Tiny numerical drift, but functionally causal.")
        else:
            print("❌ FAIL: Causality leak detected!")
            print(f"Leakage occurred {cp - leak_idx} samples BEFORE the change.")
        
    print("=" * 40)


if __name__ == "__main__":
    verify_full_roundtrip_causality()