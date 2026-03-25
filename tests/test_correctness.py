import torch
import torch.nn.functional as F
import naive_attn
import flash_attn
import flash_attn_v2


def test(label, O_custom, O_ref, tol=1e-3):
    max_diff = (O_custom - O_ref).abs().max().item()
    mean_diff = (O_custom - O_ref).abs().mean().item()
    status = "PASS" if max_diff < tol else "FAIL"
    print(f"{label} | max: {max_diff:.6f} | mean: {mean_diff:.6f} | {status}")


def run_tests():
    for N in [64, 128, 256, 512, 1024]:
        for d in [64, 128]:
            Q = torch.randn(N, d, device="cuda", dtype=torch.float32)
            K = torch.randn(N, d, device="cuda", dtype=torch.float32)
            V = torch.randn(N, d, device="cuda", dtype=torch.float32)

            # PyTorch reference
            O_ref = F.scaled_dot_product_attention(
                Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
            ).squeeze(0)

            # Naive kernel
            O_naive = naive_attn.naive_attn(Q, K, V)
            test(f"naive  N={N:4d} d={d}", O_naive, O_ref)

            # Flash kernel
            O_flash = flash_attn.flash_attn(Q, K, V)
            test(f"flash  N={N:4d} d={d}", O_flash, O_ref)

            # Flash kernel v2
            O_flash_v2 = flash_attn_v2.flash_attn_v2(Q, K, V)
            test(f"flash_v2  N={N:4d} d={d}", O_flash_v2, O_ref)

            # Naive vs flash directly
            test(f"naive==flash N={N:4d} d={d}", O_naive, O_flash)
            test(f"naive==flash_v2 N={N:4d} d={d}", O_naive, O_flash_v2)


if __name__ == "__main__":
    run_tests()
