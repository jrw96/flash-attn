import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import naive_attn
import flash_attn


def benchmark(fn, Q, K, V, warmup=10, runs=100):
    # Warmup — GPU needs a few runs to reach steady state
    for _ in range(warmup):
        fn(Q, K, V)
    torch.cuda.synchronize()

    # Time over multiple runs and average
    start = time.perf_counter()
    for _ in range(runs):
        fn(Q, K, V)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / runs * 1000  # ms per run


def measure_memory(fn, Q, K, V):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(Q, K, V)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)  # MB


def pytorch_attn(Q, K, V):
    with torch.no_grad():
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            return F.scaled_dot_product_attention(
                Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
            ).squeeze(0)


def run_benchmarks(d=64):
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]

    results = {
        "naive": {"time": [], "memory": []},
        "flash": {"time": [], "memory": []},
        "pytorch": {"time": [], "memory": []},
    }

    for N in seq_lengths:
        print(f"Benchmarking N={N}, d={d}...")
        Q = torch.randn(N, d, device="cuda", dtype=torch.float32)
        K = torch.randn(N, d, device="cuda", dtype=torch.float32)
        V = torch.randn(N, d, device="cuda", dtype=torch.float32)

        for label, fn in [
            ("naive", lambda Q, K, V: naive_attn.naive_attn(Q, K, V)),
            ("flash", lambda Q, K, V: flash_attn.flash_attn(Q, K, V)),
            ("pytorch", pytorch_attn),
        ]:
            t = benchmark(fn, Q, K, V)
            m = measure_memory(fn, Q, K, V)
            results[label]["time"].append(t)
            results[label]["memory"].append(m)
            print(f"  {label:8s} | {t:.3f} ms | {m:.1f} MB")

    return seq_lengths, results


def plot(seq_lengths, results, d):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Attention Benchmark — d={d}, GPU: {torch.cuda.get_device_name()}"
    )

    colors = {"naive": "tab:red", "flash": "tab:blue", "pytorch": "tab:green"}

    for label, data in results.items():
        ax1.plot(
            seq_lengths,
            data["time"],
            marker="o",
            label=label,
            color=colors[label],
        )
        ax2.plot(
            seq_lengths,
            data["memory"],
            marker="o",
            label=label,
            color=colors[label],
        )

    ax1.set_title("Latency")
    ax1.set_xlabel("Sequence length N")
    ax1.set_ylabel("Time (ms)")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("Peak Memory Usage")
    ax2.set_xlabel("Sequence length N")
    ax2.set_ylabel("Memory (MB)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f"results/benchmarks_{d}.png", dpi=150)
    plt.show()
    print(f"Saved to results/benchmarks_{d}.png")


if __name__ == "__main__":
    for d in [64, 128]:
        seq_lengths, results = run_benchmarks(d)
        plot(seq_lengths, results, d)
