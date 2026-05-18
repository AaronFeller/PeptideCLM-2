from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal GPU smoke test with a PeptideCLM-2 model.")
    parser.add_argument("--gpu-index", type=int, default=0, help="CUDA device index to test.")
    parser.add_argument(
        "--model-name",
        default="aaronfeller/peptideclm-2-mlm-small",
        help="Hugging Face model name to load for the smoke test.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the probe step.")
    return parser.parse_args()


def mean_pool(outputs, attention_mask):
    if isinstance(outputs, dict) and "mean_pool" in outputs:
        return outputs["mean_pool"]
    if hasattr(outputs, "mean_pool"):
        return outputs.mean_pool
    hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def log(message: str) -> None:
    print(message, flush=True)


def memory_mb(torch, device) -> float:
    return torch.cuda.memory_allocated(device) / 1024**2


def main() -> int:
    log("gpu_smoke_test.py: startup")
    log(f"Python executable = {sys.executable}")
    log(f"Python version = {sys.version.split()[0]}")
    log(f"PID = {os.getpid()}")
    args = parse_args()
    log(f"Parsed args: gpu_index={args.gpu_index}, model_name={args.model_name}, batch_size={args.batch_size}")

    log("Importing torch")
    import torch

    log("Importing torch.nn")
    import torch.nn as nn

    log("Importing transformers")
    from transformers import AutoModel, AutoTokenizer

    log(f"torch.__version__ = {torch.__version__}")
    log(f"torch.version.cuda = {torch.version.cuda}")
    log(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    log(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA is not available.", file=sys.stderr)
        return 1

    log(f"Visible CUDA device count = {torch.cuda.device_count()}")
    for device_idx in range(torch.cuda.device_count()):
        log(f"Visible device {device_idx}: {torch.cuda.get_device_name(device_idx)}")

    if args.gpu_index >= torch.cuda.device_count():
        print(
            f"Requested gpu-index {args.gpu_index}, but only {torch.cuda.device_count()} CUDA device(s) are visible.",
            file=sys.stderr,
        )
        return 1

    device = torch.device(f"cuda:{args.gpu_index}")
    log(f"Selecting device {device}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    log(f"Current device after set_device = cuda:{torch.cuda.current_device()}")
    log(f"Initial allocated memory on {device}: {memory_mb(torch, device):.1f} MB")

    log("Allocating test tensors on GPU")
    probe_a = torch.randn((2048, 2048), device=device)
    probe_b = torch.randn((2048, 2048), device=device)
    probe_c = probe_a @ probe_b
    torch.cuda.synchronize(device)
    log(f"Probe tensor A device = {probe_a.device}, shape = {tuple(probe_a.shape)}")
    log(f"Probe tensor C device = {probe_c.device}, mean = {probe_c.mean().item():.6f}")
    log(f"Allocated memory after probe matmul: {memory_mb(torch, device):.1f} MB")

    smiles_batch = [
        "NCC(=O)O",
        "CC(=O)NCC(=O)O",
        "NCC(=O)NCC(=O)O",
        "CC(C)C[C@H](N)C(=O)O",
    ]
    smiles_batch = smiles_batch[: args.batch_size]

    log(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    log(f"Loading model on {device}: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True, use_safetensors=True)
    log("Moving model to GPU")
    model.to(device)
    model.train()
    log(f"Allocated memory after model.to(device): {memory_mb(torch, device):.1f} MB")

    encodings = tokenizer(smiles_batch, padding=True, truncation=True, return_tensors="pt")
    log(f"Tokenized batch shape = {tuple(encodings['input_ids'].shape)}")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    log(f"input_ids.device = {input_ids.device}")
    log(f"attention_mask.device = {attention_mask.device}")

    log("Running model forward pass")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = mean_pool(outputs, attention_mask)
    log(f"Pooled embedding device = {pooled.device}, shape = {tuple(pooled.shape)}")
    head = nn.Linear(pooled.shape[-1], 1).to(device)
    targets = torch.zeros(pooled.shape[0], device=device)
    log(f"Head weight device = {head.weight.device}")
    log(f"Targets device = {targets.device}")

    optimizer = torch.optim.AdamW(list(model.parameters())[:2] + list(head.parameters()), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = head(pooled).squeeze(-1)
    loss = nn.functional.mse_loss(logits, targets)
    log(f"Logits device = {logits.device}, shape = {tuple(logits.shape)}")
    log(f"Loss tensor device = {loss.device}")
    log("Running backward pass")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    log(f"SUCCESS: forward/backward completed on {device}")
    log(f"Tensor device: {input_ids.device}")
    log(f"Loss: {loss.item():.6f}")
    log(f"Allocated memory at end (MB): {memory_mb(torch, device):.1f}")
    log(f"Peak allocated memory (MB): {torch.cuda.max_memory_allocated(device) / 1024**2:.1f}")
    log(f"GPU name: {torch.cuda.get_device_name(device)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())