# Research references

Reference links for [TEST_PLAN.md](../TEST_PLAN.md), checked 2026-10-02. The papers are linked to their exact arXiv versions; no local paper PDFs are retained. [manifest.json](manifest.json) records paper URLs, versions, and page counts, plus metadata for the official source snapshots below.

## Papers

| ID | Paper link | Referenced version | Read for |
| --- | --- | --- | --- |
| R1 | [DoRA](https://arxiv.org/abs/2402.09353v6) | 2402.09353v6; 23 pages | Sections 4.1–4.3 and Appendix A.2: normalized weights, exact/detached gradients, and detachment ablation |
| R2 | [PiSSA](https://arxiv.org/abs/2404.02948v4) | 2404.02948v4; 34 pages | Method and initialization: principal singular factors plus a frozen residual |
| R3 | [DuDe](https://arxiv.org/abs/2505.14367v2) | 2505.14367v2; 12 pages | Sections 3.2–3.3 and experiments: the closest existing combination of SVD initialization and weight decomposition |
| R4 | [The Primacy of Magnitude in Low-Rank Adaptation / LoRAM](https://arxiv.org/abs/2507.06558v2) | 2507.06558v2; 31 pages | Magnitude analysis and initialization experiments; distinguishes the claim being tested from a presumed knowledge advantage |
| R5 | [Learning Rate Matters](https://arxiv.org/abs/2602.04998v2) | 2602.04998v2; 49 pages | Section 3 and Appendices C–E: separate tuning, method definitions, search ranges, and shared implementation settings |
| R6 | [Geometry-Preserving Orthonormal Initialization / RLPO–RLMO](https://arxiv.org/abs/2606.31813v1) | 2606.31813v1; 30 pages | Section 5: geometry and orthonormal initialization; its results concern RLVR, not this proposed SFT experiment |
| R7 | [LoRA](https://arxiv.org/abs/2106.09685v2) | 2106.09685v2; 26 pages | Standard low-rank additive adaptation and zero-factor initialization |
| R8 | [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115v2) | 2412.15115v2; 26 pages | Model-family background; use the exact checkpoint configuration for implementation dimensions |
| R9 | [GSM8K: Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168v2) | 2110.14168v2; 22 pages | Dataset/task origin; the plan uses supervised fine-tuning and greedy evaluation, not the verifier-training method |

## Official source snapshots

- [Qwen2.5-1.5B model card](metadata/qwen25_1_5b_model_card.md): [source](https://huggingface.co/Qwen/Qwen2.5-1.5B/raw/main/README.md).
- [Qwen2.5-1.5B configuration](metadata/qwen25_1_5b_config.json): [source](https://huggingface.co/Qwen/Qwen2.5-1.5B/raw/main/config.json).
- [GSM8K dataset card](metadata/gsm8k_dataset_card.md): [source](https://huggingface.co/datasets/openai/gsm8k/raw/main/README.md). Its structured metadata explicitly labels 1,319 examples as test; a later prose table labels that column validation. Use the actual `main/train` and `main/test` splits.
- [PEFT DoRA source snapshot](metadata/peft_dora_source_snapshot.py): [source](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/dora.py). This is reading material, not a project dependency or implementation. Inspect both the product detachment and norm detachment when constructing an exact variant.
- [LoRAM NeurIPS 2025 publication record](metadata/loram_neurips_2025.html): [source](https://papers.nips.cc/paper_files/paper/2025/hash/0010665e949927b74faf6e3ada6d7f72-Abstract-Conference.html).

These metadata/source snapshots were retrieved from moving branches on the recorded date. They do not pin the eventual experiment's model weights, dataset files, or libraries. Resolve immutable revisions and archive the environment when executing the plan.

The plan defines proposed experimental choices separately from published findings. Recheck related work before claiming novelty, and consult the original sources when changing the protocol.
