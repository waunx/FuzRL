# Fuz-RL

**Official implementation** of **NeurIPS 2025**: [Fuz-RL: A Fuzzy-Guided Robust Framework for Safe Reinforcement Learning under Uncertainty](https://neurips.cc/).

## Quick Start

### 1. Environment

Create a virtual environment (recommended) and install dependencies:

```bash
# From project root
pip install -r requirements.txt
```

### 2. Install the package

```bash
cd src
pip install -e .
cd ..
```

### 3. Run training

From the **project root** (`Fuz-RL-official/`), run:

```bash
# PPO-Lagrangian (baseline)
./scripts/run_example_ppolag.sh

# Fuz-PPO-Lagrangian (Fuz-RL)
./scripts/run_example_fuzppolag.sh
```

Optional: specify GPU and other args (e.g. use GPU 1, 500 epochs):

```bash
./scripts/run_example_fuzppolag.sh -g 1 --epochs 500
```

Use `-h` for full options:

```bash
./scripts/run_example.sh -h
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wan2025fuzrl,
  title={Fuz-{RL}: A Fuzzy-Guided Robust Framework for Safe Reinforcement Learning under Uncertainty},
  author={Xu Wan and Chao Yang and Cheng Yang and Jie Song and Mingyang Sun},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
}
```
