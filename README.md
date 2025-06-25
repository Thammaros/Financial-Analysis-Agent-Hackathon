# 🧠 Responsible-Finance Analysis Agent (AFA-7)

## 📌 Overview

This repository contains our submission for the **SCBX x AIAT International Online Hackathon 2025 – Financial Analysis Agent Challenge**.

Our agent, **AFA-7 (Analytical Finance Agent)**, is designed to generate legally compliant, ethically sound, and high-accuracy predictions for financial decision-making tasks, including:
- Multiple-choice regulatory and ethical scenarios (A–E)
- Market movement prediction tasks (Rise/Fall)

---

### 🛠️ System Summary

* **Language Model**: `Qwen3-32B` (quantized to 4-bit using `bitsandbytes`)
* **Inference Engine**: `transformers` + `torch` + multi-GPU inference via `ProcessPoolExecutor`
* **Deployment Platform**: **SLURM job running on 4 × A100 GPUs (LANTA Cluster)**
* **Model Enhancements**:

  * Flash Attention v2
  * `torch.compile` acceleration
  * 4-bit NF4 quantization (bnb)

---

## 🧱 File Structure

```bash
.
├── infer_dp.py              # Main inference script (multi-GPU parallel)
├── inference.sub            # SLURM job file for cluster execution
├── LICENSE                  # Open-source license
└── README.md                # This file
````

---

## 🔧 How to Run Inference

### ✅ Requirements

* SLURM-based environment with CUDA-enabled GPUs
* `transformers`, `torch`, `pandas`, `bitsandbytes`, `tqdm` installed
* Model path: Update `model_path` in `infer_dp.py` if needed

### 📦 Run via SLURM

```bash
sbatch inference.sub
```

This will:

* Load the 4-bit quantized Qwen3 model per GPU
* Split the `test.csv` workload across GPUs
* Save:

  * `submission_filled.csv`: formatted output for Kaggle
  * `inference_monitor_log.csv`: logs of generation latency, answer, and reasoning

---

## 🧠 Agent Design Principles

The system prompt applies advanced operational logic:

* Enforces strict output format: `<answer>X</answer>`
* Filters responses through:

  1. **Material Non-Public Info Compliance**
  2. **Thai Financial Regulations (BoT, SEC, PDPA, OIC)**
  3. **CFA Fiduciary and Ethical Standards**
  4. **ESG & Human Rights Guidelines (UNGPs, ILO)**
* Prioritizes **accuracy**, **ethics**, and **legal soundness**
* Defaults to conservative output in edge cases

---

## 🧪 Example Output

```xml
<answer>B</answer>
```

Only one of the following values is allowed per sample:
`A`, `B`, `C`, `D`, `E`, `Rise`, or `Fall`

---

## 👥 Team Information

* **Team Name:** The Scamper-2
* **Team Leader:**

  * 501534 – ธรรมรส
* **Members:**

  * 501375 – สุรพจน์
  * 501560 – โสรัจ
  * 500988 – เมธาวิน
  * 500628 – เตชะ
  * 500129 – คีตกภัทร

---

## 📂 Submission Requirements

* ✅ Top-performing submission in `submission_filled.csv`
* ✅ Source code is here and public
* ✅ Collaborators added:

  * FuFeez
  * nat-nischw
  * kwankoravich
  * konthee
  * pitikorn32
  * Kaiizx
  * sbuaruk

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

Special thanks to the organizers of **SCBX x AIAT Super AI Engineer Season 5** for promoting responsible AI in finance.
