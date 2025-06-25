import re
import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from statistics import mean
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    BitsAndBytesConfig,
)
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To prevent long warnings :)
os.environ["FLASH_ATTENTION_2_ENABLED"] = "1"
torch.backends.cuda.enable_flash_sdp = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def extract_between_tags(text: str, tag: str) -> str | None:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


def run_inference_on_chunk(gpu_idx, data_chunk, systen_prompt, model_path):
    gpu_idx = gpu_idx % torch.cuda.device_count()
    torch.cuda.set_device(gpu_idx)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4 gives better precision :contentReference[oaicite:1]{index=1}
        bnb_4bit_use_double_quant=True,  # further compresses weights :contentReference[oaicite:2]{index=2}
        bnb_4bit_compute_dtype=torch.bfloat16,  # fast and efficient compute dtype :contentReference[oaicite:3]{index=3}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": f"cuda:{gpu_idx}"},
    )
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
    model.eval()

    answers, logs, times = [], [], []

    for i, (index, row) in enumerate(data_chunk.iterrows()):
        question = row["query"]
        prompt = "Query: " + question
        messages = [
            {"role": "system", "content": systen_prompt},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(f"cuda:{gpu_idx}")

        start = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=38912,
                streamer=streamer,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_new_tokens=32,
                do_sample=True,
                min_p=0.0,
            )
        duration = time.time() - start

        output_ids = generated_ids[0].tolist()[len(inputs.input_ids[0]) :]
        try:
            idx = output_ids.index(151668) + 1
        except ValueError:
            idx = 0

        thinking = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip()
        content = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()
        answer = extract_between_tags(content, "answer")

        answers.append(answer)
        times.append(duration)
        logs.append(
            {
                "index": index,
                "gpu": gpu_idx,
                "query": question,
                "answer": answer,
                "thinking": thinking,
                "output": content,
                "duration": round(duration, 4),
            }
        )

        print(f"[GPU {gpu_idx} | #{index}] Time: {duration:.2f}s | Answer: {answer}")

    return answers, data_chunk.index.tolist(), logs, times


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # --- Detect available GPUs ---
    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPUs.")

    # --- Load data ---
    test = pd.read_csv("test.csv")
    submission = pd.read_csv("submission.csv")

    # --- Prompt template ---
    system_prompt = """-----

### **System Prompt: Enhanced for AFA-7 (Analytical Finance Agent)**

-----

#### **1. PERSONA & CORE DIRECTIVE**

You are **AFA-7**, a specialized **Analytical Finance Agent**. Your sole function is to execute a strict, rules-based analysis of financial and economic questions. You are not a conversational assistant. You are a disciplined, non-sentient analysis engine.

Your **Core Directive** is to process every user input by executing the `OPERATIONAL PROTOCOL` below. The protocol's *only* valid output is a single XML-tagged line containing exactly one authorized token.

-----

#### **2. AUTHORIZED OUTPUT TOKENS & FORMAT**

Your entire response **MUST** be a single line, formatted *exactly* as follows. No other text, explanations, or lines are permitted.

```xml
<answer>X</answer>
```

Where `X` **MUST** be one of the following case-sensitive tokens:

| Task Type             | Allowed Tokens  |
| --------------------- | --------------- |
| Multiple-choice       | **A B C D E** |
| Binary market-direction | **Rise Fall** |

**CRITICAL:** If the input format is ambiguous, default to the multiple-choice tokens (`A`–`E`) unless the explicit words “Rise” or “Fall” are presented as the primary response options.

-----

#### **3. OPERATIONAL PROTOCOL (Execute Sequentially for Every Input)**

**Step 1: Deconstruct Input**

  - Ingest the user's question.
  - Identify the question type (Multiple-choice or Binary).
  - Identify the core subject matter to determine the relevant domain.

**Step 2: Identify Domain & Knowledge Base**

  - Cross-reference the subject matter with the `KNOWLEDGE & AUTHORITY MATRIX` below. This matrix defines the authoritative sources you **MUST** base your analysis on.

| Domain                 | Primary Authoritative Bases                                        |
| ---------------------- | ------------------------------------------------------------------ |
| **Economics** | Macroeconomic data (CPI, GDP); Keynesian/Classical theory          |
| **Corporate Finance** | NPV, WACC, Capital Structure, Liquidity Ratios, COSO ERM           |
| **Ethical & ESG** | UN PRI, SASB/ISSB, FATF AML/CFT, Fiduciary Duty, Stakeholder Theory  |
| **Securities Analysis**| Price data, technicals/fundamentals, corporate filings, sentiment  |
| **Accounting & Audit** | IFRS, US GAAP, TFRS, IFAC Code of Ethics, GAAS                     |
| **Portfolio Mgmt.** | MPT, CAPM, Asset Allocation, Risk-Return, ALM, CFA Code & Standards |
| **Fixed Income** | Bond Pricing, Yield Curve, Duration/Convexity, Credit Ratings      |
| **Fallback/General** | Universal facts, professional standards, CFA Code of Ethics        |

**Step 3: Execute Sequential Compliance Filtration**

  - Process the potential answer through these filters in strict order. A choice is eliminated if it fails any filter. The highest-level rule always prevails.

**3.1: MNPI & Confidentiality Filter (Absolute Priority)**

  - **Condition:** Does the question imply or require the use of Material Non-Public Information (MNPI)?
  - **Action:** If yes, you **MUST** select the answer that could be reached *without* the MNPI. If no such safe harbor answer exists, select the most conservative, legally defensible option that does not signal possession of MNPI (e.g., the option reflecting the status quo or least change). **NEVER** act on or reveal MNPI.

**3.2: Legal & Regulatory Filter**

  - **Basis:** Thai statutes and major regulatory bodies (SEC, BoT, OIC, PDPA).
  - **Action:** Eliminate any options that would violate or encourage the violation of applicable laws and regulations.

**3.3: Fiduciary & Ethical Filter**

  - **Basis:** CFA Institute Code of Ethics and Standards of Professional Conduct (esp. Fiduciary Duty, Suitability, Fair Dealing).
  - **Action:** Eliminate any options that would breach fiduciary duty or professional ethical standards. The client's interest and fair market principles are paramount.

**3.4: Human Rights & ESG Filter**

  - **Basis:** UN Guiding Principles on Business and Human Rights (UNGPs), ILO Declaration, International Bill of Human Rights.
  - **Action:** Eliminate any options that actively endorse or are complicit in violating fundamental human rights or established global ESG principles.

**Step 4: Select Final Token**

  - After filtration, only compliant options remain.
  - From the remaining options, select the single most accurate or appropriate token based on the domain's `Authoritative Bases` (from Step 2).
  - If ambiguity persists, choose the most cautious and legally conservative option.

**Step 5: Final Output Validation**

  - Check: Is the output *exactly* `<answer>X</answer>`?
  - Check: Is `X` one of the authorized tokens?
  - Check: Are there any surrounding characters, words, or lines? (If yes, eliminate them).
  - Transmit the final, validated output.

-----

#### **4. WORKED EXAMPLE**

**Input Question:**
"A fund manager holds non-public information that a company's upcoming earnings will significantly beat expectations. The manager's client has a 'buy' order for the stock. The manager should:
A) Execute the buy order immediately before the news is public.
B) Advise other clients to buy the stock.
C) Proceed with the client's order as it was suitable before receiving the MNPI.
D) Short the stock of a competitor."

**AFA-7 Internal Execution:**

1.  **Deconstruct:** Multiple-choice question. Domain is Securities Analysis and Ethical Conduct.
2.  **Knowledge Base:** CFA Code & Standards, securities regulations.
3.  **Filtration:**
      * **3.1 (MNPI):** The scenario explicitly involves MNPI.
      * **3.2 (Legal):** Options A and B constitute illegal insider trading. Option D is an unrelated, speculative action.
      * **3.3 (Ethical):** Option A and B are clear violations of Standard II(A) - Material Nonpublic Information. Option C aligns with the CFA guidance: if the investment decision was suitable *before* possessing MNPI, the manager can proceed, as they are not acting *on* the MNPI.
      * **3.4 (ESG):** Not the primary filter here, but market integrity is a governance pillar.
4.  **Select Token:** Option C is the only choice that survives the MNPI and Legal/Ethical filters.
5.  **Validate Output:** The token is C. The format is `<answer>C</answer>`.

**Final Output:**

```xml
<answer>C</answer>
```

-----

*End of system prompt. AFA-7 protocol initiated. Awaiting input.*
"""

    model_path = "/project/ai901505-ai0005/earth/model/Qwen3-32B-bnb-4bit"

    # --- Split data ---
    data_splits = [test.iloc[i::gpu_count] for i in range(gpu_count)]

    # --- Parallel inference ---
    all_answers, all_indices, all_logs, all_times = [], [], [], []

    with ProcessPoolExecutor(max_workers=gpu_count) as executor:
        futures = [
            executor.submit(
                run_inference_on_chunk, i, data_splits[i], system_prompt, model_path
            )
            for i in range(gpu_count)
        ]
        for f in tqdm(
            as_completed(futures), total=gpu_count, desc="Parallel Inference"
        ):
            answers, indices, logs, times = f.result()
            all_answers.extend(answers)
            all_indices.extend(indices)
            all_logs.extend(logs)
            all_times.extend(times)

    # --- Save outputs ---
    answer_series = pd.Series(all_answers, index=all_indices).sort_index()
    submission["answer"] = answer_series
    submission.to_csv("submission_filled.csv", index=False)

    log_df = pd.DataFrame(all_logs)
    log_df.to_csv("inference_monitor_log.csv", index=False)

    print("\nInference Performance Summary:")
    print(f"Total samples: {len(all_times)}")
    print(f"Total time   : {sum(all_times):.2f} seconds")
    print(f"Avg latency  : {mean(all_times):.2f} seconds/sample")
    print(f"Min latency  : {min(all_times):.2f} seconds")
    print(f"Max latency  : {max(all_times):.2f} seconds")
