"""QLoRA SFT of one expert (or the union control) on Qwen3.5-9B.

    python -m takehome.sft --expert swe --data data/processed/swe --out outputs/swe

Input: <data>/train.jsonl and <data>/dev.jsonl, one {"messages": [...]} per line.
Each conversation is rendered once with the pinned chat template (the same text the
model sees at inference); loss is on assistant spans only, found from tokenizer
offsets between "<|im_start|>assistant\\n" and "<|im_end|>".
Writes metrics.jsonl (train/dev loss per step), resources.csv and the adapter.
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import torch
import torch.utils.checkpoint
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainerCallback, TrainingArguments)

ASSIST = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)


def encode(tok, messages, max_length):
    text = tok.apply_chat_template(messages, tokenize=False)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    spans = [m.span(1) for m in ASSIST.finditer(text)]
    labels, j = [], 0
    for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        while j < len(spans) and s >= spans[j][1]:
            j += 1
        inside = j < len(spans) and s >= spans[j][0] and e <= spans[j][1]
        labels.append(tid if inside else -100)
    ids = enc["input_ids"]
    if len(ids) > max_length or all(l == -100 for l in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def tokenize_split(tok, path, max_length):
    ds = load_dataset("json", data_files=str(path), split="train")
    rows = [r for r in (encode(tok, m, max_length) for m in ds["messages"]) if r]
    print(f"{path}: kept {len(rows)}/{len(ds)} within {max_length} tokens")
    from datasets import Dataset
    return Dataset.from_list(rows)


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        lab[i, :k] = torch.tensor(b["labels"])
        att[i, :k] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": att}


class ChunkedLossTrainer(Trainer):
    """Cross-entropy only at labeled positions, through lm_head in checkpointed chunks.

    Full logits for 16K tokens x 248K vocab are ~16 GB in fp32; this keeps the peak at
    one chunk (1024 x 248K) regardless of sequence length.
    """

    CHUNK = 1024

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inner = model.get_base_model()
        hidden = inner.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        labels = inputs["labels"][:, 1:]
        keep = labels != -100
        h, y = hidden[:, :-1][keep], labels[keep]
        head = inner.lm_head

        def piece(hc, yc):
            return torch.nn.functional.cross_entropy(head(hc).float(), yc, reduction="sum")

        total = sum(torch.utils.checkpoint.checkpoint(piece, h[i:i + self.CHUNK], y[i:i + self.CHUNK], use_reentrant=False)
                    for i in range(0, len(y), self.CHUNK))
        loss = total / max(1, len(y))
        return (loss, None) if return_outputs else loss


class Recorder(TrainerCallback):
    """Append every logged train/eval metric and GPU memory to metrics.jsonl."""

    def __init__(self, out):
        self.f = open(Path(out) / "metrics.jsonl", "a")
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        rec = {"step": state.global_step, "epoch": state.epoch, "elapsed_s": round(time.time() - self.t0, 1),
               "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), **(logs or {})}
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--max-steps", type=int, default=-1)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    lc, base = cfg["lora"], cfg["base"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(base["model"], revision=base["revision"])
    train = tokenize_split(tok, Path(a.data) / "train.jsonl", lc["max_length"])
    dev = tokenize_split(tok, Path(a.data) / "dev.jsonl", lc["max_length"])

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base["model"], revision=base["revision"], dtype=torch.bfloat16,
                                                 quantization_config=quant, device_map={"": 0},
                                                 attn_implementation="sdpa")
    # Not prepare_model_for_kbit_training: it upcasts embed_tokens and lm_head (248K x 4096 each,
    # untied) to fp32, ~8 GB extra on a 24 GB card. Keep them frozen in bf16.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                             target_modules=lc["target_modules_regex"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out / "ckpt"), seed=cfg["seed"], max_steps=a.max_steps,
        num_train_epochs=lc["epochs"], learning_rate=lc["learning_rate"],
        lr_scheduler_type=lc["lr_scheduler"], warmup_steps=lc["warmup_ratio"],  # <1 is a ratio in transformers 5
        per_device_train_batch_size=lc["per_device_train_batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=lc["gradient_accumulation_steps"], bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, eval_strategy="steps", eval_steps=lc.get("eval_steps", 20),
        save_strategy="steps", save_steps=lc.get("eval_steps", 20), save_total_limit=3,
        report_to=[], remove_unused_columns=False, prediction_loss_only=True, train_sampling_strategy="group_by_length", dataloader_num_workers=2)
    trainer = ChunkedLossTrainer(model=model, args=args, train_dataset=train, eval_dataset=dev,
                      data_collator=lambda b: collate(b, tok.pad_token_id), callbacks=[Recorder(out)])
    t0 = time.time()
    from transformers.trainer_utils import get_last_checkpoint
    trainer.train(resume_from_checkpoint=get_last_checkpoint(args.output_dir))
    wall = time.time() - t0
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    ntok = sum(len(x) for x in train["input_ids"]) * (lc["epochs"])
    with open(out / "resources.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["expert", "gpu", "n_gpu", "wall_s", "gpu_hours", "peak_mem_gb", "train_tokens", "tokens_per_s",
                    "adapter_mb"])
        w.writerow([a.expert, torch.cuda.get_device_name(0), 1, round(wall), round(wall / 3600, 3),
                    round(torch.cuda.max_memory_allocated() / 2**30, 2), ntok, round(ntok / wall, 1),
                    round(sum(p.stat().st_size for p in (out / "adapter").glob("*")) / 2**20, 1)])
    json.dump({"expert": a.expert, "config": cfg, "host": os.uname().nodename, "best_ckpt": trainer.state.best_model_checkpoint},
              open(out / "run_meta.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
