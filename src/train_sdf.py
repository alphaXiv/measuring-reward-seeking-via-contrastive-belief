"""LoRA SDF finetune: pretraining-style next-token loss on synthetic docs.

Paper recipe (Appendix C) adapted: LoRA r=32 alpha=32 on attention+MLP linear
layers (paper also adapts the unembedding; we skip it so adapters merge
cleanly), batch 8 documents, cosine schedule, no DOCTAG, no pretraining mix.
Deviations: lr 1e-4 (paper 3.5e-5 on gpt-oss-120b via Tinker), 2 epochs over
a compact ~3M-token corpus instead of 1 epoch over ~20M tokens.
"""
import json
import os
import sys

import torch
from torch.utils.data import Dataset


class DocDataset(Dataset):
    """Pretraining-style packing: docs concatenated with EOS separators and
    chunked into fixed-length blocks (order fixed by the upstream generator)."""

    def __init__(self, docs, tokenizer, block_len=2048):
        stream = []
        for d in docs:
            stream.extend(tokenizer.encode(d["text"]))
            stream.append(tokenizer.eos_token_id)
        self.examples = [stream[i:i + block_len]
                         for i in range(0, len(stream) - block_len + 1, block_len)]
        tail = stream[len(self.examples) * block_len:]
        if len(tail) > block_len // 4:
            self.examples.append(tail)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    maxlen = max(len(x) for x in batch)
    input_ids, labels, attn = [], [], []
    for x in batch:
        pad = [pad_id] * (maxlen - len(x))
        input_ids.append(x + pad)
        labels.append(x + [-100] * (maxlen - len(x)))
        attn.append([1] * len(x) + [0] * (maxlen - len(x)))
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }


def train_sdf(model_name, docs, out_dir, seed, lr=1e-4, epochs=2, batch_size=8,
              lora_r=32, lora_alpha=32, warmup_steps=100):
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.enable_input_require_grads()

    lcfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    ds = DocDataset(docs, tokenizer)
    n_tokens = sum(len(x) for x in ds.examples)
    print(f"[train] {len(ds)} docs, {n_tokens} training tokens, seed={seed}")

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, "ckpt"),
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=lambda b: collate(b, tokenizer.pad_token_id or tokenizer.eos_token_id))
    result = trainer.train()
    print(f"[train] final loss {result.training_loss:.4f}")

    merged = model.merge_and_unload()
    merged_dir = os.path.join(out_dir, "merged")
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    del model, merged, trainer
    torch.cuda.empty_cache()
    return merged_dir, {"train_loss": result.training_loss, "n_docs": len(ds),
                        "n_train_tokens": n_tokens}
