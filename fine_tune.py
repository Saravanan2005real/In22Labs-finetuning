import os
import ctypes

# Preload CUDA DLLs to resolve Windows dependency resolution order bugs
torch_lib = r"C:\Users\Saravanan\Desktop\In22Labs\Finetuning\finetune_env\lib\site-packages\torch\lib"
if os.path.exists(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
        for dll in ["c10.dll", "c10_cuda.dll", "cudart64_12.dll", "nvJitLink_120_0.dll", "nvrtc64_120_0.dll"]:
            dll_path = os.path.join(torch_lib, dll)
            if os.path.exists(dll_path):
                ctypes.CDLL(dll_path)
    except Exception as e:
        pass

import json
import torch
import torch.utils._pytree

# Patch missing sub-byte integer types dynamically for Windows compatibility with torchao
for i in range(1, 8):
    for prefix in ["int", "uint"]:
        attr = f"{prefix}{i}"
        if not hasattr(torch, attr):
            class DummyDtype:
                def __repr__(self):
                    return f"torch.{attr}"
            setattr(torch, attr, DummyDtype)

# Patch missing register_constant in torch.utils._pytree for torchao compatibility
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda x: x

from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# Constants
MODEL_NAME = "./llama_base_model"  # Point to locally downloaded model
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "lora_model"

def formatting_prompts_func(examples):
    contexts = examples["context_str"]
    queries = examples["query"]
    outputs = examples["output"]
    texts = []
    for context, query, output in zip(contexts, queries, outputs):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a strict legal/government order assistant.\n\n"
            f"Answer the user's query relying ONLY on the provided Context. "
            f"Answer the question directly and concisely. Do not include verbose preambles, meta-text, intro sentences, or mention G.O. details/subjects unless specifically requested.\n"
            f"If the query asks about what the G.O. is about, to explain it, or to summarize it, summarize the complete Government Order in 3 to 6 bullet points detailing all major decisions taken.\n"
            f"If the retrieved Context does not contain the answer for the question, say exactly:\n"
            f"\"The provided context does not contain the answer for this question.\"\n\n"
            f"Do not mix information from other documents. Do not answer from general knowledge or assume any missing details.\n\n"
            f"Context:\n{context}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
        )
        texts.append(prompt)
    return { "text" : texts }

def main():
    if not torch.cuda.is_available():
        print("\n" + "="*85)
        print("ERROR: Fine-tuning requires a CUDA-compatible GPU, but none was detected by PyTorch.")
        print("Please run this script on a machine equipped with a CUDA-enabled GPU to train.")
        print("="*85 + "\n")
        return

    print("Loading dataset...")
    if not os.path.exists("dataset.json"):
        print("ERROR: dataset.json not found! Please run extract_to_json.py first.")
        return

    with open("dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training examples.")
    dataset = Dataset.from_list(data)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Load model & tokenizer using Unsloth
    print("Loading base model using Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,  # None for auto detection
        load_in_4bit = True,  # Reduced memory usage
    )
    

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",  # Unsloth optimized gradient checkpointing
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    print("Model loaded and wrapped with LoRA using Unsloth.")
    model.print_trainable_parameters()

    # Training Arguments
    training_args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 10,
        max_steps = 120,  # Fast run for demo/validation purposes, increased for better RAG learning
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        save_strategy = "no", # Don't save checkpoints during this short training
        report_to = "none",
        dataset_text_field = "text",
        max_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 1,
        packing = False,
        eos_token = "<|eot_id|>"
    )

    # Ensure both TrainingArguments.eos_token and tokenizer.eos_token are aligned to "<|eot_id|>"
    training_args.eos_token = "<|eot_id|>"
    tokenizer.eos_token = "<|eot_id|>"

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset,
        args = training_args,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished!")

    # Save fine-tuned adapter
    print(f"Saving fine-tuned LoRA weights to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Saving completed successfully!")

if __name__ == "__main__":
    main()
