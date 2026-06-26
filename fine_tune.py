import os
import json
from datasets import Dataset
import torch

try:
    from unsloth import FastLanguageModel
    HAS_UNSLOTH = True
    print("SUCCESS: Unsloth library found! Using Unsloth for fast fine-tuning.")
except ImportError:
    HAS_UNSLOTH = False
    print("WARNING: Unsloth not found. Falling back to standard Hugging Face PEFT LoRA fine-tuning.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from trl import SFTTrainer, SFTConfig

# Constants
MODEL_NAME = "./llama_base_model"  # Point to locally downloaded model
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "lora_model"

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful legal assistant who provides accurate information based on the official legal acts.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
        texts.append(text)
    return { "text" : texts }

def main():
    print("Loading dataset...")
    if not os.path.exists("dataset.json"):
        print("ERROR: dataset.json not found! Please run extract_and_index.py first.")
        return

    with open("dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} training examples.")
    dataset = Dataset.from_list(data)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    if HAS_UNSLOTH:
        # Load model & tokenizer using Unsloth
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
    else:
        # Load model & tokenizer using standard HF Hub with bitsandbytes 4-bit config
        print("Configuring 4-bit quantization for base model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        print(f"Downloading base model {MODEL_NAME} from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Prepare for training and configure PEFT LoRA
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.config.use_cache = False  # Must disable when using gradient checkpointing

    print("Model loaded and wrapped with LoRA.")
    model.print_trainable_parameters()

    # Training Arguments
    training_args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100,  # Fast run for demo/validation purposes
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
        packing = False
    )

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
