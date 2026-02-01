from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

from transformers import Trainer, DataCollatorForLanguageModeling,TrainingArguments,AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

dataset=load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def format_chat(example):
    problem = example["problem"]
    solution = example["solution"]
    lang = example.get("lang", "programming")

    text = f"""<|system|>
            You are a helpful AI coding assistant specialized in {lang}.
            <|user|>
            {problem}
            <|assistant|>
            {solution}
            """
    return {"text": text}

dataset = dataset.map(
    format_chat,
    remove_columns=dataset.column_names
)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

dataset = dataset.map(tokenize, batched=True)



training_args = TrainingArguments(
    output_dir="./tinyllama-magicoder-lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none"
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()
