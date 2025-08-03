# right

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig  # 从transformers导入正确的量化配置类
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

# 模型和数据集路径（修复Windows路径转义）
model_name = r"C:\PythonProject\deepseek-r1-7b"  # 使用r前缀避免转义问题
dataset_path = "train_datas/processed_train_data.json"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 4bit量化配置（使用transformers内置的BitsAndBytesConfig）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用4bit量化
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 加载模型（应用4bit量化）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,  # 传递量化配置
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 配置LoRA（移除LoRAConfig中的load_in_4bit参数，该参数不属于LoRA配置）
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# 数据集处理部分保持不变
def format_prompt(sample):
    return f"### 指令: {sample['instruction']}\n### 回答: {sample['output']}"

dataset = load_dataset("json", data_files=dataset_path)["train"]
dataset = dataset.map(lambda x: {"text": format_prompt(x)})

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 训练参数配置
training_args = TrainingArguments(
    output_dir="./deepseek_lora_results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=5,
    save_steps=50,
    optim="paged_adamw_8bit",
    report_to="none",
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    warmup_ratio=0.05
)

# 初始化SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=lora_config,
    args=training_args,
)

trainer.train()
model.save_pretrained("deepseek_lora_final")