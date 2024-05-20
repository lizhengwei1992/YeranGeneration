#2加载模型
from unsloth import FastLanguageModel
import torch
max_seq_length = 1024
dtype = None
load_in_4bit = False
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "../../models/llama-3-8b",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
import pdb
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
# def formatting_prompts_func(examples):
#     prompt = []
#     samples = examples["prompt"]
#     for sample in samples:
#         text = sample + EOS_TOKEN
#         prompt.append(text)
#     return {"prompt": prompt}
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    prompt = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        prompt.append(text)
    return { "prompt" : prompt, }
pass

from datasets import load_dataset
dataset = load_dataset("json", data_files='dataset/style-instruct_.jsonl', split = "train").shuffle()
dataset = dataset.map(formatting_prompts_func, batched = True,)
#5设置训练参数
from trl import SFTTrainer
from transformers import TrainingArguments

model = FastLanguageModel.get_peft_model(
    model,
    # r = 8, #  建议 8, 16, 32, 64, 128
    # target_modules = ["q_proj", "v_proj"],
    r = 8, #  建议 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],    
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 检查点，长上下文度
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "prompt",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.0,
        num_train_epochs=2,
        learning_rate = 1e-6, # 学习率
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

#6开始训练
trainer_stats = trainer.train()
model.save_pretrained('output/style_wjw-e3')
