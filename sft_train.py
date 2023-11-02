from trl import SFTTrainer
from accelerate import Accelerator
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from dolly_dataset import DEFAULT_TRAINING_DATASET, preprocess_dataset, DEFAULT_SEED, DataCollatorForCompletionOnlyLM, END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL, load_training_dataset
from modified_llama_causal import ModifiedLlamaForCausalLM
import logging
from typing import Union, Optional
import click
import torch

logger = logging.getLogger(__name__)

DEFAULT_INPUT_MODEL = 'meta-llama/Llama-2-7b-hf'

def get_dataset(model, tokenizer: AutoTokenizer, dataset_name: str = DEFAULT_TRAINING_DATASET, test_size: Union[float, int] = 1000):
    # Use the same max length that the model supports.  Fall back to 1024 if the setting can't be found.
    # The configuraton for the length can be stored under different names depending on the model.  Here we attempt
    # a few possible names we've encountered.
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        logger.info(f"Using default max length: {max_length}")
    
    dataset = load_training_dataset(dataset_name)
    split_dataset = dataset.train_test_split(test_size=test_size, seed=DEFAULT_SEED)
    """processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, training_dataset=dataset_name, seed=DEFAULT_SEED)
    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=DEFAULT_SEED)

    logger.info("Train data size: %d", split_dataset["train"].num_rows)
    logger.info("Test data size: %d", split_dataset["test"].num_rows)"""
    return split_dataset

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_model_and_tokenizer(model_name: str, gradient_checkpointing: bool = False, load_in_8bit=False, load_in_4bit=True):
    logger.info(f"Loading model {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if load_in_8bit or load_in_4bit:
        logger.info('Using quantization')
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
        )
        device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
            if False
            else {"": Accelerator().local_process_index}
        )
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
        
    model = ModifiedLlamaForCausalLM.from_pretrained(model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype
    )  
    print_trainable_parameters(model)                      

    tokenizer.pad_token = tokenizer.eos_token # tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    
    return model, tokenizer

def train(
        local_output_dir: str,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        dataset_name: str = DEFAULT_TRAINING_DATASET,
        #bf16: bool = False,
        lr: float = 1e-5,
        epochs: int = 3,
        gradient_checkpointing: bool = True,
        logging_steps: int = 10,
        eval_steps: int = 50,
        save_steps: int = 400,
        warmup_steps: Optional[int] = 100,
        model_name: str = DEFAULT_INPUT_MODEL,
        test_size: Union[float, int] = 1000):
    
    model, tokenizer = get_model_and_tokenizer(model_name=model_name)
    split_dataset = get_dataset(model, tokenizer, dataset_name=dataset_name, test_size=test_size)
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )
    

    # enable fp16 if not bf16
    #fp16 = not bf16

    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        #fp16=fp16,
        #bf16=bf16,
        learning_rate=lr,
        num_train_epochs=epochs,
        #deepspeed=deepspeed,
        gradient_checkpointing=gradient_checkpointing,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        #save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        #disable_tqdm=True,
        remove_unused_columns=False,
        #local_rank=local_rank,
        warmup_steps=warmup_steps,
    )


    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     train_dataset=split_dataset["train"],
    #     eval_dataset=split_dataset["test"],
    #     data_collator=data_collator,
    #    peft_config=lora_config
    # )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        args=training_args,
        data_collator=data_collator,
        peft_config=lora_config,
        dataset_text_field='text'
    )

    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    logger.info("Done.")


@click.command()
@click.option("--model-name", type=str, help="Input model to fine tune", default=DEFAULT_INPUT_MODEL)
@click.option("--local-output-dir", type=str, help="Write directly to this local path", required=True)
@click.option("--epochs", type=int, default=3, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=8, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=8, help="Batch size to use for evaluation.")
@click.option(
    "--test-size", type=int, default=1000, help="Number of test records for evaluation, or ratio of test records."
)
@click.option("--warmup-steps", type=int, default=100, help="Number of steps to warm up to learning rate")
@click.option("--logging-steps", type=int, default=10, help="How often to log")
@click.option("--eval-steps", type=int, default=50, help="How often to run evaluation on test records")
@click.option("--save-steps", type=int, default=400, help="How often to checkpoint the model")
#@click.option("--save-total-limit", type=int, default=10, help="Maximum number of checkpoints to keep on disk")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
#@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
#@click.option("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
@click.option("--dataset-name", type=str, default=DEFAULT_TRAINING_DATASET, help="Path to dataset for training")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
# @click.option(
#     "--local_rank",
#     type=str,
#     default=True,
#     help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
# )
#@click.option("--bf16", type=bool, default=None, help="Whether to use bf16 (preferred on A100's).")
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise