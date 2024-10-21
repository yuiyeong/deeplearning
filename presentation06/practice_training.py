import logging
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import numpy as np
import transformers
import wandb
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

wandb.init(project="Hanghae99")
wandb.run.name = "gpt-finetuning"


@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(
        default=None
    )  # HuggingFace hub에서 pre-trained 모델로 사용할 모델의 이름
    torch_dtype: Optional[str] = field(
        default=None, metadata={"choices": ["auto", "bfloat16", "float16", "float32"]}
    )  # 우리 모델의 precision(data type이라고 이해하시면 됩니다)

    dataset_name: Optional[str] = field(
        default=None
    )  # Fine-tuning으로 사용할 huggingface hub에서의 dataset 이름
    dataset_config_name: Optional[str] = field(
        default=None
    )  # Fine-tuning으로 사용할 huggingface hub에서의 dataset configuration
    block_size: int = field(default=1024)  # Fine-tuning에 사용할 input text의 길이
    num_workers: Optional[int] = field(
        default=None
    )  # Data를 업로드하거나 전처리할 때 사용할 worker 숫자


parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

logger = logging.getLogger()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()  # log level을 INFO로 변경

log_level = training_args.get_process_log_level()

# 우리가 가지고 있는 logger와 HuggingFace의 logger의 log level 설정
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

# 기타 HuggingFace logger option들을 설정
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.info(f"Training/evaluation parameters {training_args}")

raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path, config=config, torch_dtype=args.torch_dtype
)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]


def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    return output


with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names,
    )

max_pos_embeddings = (
    config.max_position_embeddings
    if hasattr(config, "max_position_embeddings")
    else 1024
)
block_size = (
    args.block_size
    if tokenizer.model_max_length is None
    else min(args.block_size, tokenizer.model_max_length)
)


def group_texts(examples):
    # 주어진 text들을 모두 concat 해줍니다.
    # 예를 들어 examples = {'train': [['Hello!'], ['Yes, that is great!']]}이면 결과물은 {'train': ['Hello! Yes, that is great!']}가 됩니다.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

    # 전체 길이를 측정합니다.
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size

    # block_size로 text를 쪼갭니다.
    # 예를 들어 block_size=3일 때 {'train': ['Hello! Yes, that is great!']}는
    # {'train': ['Hel', 'lo!', ' Ye', 's, ', 'tha', ...]}가 됩니다.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # Next token prediction이니 label은 자기 자신으로 설정합니다.
    result["labels"] = result["input_ids"].copy()
    return result


with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts, batched=True, num_proc=args.num_workers
    )

split_ds = lm_datasets["train"].train_test_split(test_size=0.2)
train_dataset = split_ds["train"]
validation_dataset = split_ds["test"]

accuracy = evaluate.load("accuracy")


def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

checkpoint = None
last_checkpoint = get_last_checkpoint(
    training_args.output_dir
)  # 만약 output_dir에 checkpoint가 남아있으면 이를 사용하고, 없으면 None이 return됩니다.
if (
    training_args.resume_from_checkpoint is not None
):  # output_dir이 아닌 다른 위치에서의 checkpoint를 resume_from_checkpoint로 지정할 수 있습니다.
    checkpoint = training_args.resume_from_checkpoint
else:  # 아니면 last_checkpoint로 checkpoint를 지정합니다.
    checkpoint = last_checkpoint

train_result = trainer.train(resume_from_checkpoint=checkpoint)

trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
