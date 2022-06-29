import os
import copy
from typing import List, Dict, Tuple, Union
import numpy as np
from datasets import load_metric, Metric
from torch.nn import Parameter
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset, Dataset
from tqdm import tqdm
import torch
from transformers import Trainer, TrainingArguments, \
    IntervalStrategy, EvalPrediction, ReformerForSequenceClassification
from transformers.integrations import TensorBoardCallback

from classes import CredibilityClassifierRandom, Mode, ModeItem, CredibilityTrainer, CredibilityClassifierBase
from config import Config
from preprocess_data import CovidFakeNewsDataset

print(torch.cuda.is_available())
dataset: CovidFakeNewsDataset = CovidFakeNewsDataset()
# Creating data indices for training and validation splits:
dataset_size: int = len(dataset)
indices: List[int] = list(range(dataset_size))
split: int = int(np.floor(0.1 * dataset_size))
np.random.seed(42)
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# Creating PT data samplers and loaders:
train_sampler: SubsetRandomSampler = SubsetRandomSampler(train_indices)
valid_sampler: SubsetRandomSampler = SubsetRandomSampler(val_indices)
batch_size: int = 1  # 8
eval_steps: int = 16  # 300
train_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader: DataLoader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
metric: Metric = load_metric("accuracy")


def compute_metrics(ep: EvalPrediction) -> Dict[str, float]:
    logits, labels = ep
    if len(logits) != len(labels):
        print("MISMATCH")
    predictions: np.ndarray = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=np.argmax(labels, axis=-1)[:len(predictions)])


def evaluate(mi: ModeItem):
    model: CredibilityClassifierBase = mi.get_model()
    # model: LongformerForSequenceClassification = LongformerForSequenceClassification.from_pretrained(
    #     "allenai/longformer-base-4096").to(Config.device)
    model.eval()
    with torch.no_grad():
        y_true: List[int] = []
        y_pred: List[int] = []
        too_long: int = 0
        for val_batch in tqdm(val_loader):
            if bool(val_batch['input_ids'][0][-1] != 0):
                too_long += 1
            # val_logits, val_labels = model(**val_batch)
            # y_true += list(torch.argmax(val_labels, dim=1).cpu().numpy())
            # y_pred += list(torch.argmax(val_logits, dim=1).cpu().numpy())
            # y_pred += [1] * len(val_labels)
        # print(classification_report(y_true, y_pred))
        print(too_long, " too long")


def train(mode_item: ModeItem, eval_only: bool = False):
    model: CredibilityClassifierBase = mode_item.get_model(eval_only)
    train_args: TrainingArguments = TrainingArguments(
        output_dir=".", dataloader_pin_memory=False, logging_steps=eval_steps, logging_strategy=IntervalStrategy.STEPS,
        evaluation_strategy=IntervalStrategy.STEPS, eval_steps=eval_steps, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8, num_train_epochs=1, save_strategy=IntervalStrategy.NO, learning_rate=4e-3,
        adafactor=True, gradient_accumulation_steps=16, seed=42)  # 4e-2
    val_dataset: Dataset = Subset(dataset, val_indices)
    trainer: Trainer = CredibilityTrainer(
        model=model, args=train_args, train_dataset=Subset(dataset, train_indices), callbacks=[TensorBoardCallback],
        eval_dataset=val_dataset, compute_metrics=compute_metrics)
    if eval_only:
        trainer.evaluate(eval_dataset=val_dataset)
    else:
        old_params: Tuple[Parameter, Parameter] = \
            (copy.deepcopy(model.alpaca_classifier.weight), copy.deepcopy(model.alpaca_classifier.bias))
        trainer.train()
        print("OLD: ", old_params)
        print("NEW: ", (model.alpaca_classifier.weight, model.alpaca_classifier.bias))
    torch.save(model.state_dict(), mode_item.path)


def train_large():
    torch.autograd.set_detect_anomaly(True)
    # model: LongformerForSequenceClassification = LongformerForSequenceClassification.from_pretrained(
    #     "allenai/longformer-base-4096").to(Config.device)
    model: ReformerForSequenceClassification = ReformerForSequenceClassification.from_pretrained(
        "google/reformer-enwik8").to(Config.device)
    train_args: TrainingArguments = TrainingArguments(
        output_dir=".", dataloader_pin_memory=False, logging_steps=50, logging_strategy=IntervalStrategy.STEPS,
        evaluation_strategy=IntervalStrategy.STEPS, eval_steps=400, per_device_train_batch_size=1,
        per_device_eval_batch_size=1, num_train_epochs=1)  # , learning_rate=4e-2
    trainer: Trainer = Trainer(  # CredibilityTrainer
        model=model, args=train_args, train_dataset=Subset(dataset, train_indices), callbacks=[TensorBoardCallback],
        eval_dataset=Subset(dataset, val_indices), compute_metrics=compute_metrics)
    trainer.train()
    # model.train()
    # eval_steps_interval: int = int(len(train_loader) / 5)
    # writer: SummaryWriter = SummaryWriter()
    # for idx, batch in tqdm(enumerate(train_loader)):
    #     input_ids: torch.Tensor = batch["input_ids"]
    #     global_attention_mask: torch.Tensor = torch.zeros(input_ids.shape, dtype=torch.long, device=Config.device)
    #     global_attention_mask[0][0] = 1
    #     output: LongformerSequenceClassifierOutput = model(**batch, global_attention_mask=global_attention_mask)
    #     writer.add_scalar('Loss/train', output.loss.data.item(), idx)
    #     if idx % eval_steps_interval == 0:
    #         model.eval()
    #         with torch.no_grad():
    #             for val_batch in tqdm(val_loader):
    #                 input_ids: torch.Tensor = val_batch["input_ids"]
    #                 global_attention_mask: torch.Tensor = torch.zeros(input_ids.shape, dtype=torch.long,
    #                                                                   device=Config.device)
    #                 global_attention_mask[0][0] = 1
    #                 val_output: LongformerSequenceClassifierOutput = model(**val_batch,
    #                                                                        global_attention_mask=global_attention_mask)
    #                 writer.add_scalar('Loss/validation', val_output.loss.data.item(), idx)
    #         model.train()


# train_large()
mode: Union[Mode, ModeItem] = Mode.ensemble
train(mode_item=mode.value, eval_only=False)
# train_transformers()
# evaluate()
# model: CredibilityClassifierBaseline = CredibilityClassifierBaseline().to(Config.device)
# model.load_state_dict(torch.load(Config.baseline_model_path))
