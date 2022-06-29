import random
from enum import Enum
from typing import Tuple, Type, List

import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from transformers import BertModel, AutoModelForSequenceClassification, Trainer, TrainerState, TrainingArguments
from config import Config


class CredibilityClassifierBase(torch.nn.Module):
    def __init__(self, eval_only: bool = False):
        super().__init__()
        # self.alpaca_classifier: torch.nn.Linear = torch.nn.Linear(dim, Config.num_labels, device=Config.device)
        self.alpaca_classifier: torch.nn.Linear = torch.nn.Linear(256, Config.num_labels, device=Config.device)
        self.alpaca_intermediate: torch.nn.Linear = torch.nn.Linear(Config.dim_alpaca, 256, device=Config.device)
        self.cfc_classifier: torch.nn.Linear = torch.nn.Linear(1024, Config.num_labels, device=Config.device)
        # self.cfc_classifier: torch.nn.Linear = torch.nn.Linear(dim + 1, Config.num_labels, device=Config.device)
        self.cfc_intermediate: torch.nn.Linear = torch.nn.Linear(Config.dim_cfc + 1, 1024, device=Config.device)
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(Config.dropout_value)
        self.ensemble_classifier: torch.nn.Linear = torch.nn.Linear(48, Config.num_labels, device=Config.device)
        self.ensemble_intermediate: torch.nn.Linear = torch.nn.Linear(Config.dim_ensemble, 48, device=Config.device)
        self.ensemble_lstm = torch.nn.LSTM(input_size=Config.dim_ensemble, hidden_size=128, num_layers=1,
                                           dropout=Config.dropout_value, proj_size=Config.num_labels)
        self.eval_only: bool = eval_only
        self.relu: torch.nn.LeakyReLU = torch.nn.LeakyReLU()

    def get_alpaca(self, kwargs: dict) -> torch.Tensor:
        alpaca: torch.Tensor = kwargs["alpaca"]
        if self.eval_only:
            overall_scores: torch.Tensor = alpaca[:, -1]
            ret_val: torch.Tensor = torch.zeros((overall_scores.shape[0], Config.num_labels), device=Config.device)
            for i in range(len(overall_scores)):
                # centering of values around 0.5 (mean was: 0.62)
                os: float = float(overall_scores[i]) - 0.12
                ret_val[i] = torch.tensor([1 - os, os], device=Config.device)
            return ret_val
        else:
            # alpaca = self.dropout(alpaca)
            # alpaca = self.intermediate(alpaca)
            # alpaca = self.relu(alpaca)
            # return self.alpaca_classifier(alpaca)
            return alpaca

    def get_cfc(self, kwargs: dict) -> torch.Tensor:
        cfc: torch.Tensor = kwargs["cfc"].unsqueeze(dim=1)
        new_cfc: torch.Tensor = torch.zeros((cfc.shape[0], Config.num_labels), device=Config.device)
        if self.eval_only:
            for i in range(len(cfc)):
                # centering of values around 0.5 (mean was: 0.66)
                current_cfc: float = float(cfc[i]) - 0.16
                new_cfc[i] = torch.tensor([1 - current_cfc, current_cfc], device=Config.device)
            return new_cfc
        else:
            # new_cfc = self.cfc_intermediate(cfc)
            # new_cfc = self.dropout(new_cfc)
            # new_cfc = self.relu(new_cfc)
            # return self.cfc_classifier(new_cfc)
            return cfc

    def get_combined(self, kwargs: dict) -> torch.Tensor:
        cs: torch.Tensor = self.get_cs(kwargs)
        cfc: torch.Tensor = self.get_cfc(kwargs)
        alpaca: torch.Tensor = self.get_alpaca(kwargs)
        return torch.cat([cs, cfc, alpaca], dim=1)

    def get_cs(self, kwargs: dict) -> torch.Tensor:
        cs: torch.Tensor = kwargs["cs"]
        if self.eval_only:
            overall_scores: torch.Tensor = cs[:, -1]
            ret_val: torch.Tensor = torch.zeros((overall_scores.shape[0], Config.num_labels), device=Config.device)
            for i in range(len(overall_scores)):
                # centering of values around 0.5 (mean was: 0.51)
                os: float = float(overall_scores[i]) - 0.01
                ret_val[i] = torch.tensor([1 - os, os], device=Config.device)
            return ret_val
        else:
            # cs = self.dropout(cs)
            # cs = self.intermediate(cs)
            # return self.alpaca_classifier(cs)
            return cs

    def get_random(self, labels: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = torch.zeros(labels.shape).to(Config.device)
        for i in range(len(logits)):
            target_idx: int = random.choice([0, 1])
            logits[i][target_idx] = 1
        return logits


class CredibilityClassifierAlpaca(CredibilityClassifierBase):
    def __init__(self, eval_only: bool = False):
        super().__init__(eval_only)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_alpaca(kwargs), labels


class CredibilityClassifierCfc(CredibilityClassifierBase):
    def __init__(self, eval_only: bool = False):
        super().__init__(eval_only)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_cfc(kwargs), labels


class CredibilityClassifierCs(CredibilityClassifierBase):
    def __init__(self, eval_only: bool = False):
        super().__init__(eval_only)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_cs(kwargs), labels


class CredibilityClassifierEnsemble(CredibilityClassifierBase):
    def __init__(self, eval_only: bool = False):
        super().__init__(eval_only)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        combined: torch.Tensor = self.get_combined(kwargs)
        # combined = self.dropout(combined)
        # combined = self.ensemble_intermediate(combined)
        # combined = self.dropout(combined)
        # combined = self.relu(combined)
        # combined = self.ensemble_classifier(combined)
        ps: PackedSequence = pack_sequence([combined], enforce_sorted=False)
        result: Tuple[PackedSequence, tuple] = self.ensemble_lstm(ps)
        return result[0][0], labels


class CredibilityClassifierRandom(CredibilityClassifierBase):
    def __init__(self, eval_only: bool = False):
        super().__init__(eval_only)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_random(labels), labels


class CredibilityTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(CredibilityTrainer, self).__init__(*args, **kwargs)
        self.loss_fn: torch.nn.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

    def compute_loss(self, model: CredibilityClassifierBase, inputs: dict, return_outputs: bool = False):
        logits, labels = model(**inputs)
        loss: torch.Tensor = self.loss_fn(logits, labels)
        return (loss, (loss, logits)) if return_outputs else loss
        # return (loss, logits) if return_outputs else loss


class ModeItem:
    def __init__(self, model_class: Type[CredibilityClassifierBase] = CredibilityClassifierBase, path: str = ""):
        self.model_class: Type[CredibilityClassifierBase] = model_class
        self.path: str = path

    def get_model(self, eval_only: bool = False) -> CredibilityClassifierBase:
        return self.model_class(eval_only).to(Config.device)


class Mode(Enum):
    alpaca: ModeItem = ModeItem(model_class=CredibilityClassifierAlpaca, path=Config.model_path_alpaca)
    cfc: ModeItem = ModeItem(model_class=CredibilityClassifierCfc, path=Config.model_path_cfc)
    cs: ModeItem = ModeItem(model_class=CredibilityClassifierCs, path=Config.model_path_cs)
    ensemble: ModeItem = ModeItem(model_class=CredibilityClassifierEnsemble, path=Config.model_path_ensemble)
    random: ModeItem = ModeItem(model_class=CredibilityClassifierRandom, path=Config.model_path_most_frequent)
