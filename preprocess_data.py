import os
import json
from typing import Dict, List, Generator
import numpy
import torch
from pandas import read_excel, DataFrame
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BatchEncoding, TensorType
from config import Config


class DataItem:
    def __init__(self, label: int = 0, cfc: float = 0, alpaca: Dict[str, float] = None, text: str = "",
                 cs: Dict[str, float] = None) -> None:
        self.alpaca: Dict[str, float] = dict() if alpaca is None else alpaca
        self.cfc: float = cfc
        self.cs: Dict[str, float] = dict() if cs is None else cs
        self.label: int = label
        self.text: str = text

    def serialize(self) -> str:
        return json.dumps(self.__dict__) + "\n"


class CovidFakeNewsDataset(Dataset):
    def __init__(self, indices: List[int] = None, path: str = Config.dataset_balanced_path) -> None:
        self.dataset_path: str = path
        self.positive_label_indices: List[int] = []
        dataset_size: int = 0
        for idx, di in enumerate(self.__iter__()):
            if di.label == 1:
                self.positive_label_indices.append(idx)
            dataset_size += 1
        self.indices: List[int] = list(range(dataset_size)) if indices is None else indices

    def __getitem__(self, idx: int) -> BatchEncoding:
        target_idx: int = self.indices[idx]
        with open(self.dataset_path) as f:
            for i, line in enumerate(f.readlines()):
                if i == target_idx:
                    line_dict: dict = json.loads(line)
                    di: DataItem = DataItem(**line_dict)
                    encodings: BatchEncoding = Config.tokenizer(
                        di.text, truncation=True, padding="max_length", max_length=Config.max_length,
                        return_tensors=TensorType.PYTORCH)
                    for k, v in encodings.items():
                        encodings[k] = v.squeeze().to(Config.device)
                    target: torch.Tensor = torch.zeros(2, device=Config.device)
                    target[di.label] = 1
                    encodings["labels"] = target
                    # adapt CFC scores to our policy of 1 = credible, 0 = fake
                    encodings["cfc"] = torch.tensor(1 - di.cfc, device=Config.device)
                    encodings["alpaca"] = torch.tensor(list(di.alpaca.values()), device=Config.device)
                    encodings["cs"] = torch.tensor(list(di.cs.values()), device=Config.device)
                    return encodings

    def __iter__(self) -> Generator[DataItem, None, None]:
        with open(self.dataset_path) as f:
            for line in f.readlines():
                if len(line) > 1:
                    line_dict = json.loads(line)
                    yield DataItem(**line_dict)

    def __len__(self) -> int:
        return len(self.indices)


def create_dataset():
    df: DataFrame = read_excel(Config.dataset_src_path)
    # remove NaN values
    df = df.replace({numpy.nan: ""})
    lines: list[str] = []
    for idx, series in df.iterrows():
        text: str = ". ".join([series["title"], series["text"]])
        text = text.replace("\n", " ")
        subcategory: str = series["subcategory"]
        label: int = 1 if subcategory == "true" else 0
        lines.append(f"{label} | {text}\n")
    with open(Config.dataset_target_path, "w+") as f:
        f.writelines(lines)


def resample_dataset(src_path: str):
    dataset: CovidFakeNewsDataset = CovidFakeNewsDataset(path=src_path)
    neg_count: int = 0
    dis: List[DataItem] = []
    for di in tqdm(dataset):
        if di.label == 0:
            neg_count += 1
            dis.append(di)
        elif len(dis) < neg_count * 2:
            dis.append(di)
    with open(Config.dataset_balanced_path, "a+") as f:
        for di in dis:
            f.write(di.serialize())


# resample_dataset(Config.dataset_cs_path)
# print(numpy.mean([float(x.alpaca["all"]) for x in CovidFakeNewsDataset()]))
# print(numpy.mean([float(x.cfc) for x in CovidFakeNewsDataset()]))
# print(numpy.mean([float(x.cs["credibility_score_weighted"]) for x in CovidFakeNewsDataset()]))
# print(min([x.alpaca["all"] for x in CovidFakeNewsDataset(path=Config.dataset_all_path)]))
