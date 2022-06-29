import csv
import fileinput
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from multiprocessing.pool import ApplyResult
from typing import List, Dict, Any, Tuple, Set, Generator
from urllib.parse import urlparse

import docker
import pandas as pd
import requests
from docker import DockerClient
from docker.models.containers import Container
from elg import Service
from elg.model import ClassificationResponse
from pandas import DataFrame
from tqdm import tqdm

from config import Config
from preprocess_data import CovidFakeNewsDataset, DataItem
import nltk


class FakeNewsCorpusItem:
    def __init__(self, alpaca: Dict[str, float] = None, content: str = "", domain: str = "", label: int = 0,
                 url: str = ""):
        self.alpaca: Dict[str, float] = dict() if alpaca is None else alpaca
        self.content: str = content
        self.domain: str = domain
        self.label: int = label
        self.url: str = url


class FakeNewsCorpus:
    def __init__(self):
        self.size: int = sum(1 for x in self.__iter__())

    def __iter__(self) -> Generator[FakeNewsCorpusItem, None, None]:
        with open(Config.dataset_website_credibility) as f:
            for line in f.readlines():
                if not line.strip():
                    continue
                line_dict: dict = json.loads(line[:-1])
                yield FakeNewsCorpusItem(**line_dict)

    def __len__(self) -> int:
        return self.size


def calculate_alpaca(path: str) -> None:
    client, container, service = start_docker_container(Config.alpaca_docker_port, Config.alpaca_docker_image_name, 1)
    dataset: CovidFakeNewsDataset = CovidFakeNewsDataset()
    with open(path, "a+") as f:
        try:
            item: DataItem
            for item in tqdm(dataset):
                item.alpaca = get_alpaca_scores(item.text, service)
                f.write(item.serialize())
        finally:
            shutdown(container, client)


def calculate_cfc() -> None:
    client, container, service = start_docker_container(Config.cfc_docker_port, Config.cfc_docker_image_name, 2)
    dataset: CovidFakeNewsDataset = CovidFakeNewsDataset()
    with open(Config.dataset_cfc_path, "a+") as f:
        try:
            for text, label in tqdm(dataset):
                cfc_score: float = get_cfc_score(text)
                f.write(" | ".join([str(label), str(cfc_score), text]))
        finally:
            shutdown(container, client)


def calculate_cs():
    client, container, service = start_docker_container(Config.cs_docker_port, Config.cs_docker_image_name, 1)
    dataset: CovidFakeNewsDataset = CovidFakeNewsDataset(path=Config.dataset_all_path)
    with open(Config.dataset_cs_path, "a+") as f:
        try:
            done: List[str] = [x.text for x in list(CovidFakeNewsDataset(path=Config.dataset_cs_path))]
            items: List[DataItem] = list(dataset)  # [:50]
            items = [x for x in items if x.text not in done]
            with multiprocessing.Pool(os.cpu_count() + 1) as pool:
                ars: List[ApplyResult] = [pool.apply_async(get_cs_scores, (x,)) for x in items]
                for ar in tqdm(ars):
                    item: DataItem = ar.get()
                    f.write(item.serialize())
        finally:
            shutdown(container, client)


def calculate_website_credibility_fnn(path: str) -> None:
    # client, container, service = start_docker_container(Config.alpaca_docker_port, Config.alpaca_docker_image_name, 1)
    fnn_dir: str = os.path.join(Config.data_dir, "FakeNewsNet")
    dataset_dir: str = os.path.join(fnn_dir, "dataset")
    domain_dict: Dict[str, List[int]] = dict()
    for file in [x for x in os.listdir(dataset_dir) if x.endswith(".csv")]:
        is_fake: bool = "fake" in file
        df: DataFrame = pd.read_csv(os.path.join(dataset_dir, file))
        for idx, series in tqdm(df.iterrows()):
            url: str = series["news_url"]
            if pd.isna(url):
                continue
            base_url: str = url.split("/")[0] if not url.startswith("http") else urlparse(url).netloc
            if base_url not in domain_dict:
                domain_dict[base_url] = []
            domain_dict[base_url].append(0 if is_fake else 1)
    domain_avg_dict: Dict[str, float] = dict()
    for domain in domain_dict:
        size: int = len(domain_dict[domain])
        if size < 30:
            continue
        avg: float = sum(domain_dict[domain]) / size
        domain_avg_dict[domain] = avg
    domain_avg_sorted: List[Tuple[str, float]] = sorted(domain_avg_dict.items(), key=lambda x: x[1], reverse=True)
    a = 0
    # with open(path, "a+") as f:
    #     try:
    #         item: DataItem
    #         for item in tqdm(dataset):
    #             item.alpaca = get_alpaca_scores(item.text, service)
    #             f.write(item.serialize())
    #     finally:
    #         shutdown(container, client)


def calculate_website_credibility_fnc():
    client, container, service = start_docker_container(Config.alpaca_docker_port, Config.alpaca_docker_image_name, 1)
    try:
        for idx, line in tqdm(enumerate(fileinput.FileInput(Config.dataset_website_credibility, inplace=1))):
            if not line:
                continue
            json_dict: dict = json.loads(line)
            fnci: FakeNewsCorpusItem = FakeNewsCorpusItem(**json_dict)
            if not fnci.alpaca:
                fnci.alpaca = get_alpaca_scores(fnci.content, service)
            # print to StdOut to replace the line in the file; not printing anything means deleting the line
            print(json.dumps(fnci.__dict__))
    finally:
        shutdown(container, client)
    # print(len(fnc))
    # a = set(x.domain for x in fnc)
    # print(sum(x.label for x in fnc))


def get_alpaca_scores(text: str, service: Service) -> Dict[str, float]:
    response: Any = service(text, sync_mode=True, verbose=False)
    cr: ClassificationResponse = response
    return {x.class_field: x.score for x in cr.classes}


def get_average_website_credibility():
    fnc: FakeNewsCorpus = FakeNewsCorpus()
    domain_set: Set[str] = set(fnci.domain for fnci in fnc)
    domain2alpaca: Dict[str, List[float]] = {domain: [] for domain in domain_set}
    domain2credibility: Dict[str, List[int]] = {domain: [] for domain in domain_set}
    for fnci in fnc:
        domain2alpaca[fnci.domain].append(fnci.alpaca["all"])
        domain2credibility[fnci.domain].append(fnci.label)
    aggregate: List[Tuple[str, float, float]] = []
    for domain in domain_set:
        if len(domain2credibility[domain]) + len(domain2alpaca[domain]) < 20:
            continue
        cred_avg: float = sum(domain2credibility[domain]) / len(domain2credibility[domain])
        alpaca_avg: float = sum(domain2alpaca[domain]) / len(domain2alpaca[domain])
        aggregate.append((domain, cred_avg, alpaca_avg))
    aggregate.sort(key=lambda x: x[1])
    a = 0


def get_cfc_score(text: str) -> float:
    # response: requests.Response = requests.get(f"{Config.cfc_docker_base_url}?text={quote(text)}")
    sentences: List[str] = nltk.tokenize.sent_tokenize(text)
    response: requests.Response = requests.post(Config.cfc_docker_base_url, json=sentences)
    scores: List[float] = json.loads(response.text)
    return sum(scores) / len(scores)


def get_cs_scores(di: DataItem) -> DataItem:
    response: requests.Response = requests.post(Config.cs_docker_base_url, json=dict(text=di.text))
    try:
        di.cs = json.loads(response.text)
    except Exception as e:
        print(di.text, response.text)
        raise
    return di


def make_website_credibility_dataset():
    # increase maximum field size because the website content can be quite long
    csv.field_size_limit(sys.maxsize)
    file_path: str = os.path.join(Config.data_dir, "news_cleaned_2018_02_13.csv")
    with open(file_path) as f:
        reader: csv.reader = csv.reader(f)
        # first line of a CSV file contains header information
        column_list: List[str] = next(reader)
        # '', clickbait, political, bias, hate, conspiracy, satire, unknown, rumor, unreliable, junksci, fake, reliable
        dataset_balance: int = 0
        with open(Config.dataset_website_credibility, "a+") as target_file:
            for idx, row in tqdm(enumerate(reader)):
                if len(row) < 4:
                    continue
                if idx == 100000:
                    break
                label: int = 0
                if row[3] == "reliable":
                    label = 1
                    dataset_balance += 1
                if label or dataset_balance:
                    url: str = row[4]
                    base_url: str = url.split("/")[0] if not url.startswith("http") else urlparse(url).netloc
                    content: str = row[5]
                    fnci: FakeNewsCorpusItem = FakeNewsCorpusItem(None, content, base_url, label, url)
                    target_file.write(json.dumps(fnci.__dict__) + "\n")
                    dataset_balance += label - 1


def shutdown(container: Container, client: DockerClient):
    container.stop()
    container.remove()
    client.close()


def start_docker_container(port: int, image: str, sleep: int) -> Tuple[DockerClient, Container, Service]:
    client: DockerClient = docker.from_env()
    ports_dict: dict[int, int] = {port: port}
    container: Container = client.containers.run(image, ports=ports_dict, detach=True)
    time.sleep(sleep)
    service: Service = Service.from_docker_image(image, f"http://localhost:{port}/process", port)
    return client, container, service


# if __name__ == "__main__":
    # calculate_website_credibility_fnn(Config.dataset_website_credibility)
    # make_website_credibility_dataset()
    # calculate_website_credibility_fnc()
    # get_average_website_credibility()
# calculate_cfc()
# calculate_alpaca(Config.dataset_all_path)
# calculate_cs()
