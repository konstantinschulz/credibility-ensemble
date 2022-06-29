import os

import torch
from transformers import DistilBertTokenizerFast, BertTokenizerFast, BertModel, ReformerTokenizer, ReformerTokenizerFast


class Config:
    alpaca_docker_port: int = 8000
    alpaca_docker_image_name: str = "konstantinschulz/credibility-score-service:v2"
    cfc_docker_port: int = 8000
    cfc_docker_base_url: str = f"http://localhost:{cfc_docker_port}/credibility"
    cfc_docker_image_name: str = "konstantinschulz/covid19-fact-checking:v1"
    credibility_ensemble_dir: str = os.path.abspath(".")
    cs_docker_image_name: str = "konstantinschulz/credibilityscore:v1"
    cs_docker_port: int = 8000
    cs_docker_base_url: str = f"http://localhost:{cs_docker_port}/credibility"
    data_dir: str = os.path.join(credibility_ensemble_dir, "data")
    dataset_all_path: str = os.path.join(data_dir, "dataset_all.txt")
    dataset_balanced_path: str = os.path.join(data_dir, "dataset_balanced.txt")
    dataset_cfc_path: str = os.path.join(data_dir, "dataset_cfc.txt")
    dataset_cs_path: str = os.path.join(data_dir, "dataset_cs.txt")
    dataset_src_path: str = os.path.join(data_dir, "fake_new_dataset.xlsx")
    dataset_target_path: str = os.path.join(data_dir, "dataset_final.txt")
    dataset_website_credibility: str = os.path.join(data_dir, "dataset_fake_news_corpus.txt")
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim_alpaca: int = 25
    dim_cfc: int = 1
    dim_cs: int = 22
    dim_ensemble: int = 48
    dropout_value: float = 0.1
    max_length: int = 1024  # 512
    model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"  # 'distilbert-base-uncased'
    # mini_lm: BertModel = BertModel.from_pretrained(model_name).to(device)
    # tokenizer: DistilBertTokenizerFast = DistilBertTokenizerFast.from_pretrained(model_name)
    models_dir: str = os.path.join(credibility_ensemble_dir, "models")
    model_path_alpaca: str = os.path.join(models_dir, "alpaca_classifier.pt")
    model_path_ensemble: str = os.path.join(models_dir, "ensemble_classifier.pt")
    model_path_most_frequent: str = os.path.join(models_dir, "most_frequent_classifier.pt")
    model_path_cfc: str = os.path.join(models_dir, "cfc_classifier.pt")
    model_path_cs: str = os.path.join(models_dir, "cs_classifier.pt")
    num_labels: int = 2
    tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(model_name)
