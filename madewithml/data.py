import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from madewithml.config import STOPWORDS


def load_data(dataset_loc: str, num_samples: int = None, seed: int = 1234) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame."""
    df = pd.read_csv(dataset_loc)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if num_samples:
        df = df.head(num_samples)
    return df


def stratify_split(
    df: pd.DataFrame,
    stratify: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by class while keeping tiny classes in train."""
    train_dfs, test_dfs = [], []
    for _, group in df.groupby(stratify):
        if len(group) < 2:
            train_dfs.append(group)
            continue
        train_group, test_group = train_test_split(
            group,
            test_size=test_size,
            shuffle=shuffle,
            random_state=seed,
        )
        train_dfs.append(train_group)
        test_dfs.append(test_group)

    train_df = pd.concat(train_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    if test_dfs:
        test_df = pd.concat(test_dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        test_df = train_df.copy()
    return train_df, test_df


def clean_text(text: str, stopwords: List = STOPWORDS) -> str:
    """Clean raw text string."""
    text = str(text).lower()
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text)
    text = re.sub(" +", " ", text)
    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    return text


def tokenize(df: pd.DataFrame) -> Dict:
    """Tokenize text inputs with SciBERT."""
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(df["text"].tolist(), return_tensors="np", padding="longest", truncation=True)
    outputs = {
        "ids": encoded_inputs["input_ids"],
        "masks": encoded_inputs["attention_mask"],
    }
    if "tag" in df:
        outputs["targets"] = np.array(df["tag"])
    return outputs


def preprocess(df: pd.DataFrame, class_to_index: Dict) -> Dict:
    """Preprocess a raw DataFrame into model inputs."""
    df = df.copy()
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["text"] = df.title + " " + df.description
    df["text"] = df.text.apply(clean_text)
    if "tag" in df and class_to_index:
        df["tag"] = df["tag"].map(class_to_index)
    return tokenize(df)


class CustomPreprocessor:
    """Custom pandas preprocessor."""

    def __init__(self, class_to_index=None):
        self.class_to_index = class_to_index or {}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

    def fit(self, df: pd.DataFrame):
        tags = sorted(df["tag"].dropna().unique().tolist())
        self.class_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        return self

    def transform(self, df: pd.DataFrame):
        return preprocess(df, class_to_index=self.class_to_index)
