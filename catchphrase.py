#!/usr/bin/env python3
import ast
import dataclasses
import pathlib
import requests
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import transformers
from bs4 import BeautifulSoup
from sentence_transformers import models, SentenceTransformer
from sklearn import manifold
from tqdm import trange

transformers.BertTokenizer = transformers.BertJapaneseTokenizer


def calc_cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    unit_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.matmul(unit_vectors, unit_vectors.T)


@dataclasses.dataclass
class Catchphrase:
    phrases_csv_path: pathlib.Path = pathlib.Path("./input/catchphrases.csv")
    embedded_phrases_csv_path: pathlib.Path = pathlib.Path(
        "./input/catchphrases_embedded.csv"
    )
    cols: List[str] = dataclasses.field(
        default_factory=lambda: [
            "キャッチコピー",
            "カテゴリ",
            "印象",
            "ターゲット",
            "埋め込み済キャッチコピー",
        ]
    )
    origin_df: pd.DataFrame = dataclasses.field(init=False)
    df: pd.DataFrame = dataclasses.field(init=False)
    model: SentenceTransformer = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """日本語学習済BERTモデルの読み込み＆csv読み込み。
        csv読み込みについては、キャッシュファイルの有無に応じて以下の挙動をとる。
        1. スクレイピング未実施 -> `self.origin_df`, `self.df` ともに空のdataframe
        2. スクレイピング実施済、embedding未実施 -> `self.origin_df` はcsv読み込み、`self.df` は空のdataframe
        3. スクレイピング・embeddingともに実施済 -> `self.origin_df`, `self.df` ともにcsv読み込み
        """
        self.model = self._create_pretrained_bert()

        if self.embedded_phrases_csv_path.exists():
            self._read_embedded_phrases_csv()
        else:
            if self.phrases_csv_path.exists():
                self._read_phrases_csv()
            else:
                self.origin_df = self.df = pd.DataFrame(columns=self.cols)

    def _create_pretrained_bert(self) -> SentenceTransformer:
        transformer = models.Transformer(
            "cl-tohoku/bert-base-japanese-whole-word-masking"
        )
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        model = SentenceTransformer(modules=[transformer, pooling])
        return model

    def _read_phrases_csv(self) -> None:
        print(f"Reading '{self.phrases_csv_path}'", end="")
        self.origin_df = pd.read_csv(self.phrases_csv_path, index_col=0)
        print(" -> Done")
        self.df = pd.DataFrame(columns=self.cols)

    def _read_embedded_phrases_csv(self) -> None:
        print(f"Reading '{self.embedded_phrases_csv_path}'", end="")
        self.df = pd.read_csv(self.embedded_phrases_csv_path, index_col=0)
        self.df[self.cols[-1]] = self.df[self.cols[-1]].map(
            lambda x: np.array(ast.literal_eval(x))
        )
        print(" -> Done")
        self.origin_df = self.df[self.cols[:-1]]

    def crawl(self) -> None:
        TOP_PAGE = "https://catchcopy.make1.jp"
        res = requests.get(TOP_PAGE)
        soup = BeautifulSoup(res.content, "html.parser")

        div_page_l_html_texts = soup.find_all("div", class_="page_l")
        str_pages = [
            tag_text.get_text().split(" ") for tag_text in div_page_l_html_texts
        ][0]
        pages = []
        for str_page in str_pages:
            try:
                page = int(str_page)
            except ValueError:
                pass
            else:
                pages.append(page)
        n_pages = np.array(pages).max()

        dict_for_df = {col: [] for col in self.cols[:-1]}
        for i in trange(n_pages, desc=f"Crawl '{TOP_PAGE}'"):
            target_url = (
                f"{TOP_PAGE}/index.cgi?start={i * 30 + 1}&end={(i + 1) * 30}&{i + 1}"
            )
            res = requests.get(target_url)
            soup = BeautifulSoup(res.content, "html.parser")

            div_catch_texts = soup.find_all("div", class_="catch")
            dict_for_df[self.cols[0]] += [
                # div class='catch' の中に span class='t1' などがネストされているため除外する
                html_text.get_text().split("\n")[0]
                for html_text in div_catch_texts
            ]

            span_t1_texts = soup.find_all("span", class_="t1")
            dict_for_df[self.cols[1]] += [
                html_text.get_text() for html_text in span_t1_texts
            ]
            span_t2_texts = soup.find_all("span", class_="t2")
            dict_for_df[self.cols[2]] += [
                html_text.get_text() for html_text in span_t2_texts
            ]
            span_t3_texts = soup.find_all("span", class_="t3")
            dict_for_df[self.cols[3]] += [
                html_text.get_text() for html_text in span_t3_texts
            ]

        self.origin_df = pd.DataFrame(dict_for_df)
        self.origin_df.to_csv(self.phrases_csv_path)

    def crawl_embed(self) -> None:
        if len(self.origin_df) == 0:
            self.crawl()
        self.df = self.origin_df.copy()

        embedded_phrases = self.embed(self.origin_df[self.cols[0]])

        # そのまま to_csv すると読み込みしづらい形でエクスポートされてしまうので、
        # いったんlist形式に変換して to_csv する
        self.df[self.cols[-1]] = [x.tolist() for x in embedded_phrases]
        self.df.to_csv(self.embedded_phrases_csv_path)

        self.df[self.cols[-1]] = [np.array(vec) for vec in embedded_phrases]

    def embed(self, phrases: Union[List[str], pd.Series]) -> np.ndarray:
        return self.model.encode(phrases)

    def _filter_phrases(
        self,
        max_phrase_length: int = None,
        target_categories: List[str] = None,
        target_atmospheres: List[str] = None,
        target_users: List[str] = None,
    ) -> pd.DataFrame:
        df = self.df.copy()
        if max_phrase_length:
            df = df[df[self.cols[0]].str.len() <= max_phrase_length]
        if target_categories:
            df = df[df[self.cols[1]].isin(target_categories)]
        if target_atmospheres:
            df = df[df[self.cols[2]].isin(target_atmospheres)]
        if target_users:
            df = df[df[self.cols[3]].isin(target_users)]
        assert len(df) > 0
        return df

    def _calc_cosine_similarity_matrix(
        self,
        phrase: str,
        df: pd.DataFrame,
    ) -> np.ndarray:
        embedded_phrase = self.embed([phrase])
        reference_embedded_phrases = df[self.cols[-1]]
        vectors = np.array(
            embedded_phrase.tolist() + reference_embedded_phrases.tolist()
        )
        cos_sim_matrix = calc_cosine_similarity_matrix(vectors)
        return cos_sim_matrix

    def calc_similarity_to_phrases(
        self,
        phrase: str,
        max_phrase_length: int = None,
        target_categories: List[str] = None,
        target_atmospheres: List[str] = None,
        target_users: List[str] = None,
    ) -> pd.DataFrame:
        df = self._filter_phrases(
            max_phrase_length, target_categories, target_atmospheres, target_users
        )

        embedded_phrase = self.embed([phrase])
        reference_embedded_phrases = df[self.cols[-1]]
        vectors = np.array(
            embedded_phrase.tolist() + reference_embedded_phrases.tolist()
        )
        cos_sim_matrix = calc_cosine_similarity_matrix(vectors)

        cos_sim_df = pd.DataFrame(
            cos_sim_matrix,
            index=[phrase] + reference_embedded_phrases.index.tolist(),
            columns=[phrase] + reference_embedded_phrases.index.tolist(),
        )
        similarities = cos_sim_df[phrase].drop(index=phrase)
        similarities.name = "類似度"
        return pd.merge(
            df[self.cols[:-1]], similarities, left_index=True, right_index=True
        )

    def calc_mds(
        self,
        phrase: str,
        max_phrase_length: int = None,
        target_categories: List[str] = None,
        target_atmospheres: List[str] = None,
        target_users: List[str] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        df = self._filter_phrases(
            max_phrase_length, target_categories, target_atmospheres, target_users
        )

        embedded_phrase = self.embed([phrase])
        reference_embedded_phrases = df[self.cols[-1]]
        vectors = np.array(
            embedded_phrase.tolist() + reference_embedded_phrases.tolist()
        )
        cos_sim_matrix = calc_cosine_similarity_matrix(vectors)

        mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
        positions = mds.fit_transform(cos_sim_matrix)
        labels = [phrase] + df[self.cols[0]].tolist()
        return positions, labels


if __name__ == "__main__":
    catchphrase = Catchphrase()
    catchphrase.crawl_embed()
