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
    """[キャッチコピー集めました。](https://catchcopy.make1.jp/) からスクレイピングしてきた
    日本語キャッチコピー群を、事前学習済のSentence BERTでベクトル化する。
    ベクトル化されたキャッチコピー群を用いて、類似/非類似キャッチコピーの検索等を行うためのクラス。
    """

    cache_csv_dirpath: pathlib.Path = pathlib.Path("./input")
    cols: List[str] = dataclasses.field(
        default_factory=lambda: [
            "キャッチコピー",
            "カテゴリ",
            "印象",
            "ターゲット",
            "埋め込み済キャッチコピー",
        ]
    )
    phrases_csv_path: pathlib.Path = dataclasses.field(init=False)
    embedded_phrases_csv_path: pathlib.Path = dataclasses.field(init=False)
    origin_df: pd.DataFrame = dataclasses.field(init=False)
    df: pd.DataFrame = dataclasses.field(init=False)
    model: SentenceTransformer = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """事前学習済のSentence BERTモデルの読み込み ＆ キャッチコピーに関するdataframeの
        csvファイルからの読み込み。
        後者に関しては、各csvファイルの有無に応じて以下の挙動をとる。

        `self.embedded_phrases_csv_path` が存在（＝スクレイピング・ベクトル化ともに済）
        -> `self.origin_df`, `self.df` ともにcsv読み込み

        `self.embedded_phrases_csv_path` は存在せず、 `self.phrases_csv_path` のみ存在（＝スクレイピング済、ベクトル化未実施）
        -> `self.origin_df` はcsv読み込み、`self.df` は空のdataframe

        `self.embedded_phrases_csv_path` も `self.phrases_csv_path` も存在しない
        -> `self.origin_df`, `self.df` ともに空のdataframe
        """
        self.model = self._create_pretrained_bert()

        self.cache_csv_dirpath.mkdir(exist_ok=True)
        self.phrases_csv_path = self.cache_csv_dirpath / "catchphrases.csv"
        self.embedded_phrases_csv_path = (
            self.cache_csv_dirpath / "catchphrases_embedded.csv"
        )

        if self.embedded_phrases_csv_path.exists():
            self._read_embedded_phrases_csv()
        else:
            if self.phrases_csv_path.exists():
                self._read_phrases_csv()
            else:
                self.origin_df = self.df = pd.DataFrame(columns=self.cols)

    def _create_pretrained_bert(
        self,
        transformer_model_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking",
    ) -> SentenceTransformer:
        """事前学習済のSentence BERTモデルを返す。
        参考: [はじめての自然言語処理 第9回 Sentence BERT による類似文章検索の検証](https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part9.html)

        Parameters
        ----------
        transformer_model_name : str, optional
            , by default "cl-tohoku/bert-base-japanese-whole-word-masking"

        Returns
        -------
        SentenceTransformer
        """
        transformer = models.Transformer(transformer_model_name)
        pooling = models.Pooling(
            transformer.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        model = SentenceTransformer(modules=[transformer, pooling])
        return model

    def _read_phrases_csv(self) -> None:
        """スクレイピングでもってきたキャッチコピーのdataframeをcsvから読み取り、`self.origin_df` に格納"""
        print(f"Reading '{self.phrases_csv_path}'", end="")
        self.origin_df = pd.read_csv(self.phrases_csv_path, index_col=0)
        print(" -> Done")
        self.df = pd.DataFrame(columns=self.cols)

    def _read_embedded_phrases_csv(self) -> None:
        """スクレイピングでもってきて、さらにSentence BERTによるベクトル変換済のキャッチコピーの
        dataframeをcsvから読み取り、`self.df` に格納
        """
        print(f"Reading '{self.embedded_phrases_csv_path}'", end="")
        self.df = pd.read_csv(self.embedded_phrases_csv_path, index_col=0)
        self.df[self.cols[-1]] = self.df[self.cols[-1]].map(
            lambda x: np.array(ast.literal_eval(x))
        )
        print(" -> Done")
        self.origin_df = self.df[self.cols[:-1]]

    def crawl(self) -> None:
        """[キャッチコピー集めました。](https://catchcopy.make1.jp/) をスクレイピングして
        各キャッチコピーの情報をdataframeとして `self.origin_df` に格納。
        格納されたテーブルはキャッシュとしてcsv出力する。
        """
        TOP_PAGE = "https://catchcopy.make1.jp"
        res = requests.get(TOP_PAGE)
        soup = BeautifulSoup(res.content, "html.parser")

        # トップページ下部にある各ページへのリンクからページ数を把握
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
            # 1ページにつき30キャッチコピー+そのタグ情報が載っている
            target_url = (
                f"{TOP_PAGE}/index.cgi?start={i * 30 + 1}&end={(i + 1) * 30}&{i + 1}"
            )
            res = requests.get(target_url)
            soup = BeautifulSoup(res.content, "html.parser")

            div_catch_texts = soup.find_all("div", class_="catch")
            """
            `div_catch_texts` の中身の具体例（実際はスペースやタブはないが、見やすさのために追加）

            <div class="catch">
                <h3>走ろう。新しい自分が待っている。</h3>\n
                <p>
                    <span class="t1">
                        <a href="/index.cgi?start=1&end=30&1&category1=スポーツ">スポーツ</a>
                    </span>\n
                    <span class="t2">
                        <a href="/index.cgi?start=1&end=30&1&category2=語りかけ">語りかけ</a>
                    </span>\n
                    <span class="t3">
                        <a href="/index.cgi?start=1&end=30&1&category3=全般">全般</a>
                    </span>
                </p>\n
                <ul class="sns_btn2">
                    <li>（略）</li>\n
                    <li>（略）</li>\n
                    <li>（略）</li>
                </ul>\n
                <p class="bg">（略）</p>\n
            </div>
            """

            # キャッチコピー
            dict_for_df[self.cols[0]] += [
                html_text.get_text().split("\n")[0] for html_text in div_catch_texts
            ]
            # カテゴリタグ
            span_t1_texts = soup.find_all("span", class_="t1")
            dict_for_df[self.cols[1]] += [
                html_text.get_text() for html_text in span_t1_texts
            ]
            # 雰囲気タグ
            span_t2_texts = soup.find_all("span", class_="t2")
            dict_for_df[self.cols[2]] += [
                html_text.get_text() for html_text in span_t2_texts
            ]
            # ターゲットタグ
            span_t3_texts = soup.find_all("span", class_="t3")
            dict_for_df[self.cols[3]] += [
                html_text.get_text() for html_text in span_t3_texts
            ]

        self.origin_df = pd.DataFrame(dict_for_df)
        self.origin_df.to_csv(self.phrases_csv_path)

    def embed(self, phrases: Union[List[str], pd.Series]) -> np.ndarray:
        """Sentence BERTによるベクトル化

        Parameters
        ----------
        phrases : Union[List[str], pd.Series]

        Returns
        -------
        np.ndarray
            Shape is (len(phrases), 768)
        """
        vectors = self.model.encode(phrases)
        return vectors

    def crawl_embed(self, is_init: bool = False) -> None:
        """キャッチコピーのスクレイピングとSentence BERTによるベクトル変換を両方行う。
        スクレイピング結果のdataframeに、ベクトル変換された各キャッチコピーの列を追加して、
        インスタンス変数に格納。
        格納されたdataframeはキャッシュとしてcsv出力する。

        Parameters
        ----------
        is_init : bool, optional
            Trueのときスクレイピングを強制的に実行, by default False
        """
        if len(self.origin_df) == 0 or is_init:
            self.crawl()
        self.df = self.origin_df.copy()

        embedded_phrases = self.embed(self.origin_df[self.cols[0]])

        # そのまま to_csv すると、読み込みしづらいフォーマットで書き出されてしまうので、
        # いったんlistに変換してから to_csv する
        self.df[self.cols[-1]] = [x.tolist() for x in embedded_phrases]
        self.df.to_csv(self.embedded_phrases_csv_path)

        self.df[self.cols[-1]] = [np.array(vec) for vec in embedded_phrases]

    def _filter_phrases(
        self,
        max_phrase_length: int = None,
        target_categories: List[str] = None,
        target_atmospheres: List[str] = None,
        target_users: List[str] = None,
    ) -> pd.DataFrame:
        """文字列長とタグ情報でキャッチコピーを絞り込む

        Parameters
        ----------
        max_phrase_length : int, optional
            , by default None

        target_categories : List[str], optional
            , by default None

        target_atmospheres : List[str], optional
            , by default None

        target_users : List[str], optional
            , by default None

        Returns
        -------
        pd.DataFrame
        """
        df = self.df.copy()
        if max_phrase_length:
            df = df[df[self.cols[0]].str.len() <= max_phrase_length]
        if target_categories:
            df = df[df[self.cols[1]].isin(target_categories)]
        if target_atmospheres:
            df = df[df[self.cols[2]].isin(target_atmospheres)]
        if target_users:
            df = df[df[self.cols[3]].isin(target_users)]
        return df

    def _calc_cosine_similarity_matrix(
        self,
        phrase: str,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """入力テキストをベクトル化したもの + 全キャッチコピーのベクトル化結果を用いて
        各テキスト間のコサイン類似度行列を計算。

        Parameters
        ----------
        phrase : str

        df : pd.DataFrame

        Returns
        -------
        np.ndarray
            Shape is (len(df) + 1, len(df) + 1)
        """
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
        embedded_phrase: np.ndarray = None,
        max_phrase_length: int = None,
        target_categories: List[str] = None,
        target_atmospheres: List[str] = None,
        target_users: List[str] = None,
    ) -> Union[pd.DataFrame, None]:
        """`phrase` と各キャッチコピー（絞り込みアリ）とのコサイン類似度を測定し、
        `self.df`に「類似度」列として追加したものを返す。

        Parameters
        ----------
        phrase : str

        embedded_phrase : np.ndarray, optional
            与えられている場合は `phrase` のベクトル化をスキップ, by default None

        max_phrase_length : int, optional
            , by default None

        target_categories : List[str], optional
            , by default None

        target_atmospheres : List[str], optional
            , by default None

        target_users : List[str], optional
            , by default None

        Returns
        -------
        Union[pd.DataFrame, None]
            絞り込みの結果、該当するキャッチコピーがない場合は None を返す
        """
        df = self._filter_phrases(
            max_phrase_length, target_categories, target_atmospheres, target_users
        )
        if len(df) == 0:
            return None

        if embedded_phrase is None:
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
        return pd.merge(df, similarities, left_index=True, right_index=True)

    def calc_mds(
        self,
        phrase: str,
        max_phrase_length: int = None,
        target_categories: List[str] = None,
        target_atmospheres: List[str] = None,
        target_users: List[str] = None,
    ) -> Union[Tuple[np.ndarray, List[str]], Tuple[None, None]]:
        """`phrase` と各キャッチコピー（絞り込みアリ）とのコサイン類似度を測定し、
        各テキストのMDS結果(ndarray)とテキストそのもの(List[str])をラベルとして返す。

        Parameters
        ----------
        phrase : str

        max_phrase_length : int, optional
            , by default None

        target_categories : List[str], optional
            , by default None

        target_atmospheres : List[str], optional
            , by default None

        target_users : List[str], optional
            , by default None

        Returns
        -------
        Union[Tuple[np.ndarray, List[str]], Tuple[None, None]]
            絞り込みの結果、該当するキャッチコピーがない場合は (None, None) を返す
        """
        df = self._filter_phrases(
            max_phrase_length, target_categories, target_atmospheres, target_users
        )
        if len(df) == 0:
            return (None, None)

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
