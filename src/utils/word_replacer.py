#!/usr/bin/env python3
import dataclasses
from enum import IntEnum, auto
from typing import List

from pymagnitude import Magnitude
from sudachipy import tokenizer, dictionary


class MorphemeMode(IntEnum):
    SURFACE = auto()
    DICTIONARY_FORM = auto()
    READING_FORM = auto()
    PART_OF_SPEECH = auto()


@dataclasses.dataclass
class WordReplacer:
    chive_vectors: Magnitude = dataclasses.field(init=False)
    tokenizer_obj: tokenizer.Tokenizer = dataclasses.field(init=False)
    sudachi_split_mode: tokenizer.Tokenizer.SplitMode = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        # 類語検索にはchiVe<https://github.com/WorksApplications/chiVe>の埋め込み表現を利用する
        self.chive_vectors = Magnitude(
            "https://sudachi.s3-ap-northeast-1.amazonaws.com/chive/chive-1.1-mc90-aunit.magnitude",
            # stream=True,
        )

        self.tokenizer_obj = dictionary.Dictionary().create()

        # 形態素解析はSudachiのMode C（最も単語の結合力が強いモード）で
        self.sudachi_split_mode = tokenizer.Tokenizer.SplitMode.C

    def wakati(
        self, sentence: str, mode: MorphemeMode = MorphemeMode.SURFACE
    ) -> List[str]:
        """Sudachiによる分かち書きの実行

        Parameters
        ----------
        sentence : str

        mode : IntEnum, optional
            , by default 1.
            1: SURFACE, 2: DICTIONARY_FORM, 3: READING_FORM, 4: PART_OF_SPEECH

        Returns
        -------
        List[str]
        """
        tokens = self.tokenizer_obj.tokenize(sentence, self.sudachi_split_mode)
        if mode == MorphemeMode.SURFACE:
            return [m.surface() for m in tokens]
        elif mode == MorphemeMode.DICTIONARY_FORM:
            return [m.dictionary_form() for m in tokens]
        elif mode == MorphemeMode.READING_FORM:
            return [m.reading_form() for m in tokens]
        else:
            return [m.part_of_speech() for m in tokens]

    def normalize(self, word: str) -> str:
        """入力の正規化

        Parameters
        ----------
        word : str

        Returns
        -------
        str
        """
        return self.tokenizer_obj.tokenize(word, self.sudachi_split_mode)[
            0
        ].normalized_form()

    def calc_similarity(self, word: str, topn: int = None) -> List[str]:
        """類似単語を返す

        Parameters
        ----------
        word : str

        topn : int, optional
            , by default None

        Returns
        -------
        List[str]
        """
        return self.chive_vectors.most_similar(
            word, topn=topn, return_similarities=False
        )

    def search_similar_words(
        self,
        input_word: str,
        top_N: int = 10,
        normalization: bool = True,
        allows_other_part_of_speech: bool = False,
    ) -> List[str]:
        """類似単語を返す

        Parameters
        ----------
        input_word : str

        top_N : int, optional
            , by default 10

        normalization : bool, optional
            , by default True. Trueの場合、入力単語の正規化を行う

        allows_other_part_of_speech : bool, optional
            , by default False. Trueの場合、類似単語に品詞違いのものも含める

        Returns
        -------
        List[str]
        """
        if normalization:
            input_word = self.normalize(input_word)

        if allows_other_part_of_speech:  # 品詞問わず置換する場合
            similar_words = self.calc_similarity(input_word, topn=top_N)
        else:  # 同じ品詞のものに絞って置換する場合
            similar_words = []
            input_word_part_of_speech = self.wakati(
                input_word, mode=MorphemeMode.PART_OF_SPEECH
            )[0][0]
            # 類似度が高い順に単語を見ていき、品詞が同じもののみappendする
            for word in self.calc_similarity(input_word, topn=top_N * 5):
                part_of_speech = self.wakati(word, mode=MorphemeMode.PART_OF_SPEECH)[0][
                    0
                ]
                if part_of_speech == input_word_part_of_speech:
                    similar_words.append(word)
                if len(similar_words) == top_N:
                    break
        return similar_words

    def generate_replaced_sentences(
        self, sentence: str, from_word: str, top_N: int = 10, normalization: bool = True
    ) -> List[str]:
        """`sentence` 内の `from_word` を類語で置換した文章のリストを返す

        Parameters
        ----------
        sentence : str

        from_word : str

        top_N : int, optional
            , by default 10

        normalization : bool, optional
            , by default True. Trueの場合、入力単語の正規化を行う

        Returns
        -------
        List[str]
        """
        assert from_word in sentence

        to_word_candidates = self.search_similar_words(
            from_word, top_N=top_N, normalization=normalization
        )
        replaced_sentences = []
        for to_word in to_word_candidates:
            replaced_sentence = sentence.replace(from_word, to_word, 1)
            replaced_sentences.append(replaced_sentence)
        return replaced_sentences


if __name__ == "__main__":
    word_replacer = WordReplacer()
    sentence = "そうだ、京都行こう。"
    print(word_replacer.generate_replaced_sentences(sentence, "京都"))
