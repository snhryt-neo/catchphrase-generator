#!/usr/bin/env python3
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from catchphrase import Catchphrase
from word_replacer import WordReplacer


@st.experimental_singleton
def store_embedded_catchphrases() -> Catchphrase:
    cp = Catchphrase()
    if len(cp.df) == 0:
        cp.crawl_embed(is_init=True)
    return cp


@st.experimental_singleton
def init_word_replacer() -> WordReplacer:
    return WordReplacer()


def create_candidates(cp: Catchphrase, col: str) -> List[str]:
    return cp.df[col].unique().tolist()


def search_existing_catchphrases_app(placeholder) -> None:
    # スクレイピングしたキャッチコピーおよび学習済モデルの読み込み
    cp = store_embedded_catchphrases()

    container = placeholder.container()
    with container:
        st.title("既存キャッチコピーの検索")
        st.markdown("キャッチコピーのスクレイピング元: [キャッチコピー集めました。](https://catchcopy.make1.jp/)")
        st.markdown("---")

        st.subheader("任意のキーワードを入力してください")
        keyword = st.text_input("（単語でもフレーズでも可）")

        # 表示されるキャッチコピーの絞り込み条件
        # ==============================================================================
        st.subheader("絞り込み条件")
        filter_container = st.container()

        max_phrase_length = filter_container.slider(
            "キーワードの文字数上限（値が0のときは文字数での絞り込みはナシ）", min_value=0, value=0
        )
        if max_phrase_length == 0:
            max_phrase_length = None

        col1, col2, col3 = filter_container.columns(3)
        target_categories = col1.multiselect("カテゴリ", create_candidates(cp, "カテゴリ"))
        if len(target_categories) == 0:
            target_categories = None
        target_atmospheres = col2.multiselect("印象", create_candidates(cp, "印象"))
        if len(target_atmospheres) == 0:
            target_atmospheres = None
        target_users = col3.multiselect("ターゲット", create_candidates(cp, "ターゲット"))
        if len(target_users) == 0:
            target_users = None

        filter_msg_placeholder = st.empty()
        st.markdown("---")
        # ==============================================================================

        # 絞り込み＆算出した類似度を表示
        # ==============================================================================
        if len(keyword) == 0:
            st.stop()

        n_displays = st.slider("キャッチコピー表示数", min_value=1, value=10)

        # 各キャッチコピーに対する類似度を計算
        calc_df = cp.calc_similarity_to_phrases(
            phrase=keyword,
            max_phrase_length=max_phrase_length,
            target_categories=target_categories,
            target_atmospheres=target_atmospheres,
            target_users=target_users,
        )
        if calc_df is None:
            filter_msg_placeholder.error("該当するキャッチコピーが存在しません。絞り込み条件を変えてください。")
            st.stop()

        desc_calc_df = calc_df.sort_values("類似度", ascending=False)[:n_displays]
        asc_calc_df = calc_df.sort_values("類似度", ascending=True)[:n_displays]

        similarity_container = st.container()
        col1, col2 = similarity_container.columns(2)
        col1.subheader("類似度が高い順")
        col1.table(desc_calc_df[["キャッチコピー", "類似度"]])
        col2.subheader("類似度が低い順")
        col2.table(asc_calc_df[["キャッチコピー", "類似度"]])

        st.markdown("---")
        # ==============================================================================

        # キャッチコピーの平均を算出し、類似したキャッチコピーを検索
        # ==============================================================================
        candidates = pd.concat(
            [desc_calc_df["キャッチコピー"], asc_calc_df["キャッチコピー"]]
        ).unique()
        selected_phrases = st.multiselect(
            "上記の中で「このニュアンス/雰囲気使いたいな」と思ったキャッチコピーがあれば選択してください（2〜3個程度）", candidates
        )
        if len(selected_phrases) == 0:
            st.stop()

        embedded_selected_phrases = calc_df[calc_df["キャッチコピー"].isin(selected_phrases)][
            "埋め込み済キャッチコピー"
        ]
        vectors = np.array([vec for vec in embedded_selected_phrases])
        similar_phrases = cp.calc_similarity_to_phrases(
            phrase="",
            embedded_phrase=vectors.mean(axis=0).reshape(1, len(vectors[0])),
            max_phrase_length=max_phrase_length,
            target_categories=target_categories,
            target_atmospheres=target_atmospheres,
            target_users=target_users,
        ).sort_values("類似度", ascending=False)["キャッチコピー"]

        n_mean_displays = st.slider("キャッチコピー表示数", min_value=1, max_value=20, value=5)
        counter = 0
        for phrase in similar_phrases:
            if phrase in selected_phrases:
                continue
            st.markdown(f"`{phrase}`")
            counter += 1
            if counter == n_mean_displays:
                break

        st.info("※ いまは平均から新規キャッチコピーの生成ではなく、類似した既存キャッチコピーの検索しか行えていない")
        # ==============================================================================


def word_replace_app(placeholder) -> None:
    PART_OF_SPEECH_TARGETS = {"動詞", "名詞", "代名詞", "形容詞", "形容動詞", "副詞"}
    container = placeholder.container()

    with container:
        st.title("キャッチコピーの類語置換")

        st.subheader("任意のキャッチコピーを入力してください")
        input_phrase = st.text_input("（句読点を含むものでも可）")
        if len(input_phrase) == 0:
            st.stop()

        word_replacer = init_word_replacer()

        # 特定の品詞の単語のみをmultiselectの候補として表示
        surfaces = word_replacer.wakati(input_phrase)
        part_of_speechs = [p[0] for p in word_replacer.wakati(input_phrase, mode=4)]
        surface_choices = [
            surface
            for surface, part_of_speech in zip(surfaces, part_of_speechs)
            if part_of_speech in PART_OF_SPEECH_TARGETS
        ]

        # 以降では、ここで選ばれた単語の類語を出していく
        target_surfaces = st.multiselect(
            "置換対象とする単語を選択ください（複数選択可）", set(surface_choices)
        )
        st.markdown("---")
        if len(target_surfaces) == 0:
            st.stop()

        top_N = st.slider("類似語の最大表示数", 3, 30, value=10)

        # 選択された単語ごとの類語を候補として提示し、そのうち気に入ったものをユーザーが選択する
        surface_selected_words_dict = {}
        for i, surface in enumerate(target_surfaces):
            expander = st.expander(f'単語{i + 1}: "{surface}"', expanded=i == 0)
            with expander:
                normalization = st.checkbox(
                    "置換前の単語を正規化する（例: 附属 → 付属、SUMMER → サマー、シュミレーション → シミュレーション）",
                    key=f"{i}_checkbox",
                )
                if normalization:
                    normalized_surface = word_replacer.normalize(surface)
                else:
                    normalized_surface = surface
                similar_words = word_replacer.search_similar_words(
                    normalized_surface, top_N=top_N, normalization=normalization
                )
                selected_words = st.multiselect(
                    "以下から採用する類語を選択してください", similar_words, key=f"{i}_multiselect"
                )
                surface_selected_words_dict[surface] = selected_words
        if len(list(surface_selected_words_dict.values())[0]) == 0:
            st.stop()

        # 選択された単語 x その単語の類語のうち選択された数 個分のキャッチコピーを生成
        replaced_input_phrases = set()
        for surface, selected_words in surface_selected_words_dict.items():
            for word in selected_words:
                replaced_input_phrases.add(input_phrase.replace(surface, f" `{word}` "))
                tmp = set()
                for phrase in replaced_input_phrases:
                    tmp.add(phrase.replace(surface, f" `{word}` "))
                replaced_input_phrases = replaced_input_phrases | tmp
        replaced_input_phrases = sorted(list(replaced_input_phrases))
        st.markdown("---")

        left, right = st.columns(2)
        left.markdown("**置換後**")
        for phrase in replaced_input_phrases:
            left.markdown(phrase)
        right.markdown("**オリジナル**")
        right.markdown(input_phrase)


def main():
    st.set_page_config(page_title="キャッチコピー作成補助ツール")

    mode = st.sidebar.radio("モード", ["類語置き換え", "既存キャッチコピー検索"])
    placeholder = st.empty()
    if mode == "類語置き換え":
        word_replace_app(placeholder)
    elif mode == "既存キャッチコピー検索":
        search_existing_catchphrases_app(placeholder)


if __name__ == "__main__":
    main()
