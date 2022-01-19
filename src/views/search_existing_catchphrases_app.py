import datetime as dt
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from utils.catchphrase import Catchphrase


@st.experimental_singleton
def store_embedded_catchphrases() -> Catchphrase:
    cp = Catchphrase()
    if len(cp.df) == 0:
        cp.crawl_embed(is_init=True)
    return cp


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
