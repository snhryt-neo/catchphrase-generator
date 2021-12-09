#!/usr/bin/env python3
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

# from bokeh.plotting import figure
# from bokeh.models import ColumnDataSource, HoverTool

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


# @st.experimenta_singleton
# def convert_df(df: pd.DataFrame) -> pd.DataFrame:
#     return df.to_csv().encode("UTF-8")
# @st.experimenta_singleton
# def calc_mds(
#     cp: Catchphrase,
#     phrase: str,
#     max_phrase_length: int,
#     target_categories: List[str],
#     target_atmospheres: List[str],
#     target_users: List[str],
# ):
#     return cp.calc_mds(
#         phrase,
#         max_phrase_length=max_phrase_length,
#         target_categories=target_categories,
#         target_atmospheres=target_atmospheres,
#         target_users=target_users,
#     )


def create_candidates(cp: Catchphrase, col: str) -> List[str]:
    return cp.df[col].unique().tolist()


def main():
    st.set_page_config(page_title="キャッチコピー生成のプロトタイプ")

    # スクレイピングしたキャッチコピーおよび学習済モデルの読み込み
    cp = store_embedded_catchphrases()

    st.title("キャッチコピー生成のプロトタイプ")
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
    func_mode = st.selectbox(
        "次に実行する処理を選択してください", ["-", "ニュアンス/雰囲気の類似したキャッチコピー検索", "キャッチコピー内の単語の類語への置き換え"]
    )
    st.markdown("---")

    func_container = st.container()
    with func_container:
        candidates = pd.concat(
            [desc_calc_df["キャッチコピー"], asc_calc_df["キャッチコピー"]]
        ).unique()

        if func_mode == "ニュアンス/雰囲気の類似したキャッチコピー検索":
            st.subheader(func_mode)

            selected_phrases = st.multiselect(
                "上記の中で「このニュアンス/雰囲気使いたいな」と思ったキャッチコピーがあれば選択してください（2〜3個程度）", candidates
            )
            if len(selected_phrases) == 0:
                st.stop()

            embedded_selected_phrases = calc_df[
                calc_df["キャッチコピー"].isin(selected_phrases)
            ]["埋め込み済キャッチコピー"]
            vectors = np.array([vec for vec in embedded_selected_phrases])
            similar_phrases = cp.calc_similarity_to_phrases(
                phrase="",
                embedded_phrase=vectors.mean(axis=0).reshape(1, len(vectors[0])),
                max_phrase_length=max_phrase_length,
                target_categories=target_categories,
                target_atmospheres=target_atmospheres,
                target_users=target_users,
            ).sort_values("類似度", ascending=False)["キャッチコピー"]

            n_mean_displays = st.slider(
                "キャッチコピー表示数", min_value=1, max_value=20, value=5
            )
            counter = 0
            for phrase in similar_phrases:
                if phrase in selected_phrases:
                    continue
                st.markdown(f"`{phrase}`")
                counter += 1
                if counter == n_mean_displays:
                    break

            st.info("※ いまは平均から新規キャッチコピーの生成ではなく、類似した既存キャッチコピーの検索しか行えていない")

        elif func_mode == "キャッチコピー内の単語の類語への置き換え":
            st.subheader(func_mode)

            selected_phrase = st.selectbox(
                "上記の中で「ベースにしたいな」と思ったキャッチコピーがあれば選択してください", ["-"] + candidates.tolist()
            )
            if selected_phrase == "-":
                st.stop()

            word_replacer = init_word_replacer()
            surfaces = word_replacer.wakati(selected_phrase)
            part_of_speechs = [
                part_of_speech[0]
                for part_of_speech in word_replacer.wakati(selected_phrase, mode=4)
            ]

            part_of_speech_choices = set(part_of_speechs) & set(
                ["動詞", "名詞", "代名詞", "形容詞", "形容動詞", "副詞"]
            )
            part_of_speech_targets = st.multiselect(
                "置換対象にする品詞を選択してください（複数選択可）", part_of_speech_choices
            )
            if len(part_of_speech_targets) == 0:
                st.stop()

            replacing_target_surfaces = set(
                [
                    surface
                    for surface, part_of_speech in zip(surfaces, part_of_speechs)
                    if part_of_speech in part_of_speech_targets
                ]
            )
            if len(replacing_target_surfaces) == 0:
                # st.error("キャッチコピーに含まれる品詞を1つ以上選択してください")
                st.stop()

            target_surface = st.radio(
                "置換したい単語を1つ選択してください", ["-"] + list(replacing_target_surfaces)
            )
            st.markdown("---")

            top_N = st.slider("最大表示キャッチコピー数", 3, 30, value=10)
            normalization = st.checkbox(
                "置換前の単語を正規化する（例: 附属 → 付属、SUMMER → サマー、シュミレーション → シミュレーション）"
            )
            if target_surface == "-":
                st.stop()

            with st.spinner("類似語に置換した文章生成中..."):
                replaced_phrases = word_replacer.generate_replaced_sentences(
                    selected_phrase,
                    target_surface,
                    top_N=top_N,
                    normalization=normalization,
                )

            left, right = st.columns(2)
            left.markdown("**置換後**")
            for phrase in replaced_phrases:
                left.markdown(phrase)
            right.markdown("**オリジナル**")
            if normalization:
                replaced_phrase = selected_phrase.replace(
                    target_surface, f"`{word_replacer.normalize(target_surface)}`", 1
                )
            else:
                replaced_phrase = selected_phrase.replace(
                    target_surface, f"`{target_surface}`", 1
                )
            right.markdown(replaced_phrase)

        else:
            st.stop()

    # TODO: 実装する
    # MDSによるキャッチコピーの二次元分布図
    # ==============================================================================
    # is_clicked = st.button("Draw scatter")
    # if is_clicked:
    #     # 処理に5分ぐらい時間かかるorz
    #     with st.spinner("Calculating MDS..."):
    #         positions, labels = calc_mds(
    #             cp,
    #             keyword,
    #             max_phrase_length,
    #             target_categories,
    #             target_atmospheres,
    #             target_users,
    #         )
    #     st.success("Done!")

    #     source = ColumnDataSource(
    #         data=dict(x=positions[:, 0], y=positions[:, 1], desc=labels)
    #     )
    #     hover = HoverTool(tooltips=[("desc", "@desc")])
    #     p = figure(title="catchcopy distribution by MDS", tools=[hover])
    #     p.circle(source=source, fill_alpha=0.5)
    #     st.bokeh_chart(p, use_container_width=True)
    # ==============================================================================


if __name__ == "__main__":
    main()
