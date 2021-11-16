#!/usr/bin/env python3
import datetime as dt
from typing import List

import pandas as pd
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

from catchphrase import Catchphrase


@st.cache(allow_output_mutation=True)
def init() -> Catchphrase:
    cp = Catchphrase()
    return cp


@st.cache
def convert_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.to_csv().encode("UTF-8")


@st.cache
def calc_mds(
    cp: Catchphrase,
    phrase: str,
    max_phrase_length: int,
    target_categories: List[str],
    target_atmospheres: List[str],
    target_users: List[str],
):
    return cp.calc_mds(
        phrase,
        max_phrase_length=max_phrase_length,
        target_categories=target_categories,
        target_atmospheres=target_atmospheres,
        target_users=target_users,
    )


def create_candidates(cp: Catchphrase, col: str) -> List[str]:
    return cp.df[col].unique().tolist()


def main():
    st.set_page_config(page_title="キャッチコピー生成のプロトタイプ")

    # スクレイピングしたキャッチコピーおよび学習済モデルの読み込み
    cp = init()

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
    st.markdown("---")
    # ==============================================================================

    # 絞り込み＆算出した類似度を表示
    # ==============================================================================
    n_displays = st.slider("表示数", min_value=1, value=10)

    # 各キャッチコピーに対する類似度を計算
    calc_df = cp.calc_similarity_to_phrases(
        keyword,
        max_phrase_length=max_phrase_length,
        target_categories=target_categories,
        target_atmospheres=target_atmospheres,
        target_users=target_users,
    )
    desc_calc_df = calc_df.sort_values("類似度", ascending=False)[:n_displays]
    asc_calc_df = calc_df.sort_values("類似度", ascending=True)[:n_displays]

    # ダウンロードリンク用の情報
    desc_calc_csv = convert_df(desc_calc_df)
    desc_calc_filename = f"類似度top{n_displays}_{dt.date.today()}.csv"
    asc_calc_csv = convert_df(asc_calc_df)
    asc_calc_filename = f"類似度bottom{n_displays}_{dt.date.today()}.csv"

    similarity_container = st.container()
    col1, col2 = similarity_container.columns(2)
    col1.subheader("類似度が高い順")
    col1.table(desc_calc_df[["キャッチコピー", "類似度"]])
    col1.download_button(
        label="download",
        data=desc_calc_csv,
        file_name=desc_calc_filename,
        mime="text/csv",
    )
    col2.subheader("類似度が低い順")
    col2.table(asc_calc_df[["キャッチコピー", "類似度"]])
    col2.download_button(
        label="download",
        data=asc_calc_csv,
        file_name=asc_calc_filename,
        mime="text/csv",
    )
    # ==============================================================================

    # MDSによるキャッチコピーの二次元分布図
    # ==============================================================================
    is_clicked = st.button("Draw scatter")
    if is_clicked:
        # 処理に10分ぐらい時間かかるorz
        positions, labels = calc_mds(
            cp,
            keyword,
            max_phrase_length,
            target_categories,
            target_atmospheres,
            target_users,
        )
        source = ColumnDataSource(
            data=dict(x=positions[:, 0], y=positions[:, 1], desc=labels)
        )
        hover = HoverTool(tooltips=[("desc", "@desc")])
        p = figure(title="catchcopy distribution by MDS", tools=[hover])
        p.circle(source=source, fill_alpha=0.5)
        st.bokeh_chart(p, use_container_width=True)
    # ==============================================================================


if __name__ == "__main__":
    main()
