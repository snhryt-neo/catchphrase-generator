#!/usr/bin/env python3
import streamlit as st

from views.search_existing_catchphrases_app import search_existing_catchphrases_app
from views.word_replace_app import word_replace_app


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
