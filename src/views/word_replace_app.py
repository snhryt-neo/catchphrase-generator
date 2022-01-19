import datetime as dt

import pandas as pd
import streamlit as st


from utils.word_replacer import WordReplacer
from utils.gbq import GBQ

SCHEMA = [
    {
        "description": "submit日時",
        "name": "created_at_jst",
        "type": "DATETIME",
        "mode": "REQUIRED",
    },
    {
        "description": "入力キャッチコピー",
        "name": "input_phrase",
        "type": "STRING",
        "mode": "REQUIRED",
    },
    {
        "description": "最終的に決定したキャッチコピー",
        "name": "decided_phrase",
        "type": "STRING",
        "mode": "NULLABLE",
    },
    {
        "description": "生成されたキャッチコピー",
        "name": "generated_phrase",
        "type": "STRING",
        "mode": "REQUIRED",
    },
    {
        "description": "生成されたキャッチコピーに対するフィードバック",
        "name": "is_good",
        "type": "BOOLEAN",
        "mode": "REQUIRED",
    },
]


@st.experimental_singleton
def init_word_replacer() -> WordReplacer:
    return WordReplacer()


@st.experimental_singleton
def init_gbq() -> GBQ:
    return GBQ(st.secrets["CREDENTIALS"]["project_id"], st.secrets["CREDENTIALS"])


def word_replace_app(placeholder) -> None:
    PART_OF_SPEECH_TARGETS = {"動詞", "名詞", "代名詞", "形容詞", "形容動詞", "副詞"}
    word_replacer = init_word_replacer()
    gbq = init_gbq()

    container = placeholder.container()
    with container:
        st.title("キャッチコピーの類語置換")

        st.subheader("任意のキャッチコピーを入力してください")
        input_phrase = st.text_input("（句読点を含むものでも可）")
        if len(input_phrase) == 0:
            st.stop()

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

        n_generated_phrases = len(replaced_input_phrases)
        generated_phrases = [None] * n_generated_phrases
        evaluations = [None] * n_generated_phrases

        st.markdown("**オリジナル**")
        st.markdown(input_phrase)
        left, right = st.columns((3, 1))
        left.markdown("**置換後**")
        right.markdown("気に入ったら✓")
        for i, phrase in enumerate(replaced_input_phrases):
            left.markdown(phrase)
            original_phrase = phrase.replace(" `", "").replace("` ", "")
            generated_phrases[i] = original_phrase
            evaluations[i] = right.checkbox(" ", key=original_phrase)
        final_phrase = st.text_input("最終的に決定したキャッチコピー")

        result_df = pd.DataFrame(
            {
                SCHEMA[0]["name"]: [dt.datetime.now()] * n_generated_phrases,
                SCHEMA[1]["name"]: [input_phrase] * n_generated_phrases,
                SCHEMA[2]["name"]: [final_phrase] * n_generated_phrases,
                SCHEMA[3]["name"]: generated_phrases,
                SCHEMA[4]["name"]: evaluations,
            }
        )
        st.markdown("---")
        submit_button = st.button("Submit")
        status = st.empty()
        if submit_button:
            with st.spinner("Submitting feedback..."):
                gbq.append(
                    result_df,
                    st.secrets["BQ"]["DATASET"],
                    st.secrets["BQ"]["TABLE"],
                    table_schema=SCHEMA,
                )
            status.success("Submit completed!")
            st.balloons()
