from typing import List

import streamlit as st
from pymagnitude import Magnitude
from sudachipy import tokenizer, dictionary

# 準備
# ==================================================================================

# 類語検索にはchiVe<https://github.com/WorksApplications/chiVe>の埋め込み表現を利用する
chive_vectors = Magnitude(
    "https://sudachi.s3-ap-northeast-1.amazonaws.com/chive/chive-1.1-mc90-aunit.magnitude",
    stream=True,
)

# 形態素解析はSudachiのMode C（最も単語の結合力が強いモード）で
SUDACHI_SPLIT_MODE = tokenizer.Tokenizer.SplitMode.C
tokenizer_obj = dictionary.Dictionary().create()


def search_similar_words(
    input_word: str, top_N: int = 10, allows_other_part_of_speech: bool = False
) -> List[str]:
    # 入力を正規化して辞書に含まれるフォーマットに変換
    input_word = tokenizer_obj.tokenize(input_word, SUDACHI_SPLIT_MODE)[
        0
    ].normalized_form()

    if allows_other_part_of_speech:  # 品詞問わず置換する場合
        similar_words = chive_vectors.most_similar(
            input_word, topn=top_N, return_similarities=False
        )
    else:  # 同じ品詞のものに絞って置換する場合
        similar_words = []
        input_word_part_of_speech = tokenizer_obj.tokenize(
            input_word, SUDACHI_SPLIT_MODE
        )[0].part_of_speech()[0]

        # 類似度が高い順に単語を見ていき、品詞が同じもののみappendする
        for word in chive_vectors.most_similar(
            input_word, topn=top_N * 5, return_similarities=False
        ):
            part_of_speech = tokenizer_obj.tokenize(word, SUDACHI_SPLIT_MODE)[
                0
            ].part_of_speech()[0]
            if part_of_speech == input_word_part_of_speech:
                similar_words.append(word)
            if len(similar_words) == top_N:
                break
    return similar_words


# ==================================================================================


col1, col2 = st.columns(2)
col2.write("Sudachi (Mode C) による分かち書き結果")
input_sentence = col1.text_input("キャッチコピーを入力してください")
if not input_sentence:
    st.stop()

surfaces = [
    f"`{m.surface()}`"
    for m in tokenizer_obj.tokenize(input_sentence, SUDACHI_SPLIT_MODE)
]
parts = [
    f"`{m.part_of_speech()[0]}`"
    for m in tokenizer_obj.tokenize(input_sentence, SUDACHI_SPLIT_MODE)
]
col2.markdown(" ".join(surfaces))
col2.markdown(" ".join(parts))

target_parts = st.multiselect(
    "置換対象にする品詞を選択してください（複数選択可）", ["動詞", "名詞", "代名詞", "形容詞", "形容動詞", "副詞"]
)
if len(target_parts) == 0:
    st.stop()

replace_target_surfaces = [
    surfaces[i].replace("`", "")
    for i, part in enumerate(parts)
    if part.replace("`", "") in target_parts
]
if len(replace_target_surfaces) == 0:
    st.error("キャッチコピーに含まれる品詞を1つ以上選択してください")
    st.stop()

target_surface = st.radio("置換したい単語を1つ選択してください", replace_target_surfaces)
top_N = st.slider("置換数", 3, 30, value=10)
st.markdown("---")

with st.spinner("類似単語をサーチ中..."):
    similar_words = search_similar_words(target_surface, top_N=top_N)

col1, col2 = st.columns(2)
col1.markdown("**置換後**")
for word in similar_words:
    col1.markdown(input_sentence.replace(target_surface, f"`{word}`"))
col2.markdown("**オリジナル**")
col2.markdown(input_sentence)
