# 概要
- [キャッチコピー集めました。](https://catchcopy.make1.jp/)からキャッチコピーとタグをスクレイピング
- [Sentence BERT]((https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part9.html))でベクトル化
- 任意の入力から類似/非類似のキャッチコピーを検索

# Usage

## Streamlit Cloud
https://share.streamlit.io/snhryt-neo/catchphrase-generator/main/app.py

（mainブランチの中身が反映されている）

## ローカル
```shell
$ pip install -r requirements.txt
$ cd /path/to/this/repository
$ streamlit run app.py
```
実行に成功すると、Local URLとNetwork URLが表示されるので、好きな方にアクセスする。

# Requirements
- Python version: 3.9
- Packages: `requirements.txt`
