# What is it?
キャッチコピーを考えるときに使えそうな2つの機能を実装しています。

## 1. 類語置き換え
- 入力キャッチコピーを形態素解析し、その中から選択した単語を類語で置換したものを出力する
- 形態素解析: SudachiのMode C（最も単語の結合力が強いモード）
- 類語検索: [chiVe](https://github.com/WorksApplications/chiVe)の埋め込み表現を利用する

## 2. 既存キャッチコピー検索
- [キャッチコピー集めました。](https://catchcopy.make1.jp/)からキャッチコピーとタグをスクレイピング
- [Sentence BERT]((https://www.ogis-ri.co.jp/otc/hiroba/technical/similar-document-search/part9.html))でベクトル化
- 任意の入力から類似/非類似のキャッチコピーを検索

# Usage

## Streamlit Cloud
https://share.streamlit.io/snhryt-neo/catchphrase-generator/main/src/app.py

（mainブランチの中身が反映されている）

## ローカル

事前準備①:
```shell
$ cd /path/to/this/repository
$ pip install -r requirements.txt
```

事前準備②: `src/.streamlit` ディレクトリ内に `secrets.toml` ファイルを配置する


```shell
$ cd src
$ streamlit run app.py
```
実行に成功すると、ブラウザで http://localhost:8501Local が自動的に開く。
（もしくは、ターミナルに表示されたLocal URL or Network URLをコピペしてアクセス）

# Requirements
- Python version: >= 3.9
- Packages: `requirements.txt`

# Directories
```
src/
├ .streamlit/
│ └ secrets.toml : GCP, BigQuery関連のcredentials (Gitの管理対象からは除外)
├ utils/
│ ├ catchphrase.py   : ベクトル化されたキャッチコピー群を用いて、類似/非類似キャッチコピーの検索
│ └ word_replacer.py : キャッチコピー内の特定単語の置換
│ ├ gbq.py           : BigQueryへのdataframeのアップロード
├ views/
│ ├ search_existing_catchphrases_app.py : 既存キャッチコピー検索のstreamlit画面
│ └ word_replace_app.py                 : キャッチコピー類語置換のstreamlit画面
└ app.py : mainファイル
README.md
.gitignore
```
