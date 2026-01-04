# ベースイメージとしてPython 3.12を指定
FROM python:3.12-slim

# opencv-pythonが必要とするシステムライブラリをインストール
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係ファイルをコンテナにコピー
COPY requirements.txt .

# 依存ライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# プロジェクトのソースコードをコンテナにコピー
COPY . .
