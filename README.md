# 映像からのモーションキャプチャ実験 (Motion Capture from Video Experiment)

これは、Pythonを使用してビデオ映像から人物の姿勢を検出し、モーションキャプチャを行うための実験的プロジェクトです。

## 概要

このプロジェクトは、OpenCVとMediaPipeライブラリを活用して、ビデオファイル内の人物の関節点（ランドマーク）をリアルタイムで追跡し、3Dモーションデータをアニメーション付きGLBファイルとして出力します。

## プロジェクト構成

```
.
├── Docs/
│   └── plan.md         # プロジェクト計画書
├── data/
│   ├── input/          # 入力ビデオを格納するディレクトリ
│   └── output/         # 出力結果（GLBアニメーション等）を格納するディレクトリ
├── src/
│   └── capture.py      # モーションキャプチャ処理のメインスクリプト
├── README.md           # このファイル
└── requirements.txt    # プロジェクトの依存ライブラリ
```

## セットアップ

1.  **リポジトリのクローン:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **依存ライブラリのインストール:**
    プロジェクトのルートディレクトリで、以下のコマンドを実行して必要なライブラリをインストールします。
    ```bash
    pip install -r requirements.txt
    ```

## Dockerを使った開発 (Development with Docker)

Dockerを利用することで、依存関係を気にすることなく、誰でも同じ開発環境を簡単に構築できます。

1.  **Dockerイメージのビルドとコンテナの起動:**
    プロジェクトのルートディレクトリで、以下のコマンドを実行します。
    ```bash
    docker-compose build
    docker-compose up -d
    ```
    これにより、`Dockerfile`を元にイメージがビルドされ、バックグラウンドでコンテナが起動します。

2.  **コンテナ内でのコマンド実行:**
    起動したコンテナの中で`capture.py`スクリプトを実行するには、`docker-compose exec`を使用します。
    ```bash
    docker-compose exec app python src/capture.py --input data/input/your_video.mp4 --output data/output/preview.mp4 --output-glb data/output/animation.glb
    ```
    また、コンテナのシェルにアクセスして、インタラクティブに作業することも可能です。
    ```bash
    docker-compose exec app /bin/bash
    ```

## 使い方

1.  **入力ビデオの準備:**
    処理したいビデオファイルを `data/input/` ディレクトリに配置します。

2.  **スクリプトの実行:**
    以下のコマンドを実行して、モーションキャプチャ処理を開始します。

    ```bash
    python src/capture.py --input data/input/your_video.mp4 --output data/output/preview.mp4 --output-glb data/output/animation.glb
    ```

    **引数:**
    - `--input`: 入力ビデオファイルのパスを指定します。
    - `--output`: 骨格が描画されたプレビュー用ビデオファイルのパスを指定します。
    - `--output-glb`: 3Dモーションデータを保存するGLBアニメーションファイルのパスを指定します。
    - `--model` (オプション): 使用するMediaPipeモデルファイル (.task) のパスを指定します。

    **出力ファイル:**
    処理が完了すると、`data/output/` ディレクトリに以下のファイルが生成されます。
    - `animation.glb` - **メイン出力**: 3Dモーションキャプチャデータを含むアニメーションファイル
    - `preview.mp4` - 確認用: 骨格が描画されたプレビュー動画

## 使用技術

- [OpenCV](https://opencv.org/): 映像の読み書きと描画
- [MediaPipe](https://mediapipe.dev/): 高精度な姿勢推定
