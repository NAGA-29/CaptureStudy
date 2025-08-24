# 映像からのモーションキャプチャ実験 (Motion Capture from Video Experiment)

これは、Pythonを使用してビデオ映像から人物の姿勢を検出し、モーションキャプチャを行うための実験的プロジェクトです。

## 概要

このプロジェクトは、OpenCVとMediaPipeライブラリを活用して、ビデオファイル内の人物の関節点（ランドマーク）をリアルタイムで追跡し、その結果を可視化して新しいビデオファイルとして保存します。

## プロジェクト構成

```
.
├── Docs/
│   └── plan.md         # プロジェクト計画書
├── data/
│   ├── input/          # 入力ビデオを格納するディレクトリ
│   └── output/         # 出力結果（映像やデータ）を格納するディレクトリ
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

## 使い方

1.  **入力ビデオの準備:**
    処理したいビデオファイルを `data/input/` ディレクトリに配置します。

2.  **スクリプトの実行:**
    以下のコマンドを実行して、モーションキャプチャ処理を開始します。

    ```bash
    python src/capture.py --input data/input/your_video.mp4 --output data/output/result.mp4 --output-glb data/output/animation.glb --model-complexity 2
    python src/capture.py --input data/input/your_video.mp4 --output data/output/result.mp4 --output-glb data/output/animation.glb
    ```
    - `--input`: 入力ビデオファイルのパスを指定します。
    - `--output`: 骨格が描画された結果を保存する出力ビデオファイルのパスを指定します。
    - `--output-glb` (オプション): 3Dモーションデータを保存するGLBファイルのパスを指定します。
    - `--model-complexity` (オプション): 姿勢推定モデルの複雑さを指定します (0: lite, 1: full, 2: heavy)。数値が大きいほど高精度ですが、処理が遅くなります。デフォルトは `2` です。

    処理が完了すると、`data/output/` ディレクトリに `result.mp4` と `animation.glb` が生成されます。

## 使用技術

- [OpenCV](https://opencv.org/): 映像の読み書きと描画
- [MediaPipe](https://mediapipe.dev/): 高精度な姿勢推定
