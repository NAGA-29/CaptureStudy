import cv2
import mediapipe as mp
import argparse
import os
import numpy as np
import pygltflib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
    """OpenCVを使って、画像にランドマークと接続線を描画する"""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    # ランドマークを描画
    for pose_landmarks in pose_landmarks_list:
        # Draw the landmarks
        for landmark in pose_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(annotated_image, (x, y), 5, (245, 117, 66), -1)

    return annotated_image

def process_video(model_path, input_path, output_path, output_glb_path=None):
    """
    指定されたビデオファイルを読み込み、新しいMediaPipe Tasks APIで姿勢推定を行い、
    結果を新しいビデオファイルとオプションでGLBファイルに保存する。
    """
    # --- MediaPipe Pose Landmarkerの初期化 ---
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    # モデルのパスを解決
    if not os.path.exists(model_path):
        print(f"Error: モデルファイルが見つかりません: {model_path}")
        return

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_segmentation_masks=False  # セグメンテーションは不要
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        # --- 入力ビデオの読み込み ---
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: 入力ファイルを開けません: {input_path}")
            return

        # --- ビデオのプロパティと出力ライターの準備 ---
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_world_landmarks = []
        frame_times = []
        frame_count = 0

        print("処理を開始します...")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # BGRからRGBに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe Imageオブジェクトに変換
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # タイムスタンプを計算 (ミリ秒)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # 姿勢推定を実行
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # --- 結果の処理と保存 ---
            if detection_result.pose_world_landmarks:
                # 3Dワールドランドマークを収集
                all_world_landmarks.append(detection_result.pose_world_landmarks[0]) # 最初の人のみ
                frame_times.append(frame_count / fps)

            # 描画
            annotated_image = draw_landmarks_on_image(frame_rgb, detection_result)

            # RGBからBGRに戻して出力
            out.write(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            frame_count += 1

        print(f"処理が完了しました。出力ファイル: {output_path}")

        # --- GLBファイルのエクスポート ---
        if output_glb_path and all_world_landmarks:
            print(f"GLBファイルを出力します: {output_glb_path}")
            save_to_glb(all_world_landmarks, frame_times, output_glb_path)
        elif output_glb_path:
            print("警告: ランドマークが検出されなかったため、GLBファイルは出力されません。")

        # --- 後処理 ---
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def save_to_glb(all_world_landmarks, frame_times, output_path):
    """収集した3DランドマークデータからGLBファイルを作成する"""
    gltf = pygltflib.GLTF2()
    scene = pygltflib.Scene()
    gltf.scenes.append(scene)

    landmark_names = [f'landmark_{i}' for i in range(33)] # ランドマーク名はシンプルに
    nodes = [pygltflib.Node(name=name) for name in landmark_names]
    gltf.nodes.extend(nodes)
    scene.nodes.extend(range(len(nodes))) # すべてのノードをシーンのルートに追加

    # 1. タイムスタンプのバッファデータ
    times_data = np.array(frame_times, dtype=np.float32)
    times_blob = times_data.tobytes()

    # 2. ランドマークのトランスレーションデータ
    translations_blob = b''
    all_translations = []
    for i in range(len(landmark_names)):
        node_translations = []
        for frame_landmarks in all_world_landmarks:
            lm = frame_landmarks[i]
            # MediaPipeの右手系ワールド座標からGLTFの右手系座標へ変換。
            # XとZを反転させることで、カメラの前方を-Z軸とする標準的なGLTFの慣習に合わせる。
            # 新しいTasks APIでは、Y軸はすでに上向き。
            node_translations.append([-lm.x, lm.y, -lm.z])
        all_translations.append(np.array(node_translations, dtype=np.float32))

    # --- GLTFバッファとビューの作成 ---
    buffer_offset = 0
    gltf.buffers.append(pygltflib.Buffer(byteLength=0)) # 後でサイズを更新

    # タイムスタンプのビューとアクセサ
    times_buffer_view = pygltflib.BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(times_blob))
    gltf.bufferViews.append(times_buffer_view)
    times_accessor = pygltflib.Accessor(
        bufferView=0, componentType=pygltflib.FLOAT, count=len(frame_times), type=pygltflib.SCALAR,
        max=[float(np.max(times_data))], min=[float(np.min(times_data))]
    )
    gltf.accessors.append(times_accessor)
    buffer_offset += len(times_blob)

    # アニメーションの作成
    animation = pygltflib.Animation()
    gltf.animations.append(animation)

    # 各ノード（ランドマーク）のアニメーションチャネルを作成
    for i, node_translations_data in enumerate(all_translations):
        node_translations_blob = node_translations_data.tobytes()

        # トランスレーションのビューとアクセサ
        trans_buffer_view = pygltflib.BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(node_translations_blob))
        gltf.bufferViews.append(trans_buffer_view)

        trans_accessor = pygltflib.Accessor(
            bufferView=len(gltf.bufferViews) - 1, componentType=pygltflib.FLOAT, count=len(node_translations_data), type=pygltflib.VEC3,
            max=np.max(node_translations_data, axis=0).tolist(),
            min=np.min(node_translations_data, axis=0).tolist()
        )
        gltf.accessors.append(trans_accessor)
        buffer_offset += len(node_translations_blob)

        # サンプラー
        sampler = pygltflib.AnimationSampler(
            input=0, # タイムスタンプアクセサのインデックス
            output=len(gltf.accessors) - 1, # トランスレーションアクセサのインデックス
            interpolation=pygltflib.LINEAR
        )
        animation.samplers.append(sampler)

        # チャネル
        channel = pygltflib.AnimationChannel(
            sampler=len(animation.samplers) - 1,
            target=pygltflib.AnimationChannelTarget(node=i, path="translation")
        )
        animation.channels.append(channel)

    # 結合したバッファデータを作成
    final_blob = times_blob
    for node_translations_data in all_translations:
        final_blob += node_translations_data.tobytes()

    gltf.buffers[0].uri = final_blob
    gltf.buffers[0].byteLength = len(final_blob)

    # GLBとして保存
    gltf.convert_buffers(pygltflib.BufferFormat.BINARY)
    gltf.save_binary(output_path)


def main():
    parser = argparse.ArgumentParser(description="ビデオから姿勢推定を行い、結果を保存するスクリプト。")
    parser.add_argument("--model", type=str, default="pose_landmarker_heavy.task", help="使用するMediaPipeモデルファイル (.task) のパス。")
    parser.add_argument("--input", type=str, required=True, help="入力ビデオファイルのパス。")
    parser.add_argument("--output", type=str, required=True, help="出力ビデオファイルのパス。")
    parser.add_argument("--output-glb", type=str, help="出力GLBアニメーションファイルのパス。")
    args = parser.parse_args()

    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.output_glb:
        glb_output_dir = os.path.dirname(args.output_glb)
        if glb_output_dir and not os.path.exists(glb_output_dir):
            os.makedirs(glb_output_dir)

    process_video(args.model, args.input, args.output, args.output_glb)

if __name__ == '__main__':
    main()
