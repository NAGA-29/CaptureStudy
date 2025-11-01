import cv2
import mediapipe as mp
import argparse
import os
import numpy as np
from pygltflib import *

def process_video(input_path, output_path, output_glb_path=None, model_complexity=2):
    """
    指定されたビデオファイルを読み込み、姿勢推定を行って結果を新しいビデオファイルに保存する。
    オプションで、3DランドマークをGLBファイルとしてエクスポートする。

    Args:
        input_path (str): 入力ビデオファイルのパス。
        output_path (str): 出力ビデオファイルのパス。
        output_glb_path (str, optional): 出力GLBファイルのパス。
        model_complexity (int, optional): モデルの複雑さ (0, 1, 2)。
    """
    # MediaPipeの準備
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=model_complexity,
                        smooth_landmarks=True,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # 入力ビデオの読み込み
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: 入力ファイルを開けません: {input_path}")
        return

    # ビデオのプロパティを取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 出力ビデオライターの準備
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_world_landmarks = []
    frame_times = []
    frame_count = 0

    print("処理を開始します...")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # パフォーマンス向上のため、画像を書き込み不可に設定
        image.flags.setflags(write=False)
        # BGRからRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 姿勢推定を実行
        results = pose.process(image_rgb)

        # 3Dランドマークを収集
        if results.pose_world_landmarks:
            all_world_landmarks.append(results.pose_world_landmarks.landmark)
            frame_times.append(frame_count / fps)

        frame_count += 1

        # 画像を書き込み可能に戻す
        image.flags.setflags(write=True)

        # 2Dランドマークを描画
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # 出力ビデオにフレームを書き込む
        out.write(image)

    print(f"処理が完了しました。出力ファイル: {output_path}")

    # GLBファイルを出力
    if output_glb_path and all_world_landmarks:
        print(f"GLBファイルを出力します: {output_glb_path}")
        export_to_glb(all_world_landmarks, frame_times, output_glb_path)
    elif output_glb_path:
        print("警告: ランドマークが検出されなかったため、GLBファイルは出力されません。")


    # 後処理
    pose.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def export_to_glb(all_world_landmarks, frame_times, output_path):
    """
    収集した3DランドマークデータからGLBファイルを作成する。
    Args:
        all_world_landmarks (list): フレームごとのランドマークのリスト。
        frame_times (list): 各フレームのタイムスタンプのリスト。
        output_path (str): 出力GLBファイルのパス。
    """
    gltf = GLTF2()
    scene = Scene()
    gltf.scenes.append(scene)

    # --- ノードとスケルトンの設定 ---
    # MediaPipeのランドマーク名
    landmark_names = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow',
        'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
        'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
        'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
    ]

    nodes = []
    for i, name in enumerate(landmark_names):
        # MediaPipeの座標系は右手系、Yが下向き。GLTFは右手系、Yが上向き。
        # YとZを反転させて調整する。
        node = Node(name=name, translation=[0.0, 0.0, 0.0])
        nodes.append(node)

    gltf.nodes.extend(nodes)

    # 簡単な親子関係を設定 (例: 肩 -> 肘 -> 手首)
    # 本来は完全なスケルトン階層を定義する
    parent_child_map = {
        11: [13], 13: [15], 15: [17, 19, 21], # Left Arm
        12: [14], 14: [16], 16: [18, 20, 22], # Right Arm
        23: [25], 25: [27], 27: [29, 31],    # Left Leg
        24: [26], 26: [28], 28: [30, 32],    # Right Leg
        11: [12, 23], 12: [24] # Torso
    }
    root_nodes = []
    all_children = set(c for children in parent_child_map.values() for c in children)
    for i in range(len(nodes)):
        if i not in all_children:
            root_nodes.append(i)

    scene.nodes.extend(root_nodes)

    for parent_idx, children_indices in parent_child_map.items():
        for child_idx in children_indices:
            if child_idx < len(gltf.nodes):
                gltf.nodes[parent_idx].children.append(child_idx)

    # --- アニメーションデータ ---
    # 1. タイムスタンプのバッファ
    times_data = np.array(frame_times, dtype=np.float32)
    times_blob = times_data.tobytes()
    gltf.buffers.append(Buffer(byteLength=len(times_blob)))
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=0, byteLength=len(times_blob)))
    times_accessor = Accessor(
        bufferView=0, componentType=FLOAT, count=len(frame_times), type=SCALAR,
        max=[max(frame_times)], min=[min(frame_times)]
    )
    gltf.accessors.append(times_accessor)

    # 2. ランドマークごとのアニメーションチャネルを作成
    animation = Animation()
    gltf.animations.append(animation)

    buffer_offset = len(times_blob)

    for i in range(len(landmark_names)):
        # 各フレームからi番目のランドマークの座標を抽出
        translations = []
        for frame_landmarks in all_world_landmarks:
            lm = frame_landmarks[i]
            # YとZを反転
            translations.append([-lm.x, -lm.y, -lm.z])

        translations_data = np.array(translations, dtype=np.float32)
        translations_blob = translations_data.tobytes()

        # 既存のバッファに追記
        gltf.buffers[0].uri += translations_blob

        buffer_view_index = len(gltf.bufferViews)
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=buffer_offset, byteLength=len(translations_blob)))

        accessor_index = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=buffer_view_index, componentType=FLOAT, count=len(translations), type=VEC3
        ))

        sampler = AnimationSampler(input=0, output=accessor_index, interpolation=LINEAR) # input=0 はタイムアクセサ
        animation.samplers.append(sampler)

        channel = AnimationChannel(sampler=len(animation.samplers)-1, target=AnimationChannelTarget(node=i, path="translation"))
        animation.channels.append(channel)

        buffer_offset += len(translations_blob)

    gltf.buffers[0].byteLength = buffer_offset

    # GLBとして保存
    gltf.convert_buffers(BufferFormat.BINARY)
    gltf.save_binary(output_path)


def main():
    parser = argparse.ArgumentParser(description="ビデオから姿勢推定を行い、結果を保存するスクリプト。")
    parser.add_argument("--input", type=str, required=True, help="入力ビデオファイルのパス。")
    parser.add_argument("--output", type=str, required=True, help="出力ビデオファイルのパス。")
    parser.add_argument("--output-glb", type=str, help="出力GLBアニメーションファイルのパス。")
    parser.add_argument('--model-complexity', type=int, default=2, choices=[0, 1, 2], help="姿勢推定モデルの複雑さ (0: lite, 1: full, 2: heavy)。デフォルト: 2")
    args = parser.parse_args()

    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.output_glb:
        glb_output_dir = os.path.dirname(args.output_glb)
        if glb_output_dir and not os.path.exists(glb_output_dir):
            os.makedirs(glb_output_dir)

    process_video(args.input, args.output, args.output_glb, args.model_complexity)

if __name__ == '__main__':
    main()
