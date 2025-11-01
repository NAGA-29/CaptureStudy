import argparse
import os
import cv2
import torch
import requests
from tqdm import tqdm

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import get_group_result
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer

# --- Model & Config URLs ---
# Pose Estimator (HRNet-w48 from MMPose)
POSE_CONFIG_URL = 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py'
POSE_CHECKPOINT_URL = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'

# Person Detector (RTMDet-x from MMDetection)
DET_CONFIG_URL = 'https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/rtmdet/rtmdet_x_8xb32-300e_coco.py'
DET_CHECKPOINT_URL = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'

def download_file(url, out_path):
    """Downloads a file from a URL to a given path with a progress bar."""
    if os.path.exists(out_path):
        print(f"File already exists: {out_path}")
        return

    print(f"Downloading {os.path.basename(out_path)} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024

    with open(out_path, 'wb') as file, tqdm(
        desc=os.path.basename(out_path),
        total=total_size_in_bytes,
        unit='iB',
        unit_scale=True,
    ) as progress_bar:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong during download")


import numpy as np
from pygltflib import *

def export_to_glb(all_pose_results, frame_times, output_path, width, height):
    """
    収集したMMPoseのデータからGLBファイルを作成する。
    """
    gltf = GLTF2()
    scene = Scene()
    gltf.scenes.append(scene)

    # COCO-WholeBodyの定義
    keypoint_info = {
        0: dict(name='nose'), 1: dict(name='left_eye'), 2: dict(name='right_eye'), 3: dict(name='left_ear'), 4: dict(name='right_ear'),
        5: dict(name='left_shoulder'), 6: dict(name='right_shoulder'), 7: dict(name='left_elbow'), 8: dict(name='right_elbow'),
        9: dict(name='left_wrist'), 10: dict(name='right_wrist'), 11: dict(name='left_hip'), 12: dict(name='right_hip'),
        13: dict(name='left_knee'), 14: dict(name='right_knee'), 15: dict(name='left_ankle'), 16: dict(name='right_ankle'),
        17: dict(name='left_big_toe'), 18: dict(name='left_small_toe'), 19: dict(name='left_heel'), 20: dict(name='right_big_toe'),
        21: dict(name='right_small_toe'), 22: dict(name='right_heel'),
        # Face (simplified)
        23: dict(name='face_0'), 90: dict(name='face_67'),
        # Hands (simplified)
        91: dict(name='left_hand_root'), 112: dict(name='right_hand_root')
    }
    # 簡単のため、キーポイント名を簡略化
    landmark_names = [f'kpt_{i}' for i in range(133)]
    for i, info in keypoint_info.items():
        if i < len(landmark_names):
            landmark_names[i] = info['name']

    nodes = [Node(name=name) for name in landmark_names]
    gltf.nodes.extend(nodes)

    # スケルトン定義 (主要なもののみ)
    skeleton_info = {
        0: ('left_ankle', 'left_knee'), 1: ('left_knee', 'left_hip'), 2: ('right_ankle', 'right_knee'), 3: ('right_knee', 'right_hip'),
        4: ('left_hip', 'right_hip'), 5: ('left_shoulder', 'left_hip'), 6: ('right_shoulder', 'right_hip'), 7: ('left_shoulder', 'right_shoulder'),
        8: ('left_shoulder', 'left_elbow'), 9: ('right_shoulder', 'right_elbow'), 10: ('left_elbow', 'left_wrist'), 11: ('right_elbow', 'right_wrist'),
    }

    # ノードの親子関係を設定
    name_to_id = {name: i for i, name in enumerate(landmark_names)}
    parent_child_map = {}
    for _, (p_name, c_name) in skeleton_info.items():
        if p_name in name_to_id and c_name in name_to_id:
            p_id, c_id = name_to_id[p_name], name_to_id[c_name]
            if p_id not in parent_child_map:
                parent_child_map[p_id] = []
            parent_child_map[p_id].append(c_id)

    root_nodes_indices = set(range(len(nodes))) - set(c for children in parent_child_map.values() for c in children)
    scene.nodes.extend(list(root_nodes_indices))
    for parent, children in parent_child_map.items():
        gltf.nodes[parent].children.extend(children)

    # --- アニメーションデータ ---
    animation = Animation()
    gltf.animations.append(animation)
    binary_blob = b''

    times_data = np.array(frame_times, dtype=np.float32)
    times_blob = times_data.tobytes()

    buffer_view_offset = len(binary_blob)
    binary_blob += times_blob
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=buffer_view_offset, byteLength=len(times_blob)))
    times_accessor_idx = len(gltf.accessors)
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews) - 1, componentType=FLOAT, count=len(frame_times), type=SCALAR,
        max=[float(np.max(times_data))], min=[float(np.min(times_data))]
    ))

    for i in range(len(landmark_names)):
        translations = []
        for pose_result in all_pose_results:
            # 最初の人物のキーポイントを取得
            keypoints = pose_result[0].pred_instances.keypoints[0]
            if i < len(keypoints):
                x, y = keypoints[i]
                # 2D座標をGLBの3D座標に変換 (Zは0に)
                # Y軸を反転, 中央を原点に
                x_norm = (x - width / 2) / width
                y_norm = -(y - height / 2) / height
                translations.append([x_norm, y_norm, 0.0])
            else:
                translations.append([0,0,0]) # データがない場合は原点

        translations_data = np.array(translations, dtype=np.float32)
        translations_blob = translations_data.tobytes()

        buffer_view_offset = len(binary_blob)
        binary_blob += translations_blob
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=buffer_view_offset, byteLength=len(translations_blob)))

        translations_accessor_idx = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=len(gltf.bufferViews) - 1, componentType=FLOAT, count=len(translations_data), type=VEC3,
            max=np.max(translations_data, axis=0).tolist(), min=np.min(translations_data, axis=0).tolist()
        ))

        sampler = AnimationSampler(input=times_accessor_idx, output=translations_accessor_idx, interpolation=LINEAR)
        animation.samplers.append(sampler)

        channel = AnimationChannel(sampler=len(animation.samplers)-1, target=AnimationChannelTarget(node=i, path="translation"))
        animation.channels.append(channel)

    gltf.buffers.append(Buffer(byteLength=len(binary_blob)))
    gltf.set_binary_blob(binary_blob)
    gltf.save_binary(output_path)
    print(f"GLBファイルを出力しました: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="MMPoseを使ってビデオから高精度な姿勢推定を行います。")
    parser.add_argument('input_video', help="入力ビデオファイルのパス")
    parser.add_argument('output_video', help="骨格を描画した出力ビデオファイルのパス")
    parser.add_argument('--output-glb', help="3Dモーションデータを保存するGLBファイルのパス。")
    parser.add_argument('--device', default='cpu', help="推論に使用するデバイス (例: 'cpu', 'cuda:0')")
    args = parser.parse_args()

    # --- 1. 環境設定とモデルのダウンロード ---
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    pose_config_path = os.path.join(models_dir, os.path.basename(POSE_CONFIG_URL))
    pose_checkpoint_path = os.path.join(models_dir, os.path.basename(POSE_CHECKPOINT_URL))
    det_config_path = os.path.join(models_dir, os.path.basename(DET_CONFIG_URL))
    det_checkpoint_path = os.path.join(models_dir, os.path.basename(DET_CHECKPOINT_URL))

    download_file(POSE_CONFIG_URL, pose_config_path)
    download_file(POSE_CHECKPOINT_URL, pose_checkpoint_path)
    download_file(DET_CONFIG_URL, det_config_path)
    download_file(DET_CHECKPOINT_URL, det_checkpoint_path)

    # --- 2. モデルの初期化 ---
    print("モデルを初期化しています...")
    pose_estimator = init_pose_estimator(
        pose_config_path,
        pose_checkpoint_path,
        device=args.device,
        cfg_options={'model': {'detector': 'human'}}
    )

    # --- 3. ビデオ処理 ---
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"エラー: ビデオファイルを開けません: {args.input_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    visualizer = PoseLocalVisualizer()
    all_pose_results = []
    frame_times = []
    frame_count = 0

    print("ビデオ処理を開始します...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 推論の実行
        pose_results = inference_topdown(pose_estimator, frame)
        all_pose_results.append(pose_results)
        frame_times.append(frame_count / fps)
        frame_count += 1

        # 描画
        # 最初の人物のみ描画
        if pose_results:
            visualizer.add_datasample('result', frame, data_sample=pose_results[0])
            vis_frame = visualizer.get_image()
            out.write(vis_frame)
        else:
            out.write(frame)

    print(f"処理が完了しました。出力ビデオ: {args.output_video}")

    # GLB出力
    if args.output_glb and all_pose_results:
        export_to_glb(all_pose_results, frame_times, args.output_glb, width, height)
    elif args.output_glb:
        print("警告: ランドマークが検出されなかったため、GLBファイルは出力されません。")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
