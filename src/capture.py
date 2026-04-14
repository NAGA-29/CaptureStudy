import cv2
import mediapipe as mp
import argparse
import os
import numpy as np
import pygltflib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from filters import PoseLandmarkSmoother


def draw_landmarks_on_image(rgb_image, pose_landmarks_list, visibility_threshold=0.0):
    """OpenCVを使って、画像にランドマークと接続線を描画する。

    Parameters
    ----------
    pose_landmarks_list : list
        `NormalizedLandmark` または `SmoothedLandmark` のリストのリスト。
    visibility_threshold : float
        これ未満の可視性を持つランドマークは薄く描画する。
    """
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    for pose_landmarks in pose_landmarks_list:
        for landmark in pose_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            vis = float(getattr(landmark, "visibility", 1.0))
            if vis < visibility_threshold:
                # 信頼度が低い点は暗めの色で描画して視覚的に区別する。
                cv2.circle(annotated_image, (x, y), 3, (120, 120, 120), -1)
            else:
                cv2.circle(annotated_image, (x, y), 5, (245, 117, 66), -1)

    return annotated_image


def preprocess_frame(frame_rgb, enable_clahe=False):
    """暗所・低コントラスト動画向けの前処理。

    LAB 色空間の L チャネルに CLAHE をかけ、色相を壊さずに局所コントラストを
    改善する。検出器の事前処理（リサイズ等）は MediaPipe 側で行うため、
    ここではコントラストのみを扱う。
    """
    if not enable_clahe:
        return frame_rgb
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def process_video(model_path, input_path, output_path, output_glb_path=None,
                   min_detection_confidence=0.6, min_tracking_confidence=0.6,
                   min_presence_confidence=0.6,
                   num_poses=1, smooth_landmarks=True,
                   smooth_min_cutoff=1.0, smooth_beta=0.1,
                   visibility_threshold=0.5,
                   max_hold_frames=10, enable_clahe=False):
    """
    指定されたビデオファイルを読み込み、新しいMediaPipe Tasks APIで姿勢推定を行い、
    結果を新しいビデオファイルとオプションでGLBファイルに保存する。

    精度向上のため以下の後処理を掛ける:
      - OneEuroFilter による各ランドマーク座標の時間方向平滑化。
      - visibility が閾値未満の点は前フレーム値を保持（hold-last）。
      - 検出失敗フレームは直前フレームを再利用してタイムラインを維持。
      - CLAHE による入力フレームの局所コントラスト補正（オプション）。
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

    print(f"精度設定:")
    print(f"  - 検出信頼度閾値: {min_detection_confidence}")
    print(f"  - プレゼンス信頼度閾値: {min_presence_confidence}")
    print(f"  - トラッキング信頼度閾値: {min_tracking_confidence}")
    print(f"  - 検出人数上限: {num_poses}")
    print(f"  - スムージング: {'有効' if smooth_landmarks else '無効'}"
          f" (min_cutoff={smooth_min_cutoff}, beta={smooth_beta})")
    print(f"  - 可視性しきい値: {visibility_threshold}")
    print(f"  - 欠損保持フレーム上限: {max_hold_frames}")
    print(f"  - CLAHE 前処理: {'有効' if enable_clahe else '無効'}")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_segmentation_masks=False,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        num_poses=num_poses
    )

    # world 用と画面座標用でフィルタ状態を分ける。
    world_smoother = PoseLandmarkSmoother(
        min_cutoff=smooth_min_cutoff, beta=smooth_beta,
        visibility_threshold=visibility_threshold,
    )
    image_smoother = PoseLandmarkSmoother(
        min_cutoff=smooth_min_cutoff, beta=smooth_beta,
        visibility_threshold=visibility_threshold,
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
        if fps <= 0:
            fps = 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_world_landmarks = []
        frame_times = []
        frame_count = 0
        missed_streak = 0
        total_missed = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"処理を開始します... (総フレーム数: {total_frames})")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # BGRからRGBに変換 + オプションで CLAHE による前処理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_for_detect = preprocess_frame(frame_rgb, enable_clahe=enable_clahe)

            # MediaPipe Imageオブジェクトに変換
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_for_detect)

            # タイムスタンプは単調増加が必須なのでフレーム番号から計算する
            # (CAP_PROP_POS_MSEC は一部コンテナで 0 を返すことがある)
            t_sec = frame_count / fps
            timestamp_ms = int(t_sec * 1000)

            # 姿勢推定を実行
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # --- 後段スムージング ---
            world_raw = (detection_result.pose_world_landmarks[0]
                         if detection_result.pose_world_landmarks else None)
            image_raw = (detection_result.pose_landmarks[0]
                         if detection_result.pose_landmarks else None)

            if world_raw is not None:
                missed_streak = 0
                if smooth_landmarks:
                    world_smoothed = world_smoother.smooth(world_raw, t_sec)
                else:
                    world_smoothed = list(world_raw)
            else:
                missed_streak += 1
                total_missed += 1
                # 直前フレームを保持して時間軸のズレを防ぐ（連続失敗が上限内の場合のみ）
                if (smooth_landmarks and world_smoother.has_history
                        and missed_streak <= max_hold_frames):
                    world_smoothed = world_smoother.hold(t_sec)
                else:
                    world_smoothed = None

            if image_raw is not None and smooth_landmarks:
                image_smoothed = image_smoother.smooth(image_raw, t_sec)
            elif image_raw is not None:
                image_smoothed = list(image_raw)
            elif (smooth_landmarks and image_smoother.has_history
                    and missed_streak <= max_hold_frames):
                image_smoothed = image_smoother.hold(t_sec)
            else:
                image_smoothed = None

            # --- GLB 用のランドマーク蓄積 ---
            if world_smoothed is not None:
                all_world_landmarks.append(world_smoothed)
                frame_times.append(t_sec)

            # --- 描画 ---
            landmarks_for_draw = [image_smoothed] if image_smoothed is not None else []
            annotated_image = draw_landmarks_on_image(
                frame_rgb, landmarks_for_draw, visibility_threshold=visibility_threshold,
            )

            # RGBからBGRに戻して出力
            out.write(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            frame_count += 1

            # 進捗表示（10フレームごと）
            if frame_count % 10 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"\r処理中: {frame_count}/{total_frames} フレーム ({progress:.1f}%)", end="", flush=True)

        print()  # 改行
        print(f"処理が完了しました。出力ファイル: {output_path}")
        if total_missed > 0:
            print(f"  - 検出失敗: {total_missed}/{frame_count} フレーム (補間/スキップ済み)")

        # --- GLBファイルのエクスポート ---
        if output_glb_path and all_world_landmarks:
            print(f"GLBファイルを出力します: {output_glb_path}")
            save_to_glb(all_world_landmarks, frame_times, output_glb_path)
            
            # HTMLビューアーも生成
            html_path = output_glb_path.replace('.glb', '.html')
            save_to_html_viewer(all_world_landmarks, frame_times, fps, html_path)
        elif output_glb_path:
            print("警告: ランドマークが検出されなかったため、GLBファイルは出力されません。")

        # --- 後処理 ---
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# MediaPipe Pose の接続定義（スケルトンの骨格線）
POSE_CONNECTIONS = [
    # 顔
    (0, 1), (1, 2), (2, 3), (3, 7),  # 左目
    (0, 4), (4, 5), (5, 6), (6, 8),  # 右目
    (9, 10),  # 口
    # 胴体
    (11, 12),  # 肩
    (11, 23), (12, 24),  # 肩から腰
    (23, 24),  # 腰
    # 左腕
    (11, 13), (13, 15),  # 肩→肘→手首
    (15, 17), (15, 19), (15, 21), (17, 19),  # 手
    # 右腕
    (12, 14), (14, 16),  # 肩→肘→手首
    (16, 18), (16, 20), (16, 22), (18, 20),  # 手
    # 左脚
    (23, 25), (25, 27),  # 腰→膝→足首
    (27, 29), (27, 31), (29, 31),  # 足
    # 右脚
    (24, 26), (26, 28),  # 腰→膝→足首
    (28, 30), (28, 32), (30, 32),  # 足
]


def save_to_glb(all_world_landmarks, frame_times, output_path):
    """収集した3DランドマークデータからGLBファイルを作成する（スケルトン表示対応）"""
    gltf = pygltflib.GLTF2()
    scene = pygltflib.Scene()
    gltf.scenes.append(scene)
    gltf.scene = 0

    num_landmarks = 33
    num_frames = len(all_world_landmarks)

    # --- 座標変換したランドマークデータを準備 ---
    # MediaPipeからGLTF座標系へ変換（Y軸反転で正しい向きに）
    converted_landmarks = []
    for frame_landmarks in all_world_landmarks:
        frame_data = []
        for lm in frame_landmarks:
            # X: 反転, Y: 反転（上下を正しく）, Z: 反転
            frame_data.append([-lm.x, -lm.y, -lm.z])
        converted_landmarks.append(frame_data)

    # バッファデータの準備
    all_blob_data = b''
    buffer_offset = 0

    # --- 1. ジョイント（球体）用のメッシュデータ ---
    joint_radius = 0.015
    octahedron_vertices = np.array([
        [0, joint_radius, 0],
        [0, -joint_radius, 0],
        [joint_radius, 0, 0],
        [-joint_radius, 0, 0],
        [0, 0, joint_radius],
        [0, 0, -joint_radius],
    ], dtype=np.float32)

    octahedron_indices = np.array([
        0, 2, 4,  0, 4, 3,  0, 3, 5,  0, 5, 2,
        1, 4, 2,  1, 3, 4,  1, 5, 3,  1, 2, 5,
    ], dtype=np.uint16)

    joint_vertices_blob = octahedron_vertices.tobytes()
    joint_vertices_view_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=buffer_offset, byteLength=len(joint_vertices_blob),
        target=pygltflib.ARRAY_BUFFER
    ))
    joint_vertices_accessor_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=joint_vertices_view_idx, componentType=pygltflib.FLOAT,
        count=len(octahedron_vertices), type=pygltflib.VEC3,
        max=np.max(octahedron_vertices, axis=0).tolist(),
        min=np.min(octahedron_vertices, axis=0).tolist()
    ))
    all_blob_data += joint_vertices_blob
    buffer_offset += len(joint_vertices_blob)

    joint_indices_blob = octahedron_indices.tobytes()
    joint_indices_view_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=buffer_offset, byteLength=len(joint_indices_blob),
        target=pygltflib.ELEMENT_ARRAY_BUFFER
    ))
    joint_indices_accessor_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=joint_indices_view_idx, componentType=pygltflib.UNSIGNED_SHORT,
        count=len(octahedron_indices), type=pygltflib.SCALAR,
        max=[int(np.max(octahedron_indices))], min=[int(np.min(octahedron_indices))]
    ))
    all_blob_data += joint_indices_blob
    buffer_offset += len(joint_indices_blob)

    # --- 2. スケルトンライン用のメッシュデータ（すべての接続を1つのメッシュに） ---
    # 最初のフレームの位置でベース頂点を作成
    first_frame = converted_landmarks[0]
    num_connections = len(POSE_CONNECTIONS)
    
    # 各接続は2つの頂点（始点と終点）
    skeleton_vertices = []
    skeleton_indices = []
    for conn_idx, (start_idx, end_idx) in enumerate(POSE_CONNECTIONS):
        skeleton_vertices.append(first_frame[start_idx])
        skeleton_vertices.append(first_frame[end_idx])
        skeleton_indices.extend([conn_idx * 2, conn_idx * 2 + 1])
    
    skeleton_vertices = np.array(skeleton_vertices, dtype=np.float32)
    skeleton_indices = np.array(skeleton_indices, dtype=np.uint16)

    skeleton_vertices_blob = skeleton_vertices.tobytes()
    skeleton_vertices_view_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=buffer_offset, byteLength=len(skeleton_vertices_blob),
        target=pygltflib.ARRAY_BUFFER
    ))
    skeleton_vertices_accessor_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=skeleton_vertices_view_idx, componentType=pygltflib.FLOAT,
        count=len(skeleton_vertices), type=pygltflib.VEC3,
        max=np.max(skeleton_vertices, axis=0).tolist(),
        min=np.min(skeleton_vertices, axis=0).tolist()
    ))
    all_blob_data += skeleton_vertices_blob
    buffer_offset += len(skeleton_vertices_blob)

    skeleton_indices_blob = skeleton_indices.tobytes()
    skeleton_indices_view_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=buffer_offset, byteLength=len(skeleton_indices_blob),
        target=pygltflib.ELEMENT_ARRAY_BUFFER
    ))
    skeleton_indices_accessor_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=skeleton_indices_view_idx, componentType=pygltflib.UNSIGNED_SHORT,
        count=len(skeleton_indices), type=pygltflib.SCALAR,
        max=[int(np.max(skeleton_indices))], min=[int(np.min(skeleton_indices))]
    ))
    all_blob_data += skeleton_indices_blob
    buffer_offset += len(skeleton_indices_blob)

    # --- 3. モーフターゲット（各フレームの頂点位置）---
    morph_accessors = []
    for frame_idx in range(1, num_frames):  # フレーム0はベース
        frame = converted_landmarks[frame_idx]
        base_frame = converted_landmarks[0]
        
        # モーフターゲットは差分で指定
        morph_vertices = []
        for conn_idx, (start_idx, end_idx) in enumerate(POSE_CONNECTIONS):
            start_diff = np.array(frame[start_idx]) - np.array(base_frame[start_idx])
            end_diff = np.array(frame[end_idx]) - np.array(base_frame[end_idx])
            morph_vertices.append(start_diff.tolist())
            morph_vertices.append(end_diff.tolist())
        
        morph_vertices = np.array(morph_vertices, dtype=np.float32)
        morph_blob = morph_vertices.tobytes()
        
        morph_view_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=buffer_offset, byteLength=len(morph_blob),
            target=pygltflib.ARRAY_BUFFER
        ))
        morph_accessor_idx = len(gltf.accessors)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=morph_view_idx, componentType=pygltflib.FLOAT,
            count=len(morph_vertices), type=pygltflib.VEC3,
            max=np.max(morph_vertices, axis=0).tolist(),
            min=np.min(morph_vertices, axis=0).tolist()
        ))
        morph_accessors.append(morph_accessor_idx)
        all_blob_data += morph_blob
        buffer_offset += len(morph_blob)

    # --- 4. マテリアルの作成 ---
    gltf.materials.append(pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[1.0, 0.3, 0.3, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.5
        ),
        name="JointMaterial"
    ))
    joint_material_idx = 0

    gltf.materials.append(pygltflib.Material(
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=[0.3, 0.8, 1.0, 1.0],
            metallicFactor=0.0,
            roughnessFactor=0.5
        ),
        name="BoneMaterial"
    ))
    bone_material_idx = 1

    # --- 5. メッシュの作成 ---
    # ジョイント用メッシュ
    gltf.meshes.append(pygltflib.Mesh(
        primitives=[pygltflib.Primitive(
            attributes=pygltflib.Attributes(POSITION=joint_vertices_accessor_idx),
            indices=joint_indices_accessor_idx,
            material=joint_material_idx
        )],
        name="JointMesh"
    ))
    joint_mesh_idx = 0

    # スケルトンライン用メッシュ（モーフターゲット付き）
    skeleton_primitive = pygltflib.Primitive(
        attributes=pygltflib.Attributes(POSITION=skeleton_vertices_accessor_idx),
        indices=skeleton_indices_accessor_idx,
        material=bone_material_idx,
        mode=pygltflib.LINES,
        targets=[{"POSITION": acc_idx} for acc_idx in morph_accessors] if morph_accessors else None
    )
    gltf.meshes.append(pygltflib.Mesh(
        primitives=[skeleton_primitive],
        name="SkeletonMesh",
        weights=[0.0] * len(morph_accessors) if morph_accessors else None
    ))
    skeleton_mesh_idx = 1

    # --- 6. ノードの作成 ---
    root_node_idx = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(name="Skeleton"))
    scene.nodes = [root_node_idx]

    # ジョイントノード
    joint_node_indices = []
    for i in range(num_landmarks):
        node_idx = len(gltf.nodes)
        joint_node_indices.append(node_idx)
        gltf.nodes.append(pygltflib.Node(
            name=f"Joint_{i}",
            mesh=joint_mesh_idx,
            translation=first_frame[i]
        ))

    # スケルトンラインノード
    skeleton_node_idx = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(
        name="SkeletonLines",
        mesh=skeleton_mesh_idx
    ))

    gltf.nodes[root_node_idx].children = joint_node_indices + [skeleton_node_idx]

    # --- 7. アニメーションの作成 ---
    times_data = np.array(frame_times, dtype=np.float32)
    times_blob = times_data.tobytes()
    times_view_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=0, byteOffset=buffer_offset, byteLength=len(times_blob)
    ))
    times_accessor_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=times_view_idx, componentType=pygltflib.FLOAT,
        count=len(frame_times), type=pygltflib.SCALAR,
        max=[float(np.max(times_data))], min=[float(np.min(times_data))]
    ))
    all_blob_data += times_blob
    buffer_offset += len(times_blob)

    animation = pygltflib.Animation(name="PoseAnimation")
    gltf.animations.append(animation)

    # ジョイントのトランスレーションアニメーション
    for i in range(num_landmarks):
        translations = np.array([frame[i] for frame in converted_landmarks], dtype=np.float32)
        trans_blob = translations.tobytes()
        
        trans_view_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=buffer_offset, byteLength=len(trans_blob)
        ))
        trans_accessor_idx = len(gltf.accessors)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=trans_view_idx, componentType=pygltflib.FLOAT,
            count=len(translations), type=pygltflib.VEC3,
            max=np.max(translations, axis=0).tolist(),
            min=np.min(translations, axis=0).tolist()
        ))
        all_blob_data += trans_blob
        buffer_offset += len(trans_blob)

        sampler_idx = len(animation.samplers)
        animation.samplers.append(pygltflib.AnimationSampler(
            input=times_accessor_idx,
            output=trans_accessor_idx,
            interpolation=pygltflib.LINEAR
        ))
        animation.channels.append(pygltflib.AnimationChannel(
            sampler=sampler_idx,
            target=pygltflib.AnimationChannelTarget(
                node=joint_node_indices[i],
                path="translation"
            )
        ))

    # スケルトンラインのモーフターゲットウェイトアニメーション
    if morph_accessors:
        # 各フレームのウェイト配列（1つのフレームだけ1.0、他は0.0）
        weights_per_frame = []
        for frame_idx in range(num_frames):
            weights = [0.0] * len(morph_accessors)
            if frame_idx > 0:
                # フレーム0はベースなのでウェイト設定不要
                # frame_idx=1はmorph_accessors[0]、frame_idx=2はmorph_accessors[1]...
                weights[frame_idx - 1] = 1.0
            weights_per_frame.append(weights)
        
        weights_data = np.array(weights_per_frame, dtype=np.float32)
        weights_blob = weights_data.tobytes()
        
        weights_view_idx = len(gltf.bufferViews)
        gltf.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=buffer_offset, byteLength=len(weights_blob)
        ))
        weights_accessor_idx = len(gltf.accessors)
        gltf.accessors.append(pygltflib.Accessor(
            bufferView=weights_view_idx, componentType=pygltflib.FLOAT,
            count=num_frames, type=pygltflib.SCALAR if len(morph_accessors) == 1 else f"VEC{min(len(morph_accessors), 4)}",
        ))
        all_blob_data += weights_blob
        buffer_offset += len(weights_blob)
        
        # Note: GLTFのweightsアニメーションは複雑なので、
        # 代わりにシンプルなアプローチとしてHTMLビューアーを推奨

    # --- 8. バッファの設定と保存 ---
    gltf.buffers.append(pygltflib.Buffer(byteLength=len(all_blob_data)))
    gltf.set_binary_blob(all_blob_data)
    gltf.save_binary(output_path)
    print(f"GLBファイルを保存しました: {output_path}")
    print(f"  - ジョイント数: {num_landmarks}")
    print(f"  - ボーン数: {len(POSE_CONNECTIONS)}")
    print(f"  - フレーム数: {num_frames}")


def save_to_html_viewer(all_world_landmarks, frame_times, fps, output_path):
    """Three.jsベースのHTMLビューアーを生成する"""
    import json
    
    # フレームデータを変換
    frames = []
    for frame_landmarks in all_world_landmarks:
        frame = []
        for lm in frame_landmarks:
            # MediaPipeからThree.js座標系へ変換（Y軸反転）
            frame.append([-lm.x, -lm.y, -lm.z])
        frames.append(frame)
    
    motion_data = {
        "fps": fps,
        "frames": frames
    }
    
    html_content = HTML_VIEWER_TEMPLATE.replace(
        '__MOTION_DATA__', json.dumps(motion_data)
    ).replace(
        '__CONNECTIONS__', json.dumps(POSE_CONNECTIONS)
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTMLビューアーを保存しました: {output_path}")


# HTMLビューアーテンプレート
HTML_VIEWER_TEMPLATE = '''<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Motion Capture Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: Arial, sans-serif; 
            background: #1a1a2e;
            color: white;
            overflow: hidden;
        }
        #container { width: 100vw; height: 100vh; }
        #controls {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            padding: 15px 25px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        #controls button {
            background: #4a90d9;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        #controls button:hover { background: #5aa0e9; }
        #timeline { width: 400px; cursor: pointer; }
        #frameInfo { min-width: 120px; text-align: center; }
        #info {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 10px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <div>フレーム数: <span id="totalFrames">0</span></div>
        <div>FPS: <span id="fps">0</span></div>
        <div>ドラッグで回転 / スクロールでズーム</div>
    </div>
    <div id="controls">
        <button id="playBtn">▶ 再生</button>
        <button id="resetBtn">⟲ リセット</button>
        <input type="range" id="timeline" min="0" max="100" value="0">
        <span id="frameInfo">0 / 0</span>
        <input type="range" id="speedSlider" min="0.1" max="3" step="0.1" value="1" style="width:80px;">
        <span id="speedInfo">1.0x</span>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        const motionData = __MOTION_DATA__;
        const connections = __CONNECTIONS__;
        
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 100);
        camera.position.set(0, 0, 2);
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        container.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 7);
        scene.add(directionalLight);
        
        const gridHelper = new THREE.GridHelper(2, 20, 0x444444, 0x333333);
        gridHelper.rotation.x = Math.PI / 2;
        scene.add(gridHelper);
        
        const jointGeometry = new THREE.SphereGeometry(0.015, 16, 16);
        const jointMaterial = new THREE.MeshPhongMaterial({ color: 0xff5555 });
        const joints = [];
        
        for (let i = 0; i < 33; i++) {
            const joint = new THREE.Mesh(jointGeometry, jointMaterial);
            joints.push(joint);
            scene.add(joint);
        }
        
        const boneMaterial = new THREE.LineBasicMaterial({ color: 0x55aaff, linewidth: 2 });
        const bones = [];
        
        for (const [start, end] of connections) {
            const geometry = new THREE.BufferGeometry();
            const positions = new Float32Array(6);
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const line = new THREE.Line(geometry, boneMaterial);
            bones.push({ line, start, end });
            scene.add(line);
        }
        
        let currentFrame = 0;
        let isPlaying = false;
        let playbackSpeed = 1.0;
        let lastTime = 0;
        const totalFrames = motionData.frames.length;
        const fps = motionData.fps || 30;
        
        document.getElementById('totalFrames').textContent = totalFrames;
        document.getElementById('fps').textContent = fps.toFixed(1);
        document.getElementById('timeline').max = totalFrames - 1;
        
        function updateFrame(frameIndex) {
            const frame = motionData.frames[frameIndex];
            if (!frame) return;
            
            for (let i = 0; i < 33; i++) {
                const [x, y, z] = frame[i];
                joints[i].position.set(x, y, z);
            }
            
            for (const bone of bones) {
                const startPos = frame[bone.start];
                const endPos = frame[bone.end];
                const positions = bone.line.geometry.attributes.position.array;
                positions[0] = startPos[0]; positions[1] = startPos[1]; positions[2] = startPos[2];
                positions[3] = endPos[0]; positions[4] = endPos[1]; positions[5] = endPos[2];
                bone.line.geometry.attributes.position.needsUpdate = true;
            }
            
            document.getElementById('frameInfo').textContent = `${frameIndex + 1} / ${totalFrames}`;
            document.getElementById('timeline').value = frameIndex;
        }
        
        updateFrame(0);
        
        function animate(time) {
            requestAnimationFrame(animate);
            
            if (isPlaying) {
                const deltaTime = (time - lastTime) / 1000;
                if (deltaTime >= (1 / fps) / playbackSpeed) {
                    currentFrame = (currentFrame + 1) % totalFrames;
                    updateFrame(currentFrame);
                    lastTime = time;
                }
            }
            
            controls.update();
            renderer.render(scene, camera);
        }
        
        requestAnimationFrame(animate);
        
        document.getElementById('playBtn').addEventListener('click', function() {
            isPlaying = !isPlaying;
            this.textContent = isPlaying ? '⏸ 停止' : '▶ 再生';
            lastTime = performance.now();
        });
        
        document.getElementById('resetBtn').addEventListener('click', () => {
            currentFrame = 0;
            updateFrame(0);
            camera.position.set(0, 0, 2);
            controls.reset();
        });
        
        document.getElementById('timeline').addEventListener('input', (e) => {
            currentFrame = parseInt(e.target.value);
            updateFrame(currentFrame);
        });
        
        document.getElementById('speedSlider').addEventListener('input', (e) => {
            playbackSpeed = parseFloat(e.target.value);
            document.getElementById('speedInfo').textContent = playbackSpeed.toFixed(1) + 'x';
        });
        
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                document.getElementById('playBtn').click();
                e.preventDefault();
            } else if (e.code === 'ArrowRight') {
                currentFrame = Math.min(currentFrame + 1, totalFrames - 1);
                updateFrame(currentFrame);
            } else if (e.code === 'ArrowLeft') {
                currentFrame = Math.max(currentFrame - 1, 0);
                updateFrame(currentFrame);
            }
        });
    </script>
</body>
</html>
'''


def main():
    parser = argparse.ArgumentParser(description="ビデオから姿勢推定を行い、結果を保存するスクリプト。")
    parser.add_argument("--model", type=str, default="pose_landmarker_heavy.task", help="使用するMediaPipeモデルファイル (.task) のパス。")
    parser.add_argument("--input", type=str, required=True, help="入力ビデオファイルのパス。")
    parser.add_argument("--output", type=str, required=True, help="出力ビデオファイルのパス。")
    parser.add_argument("--output-glb", type=str, help="出力GLBアニメーションファイルのパス。")
    
    # 精度向上オプション
    parser.add_argument("--min-detection-confidence", type=float, default=0.6,
                        help="検出信頼度の閾値 (0.0-1.0)。高いほど検出が厳格になります。デフォルト: 0.6")
    parser.add_argument("--min-presence-confidence", type=float, default=0.6,
                        help="ポーズ存在信頼度の閾値 (0.0-1.0)。デフォルト: 0.6")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.6,
                        help="トラッキング信頼度の閾値 (0.0-1.0)。高いほどトラッキングが厳格になります。デフォルト: 0.6")
    parser.add_argument("--num-poses", type=int, default=1,
                        help="検出する人数の上限。デフォルト: 1")
    parser.add_argument("--no-smooth", action="store_true",
                        help="OneEuroFilter によるランドマーク平滑化を無効にする")
    parser.add_argument("--smooth-min-cutoff", type=float, default=1.0,
                        help="OneEuroFilter の min_cutoff (Hz)。下げると静止時のジッタが減る。デフォルト: 1.0")
    parser.add_argument("--smooth-beta", type=float, default=0.1,
                        help="OneEuroFilter の beta。上げると高速移動時の追従性が増す。デフォルト: 0.1")
    parser.add_argument("--visibility-threshold", type=float, default=0.5,
                        help="この値未満の可視性を持つランドマークは前フレーム値を保持する。デフォルト: 0.5")
    parser.add_argument("--max-hold-frames", type=int, default=10,
                        help="検出失敗が連続した場合に前フレームを再利用する最大フレーム数。デフォルト: 10")
    parser.add_argument("--clahe", action="store_true",
                        help="入力フレームに CLAHE を適用して低コントラスト動画での検出率を上げる")
    args = parser.parse_args()

    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.output_glb:
        glb_output_dir = os.path.dirname(args.output_glb)
        if glb_output_dir and not os.path.exists(glb_output_dir):
            os.makedirs(glb_output_dir)

    process_video(
        args.model, args.input, args.output, args.output_glb,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        min_presence_confidence=args.min_presence_confidence,
        num_poses=args.num_poses,
        smooth_landmarks=not args.no_smooth,
        smooth_min_cutoff=args.smooth_min_cutoff,
        smooth_beta=args.smooth_beta,
        visibility_threshold=args.visibility_threshold,
        max_hold_frames=args.max_hold_frames,
        enable_clahe=args.clahe,
    )

if __name__ == '__main__':
    main()
