import argparse
import base64
import os

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GLB Animation Viewer</title>
    <style>
        body {{ margin: 0; }}
        canvas {{ display: block; }}
    </style>
</head>
<body>
    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@0.164.1/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.164.1/examples/jsm/"
        }}
    }}
    </script>
    <script type="module">
        import * as THREE from 'three';
        import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        let scene, camera, renderer, mixer;
        const clock = new THREE.Clock();

        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x222222);

            // Camera
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 0.5, 2);

            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.body.appendChild(renderer.domElement);

            // Controls
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.target.set(0, 0.5, 0);
            controls.update();

            // Light
            const ambientLight = new THREE.AmbientLight(0xffffff, 1.5);
            scene.add(ambientLight);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 2);
            directionalLight.position.set(1, 1, 1).normalize();
            scene.add(directionalLight);

            // Load Model
            const loader = new GLTFLoader();
            const glbDataUri = `{data_uri}`;

            // The data URI is already a base64 string, so we just need to decode it.
            const decodedData = atob(glbDataUri);
            const arrayBuffer = new ArrayBuffer(decodedData.length);
            const uint8Array = new Uint8Array(arrayBuffer);
            for (let i = 0; i < decodedData.length; i++) {{
                uint8Array[i] = decodedData.charCodeAt(i);
            }}

            loader.parse(arrayBuffer, '', (gltf) => {{
                const model = gltf.scene;
                scene.add(model);

                // Add skeleton helper
                const skeletonHelper = new THREE.SkeletonHelper(model);
                scene.add(skeletonHelper);

                // Animation
                if (gltf.animations && gltf.animations.length) {{
                    mixer = new THREE.AnimationMixer(model);
                    const action = mixer.clipAction(gltf.animations[0]);
                    action.play();
                }}

                animate();
            }}, undefined, (error) => {{
                console.error('An error happened during parsing', error);
            }});
        }}

        function animate() {{
            requestAnimationFrame(animate);
            const delta = clock.getDelta();
            if (mixer) mixer.update(delta);
            renderer.render(scene, camera);
        }}

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        init();
    </script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description="GLBファイルを埋め込んだ自己完結型のHTMLビューワーを生成します。")
    parser.add_argument('input_glb', help="入力GLBファイルのパス")
    parser.add_argument('--output', default='viewer.html', help="出力HTMLファイルのパス (デフォルト: viewer.html)")
    args = parser.parse_args()

    if not os.path.exists(args.input_glb):
        print(f"エラー: 入力ファイルが見つかりません: {args.input_glb}")
        return

    try:
        with open(args.input_glb, 'rb') as f:
            glb_data = f.read()

        base64_data = base64.b64encode(glb_data).decode('utf-8')

        final_html = HTML_TEMPLATE.format(data_uri=base64_data)

        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(final_html)

        print(f"ビューワーを生成しました: {os.path.abspath(args.output)}")
        print("このHTMLファイルをブラウザで開いてください。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == '__main__':
    main()
