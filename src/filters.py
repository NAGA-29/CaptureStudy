"""
ランドマーク後処理用の平滑化フィルタ群。

MediaPipe Tasks API（新APIの PoseLandmarker）は legacy Solutions API に
あった `smooth_landmarks` を提供しないため、後段でフィルタを掛けるのが
標準的なプラクティス。ここでは OneEuroFilter を軽量実装する。

参考:
- Géry Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter for
  Noisy Input in Interactive Systems", CHI 2012.
- https://jaantollander.com/post/noise-filtering-using-one-euro-filter/
- https://github.com/google/mediapipe/issues/4507
"""

import math
from dataclasses import dataclass


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    """cutoff（Hz）とサンプリング周期 t_e（秒）から EMA 係数 alpha を求める。"""
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exponential_smoothing(alpha: float, x: float, x_prev: float) -> float:
    return alpha * x + (1.0 - alpha) * x_prev


class OneEuroFilter:
    """単一スカラー値に対する 1€ Filter。

    Parameters
    ----------
    min_cutoff : float
        低速域のカットオフ周波数（Hz）。小さいほど静止時のジッタが減る。
    beta : float
        速度係数。大きいほど高速移動時の追従性が上がる（遅延が減る）。
    d_cutoff : float
        速度ローパスのカットオフ周波数（Hz）。
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.1, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float, t: float) -> float:
        if self._t_prev is None or self._x_prev is None:
            self._t_prev = t
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        t_e = t - self._t_prev
        if t_e <= 0.0:
            # 時刻が進んでいない／逆行しているときは現値を返して破綻を防ぐ。
            return self._x_prev

        # 速度の推定 → d_cutoff でローパス
        dx = (x - self._x_prev) / t_e
        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx_hat = _exponential_smoothing(a_d, dx, self._dx_prev)

        # 速度に応じて cutoff を可変にし、位置を平滑化
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


@dataclass
class SmoothedLandmark:
    """MediaPipe の `NormalizedLandmark` / world landmark と同じ属性を持つ軽量コピー。

    `save_to_glb` / `save_to_html_viewer` / `draw_landmarks_on_image` が
    `.x / .y / .z / .visibility` しか参照しないため、これで差し替え可能。
    """

    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0


class PoseLandmarkSmoother:
    """33 点 × (x, y, z) を一括で管理する OneEuroFilter ラッパ。

    - visibility が閾値未満の点は「前フレームの平滑値を保持」して
      ノイズの多い点が暴れるのを防ぐ（hold-last）。
    - 検出失敗フレームに対しては `hold()` を呼び出すことで、直前値をそのまま
      返して時間軸との整合を取る。
    """

    NUM_LANDMARKS = 33

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.1,
        d_cutoff: float = 1.0,
        visibility_threshold: float = 0.5,
    ):
        self.visibility_threshold = float(visibility_threshold)
        self._filters = [
            [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(3)]
            for _ in range(self.NUM_LANDMARKS)
        ]
        self._last: list[SmoothedLandmark] | None = None

    def reset(self) -> None:
        for axes in self._filters:
            for f in axes:
                f.reset()
        self._last = None

    @property
    def has_history(self) -> bool:
        return self._last is not None

    def smooth(self, landmarks, t: float) -> list[SmoothedLandmark]:
        """`landmarks` は MediaPipe の NormalizedLandmark リスト互換。"""
        out: list[SmoothedLandmark] = []
        for i, lm in enumerate(landmarks):
            vis = float(getattr(lm, "visibility", 1.0))
            presence = float(getattr(lm, "presence", 1.0))

            if vis < self.visibility_threshold and self._last is not None:
                # 低可視性: 前フレームの平滑値を保持してジッタを抑える。
                prev = self._last[i]
                out.append(SmoothedLandmark(prev.x, prev.y, prev.z, vis, presence))
                continue

            fx, fy, fz = self._filters[i]
            x = fx(float(lm.x), t)
            y = fy(float(lm.y), t)
            z = fz(float(lm.z), t)
            out.append(SmoothedLandmark(x, y, z, vis, presence))

        self._last = out
        return out

    def hold(self) -> list[SmoothedLandmark] | None:
        """検出失敗時に前フレームをそのまま再利用する。"""
        if self._last is None:
            return None
        return [SmoothedLandmark(p.x, p.y, p.z, p.visibility, p.presence) for p in self._last]


if __name__ == "__main__":
    # 静止信号 + ガウスノイズでジッタ低減効果の簡易チェック。
    # (モーキャプの主要ジッタケース: 被写体が止まっている時の揺らぎ)
    import random

    random.seed(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.1)
    raw_err = 0.0
    filt_err = 0.0
    fps = 30
    for i in range(300):
        t = i / fps
        truth = 0.0  # 静止
        noisy = truth + random.gauss(0.0, 0.05)
        y = f(noisy, t)
        raw_err += (noisy - truth) ** 2
        filt_err += (y - truth) ** 2
    raw_rmse = math.sqrt(raw_err / 300)
    filt_rmse = math.sqrt(filt_err / 300)
    print(f"RMSE raw={raw_rmse:.4f}  filtered={filt_rmse:.4f}  "
          f"reduction={raw_rmse / filt_rmse:.2f}x")
    assert filt_rmse < raw_rmse, "OneEuroFilter がジッタを削減できていません"
