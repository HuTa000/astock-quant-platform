"""生成内置提示音 ``assets/alert.wav``（P4-S2）。

用标准库 ``wave`` + ``math`` 生成 0.3 秒 880Hz 正弦短音（含淡入淡出防爆音），
**不依赖任何外部资源 / 网络**。运行方式（在项目根 ``D:\\Stock\\astock`` 下）::

    .venv/Scripts/python.exe scripts/gen_alert_wav.py
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

# 项目根（scripts/ 的上一级）。
ROOT: Path = Path(__file__).resolve().parents[1]
OUT: Path = ROOT / "assets" / "alert.wav"

# 采样参数：单声道 / 16-bit / 44.1kHz，0.3 秒 880Hz。
SAMPLE_RATE: int = 44100
DURATION: float = 0.3
FREQ: float = 880.0
AMP: float = 0.6
FADE: float = 0.02  # 淡入淡出时长（秒），避免起止爆音。


def build_samples(sample_rate: int = SAMPLE_RATE) -> list[int]:
    """生成 16-bit 正弦采样序列（带线性淡入淡出）。

    Args:
        sample_rate: 采样率（Hz）。

    Returns:
        整数采样值列表（范围 [-32767, 32767]）。
    """
    total = int(sample_rate * DURATION)
    fade_n = max(1, int(sample_rate * FADE))
    samples: list[int] = []
    for i in range(total):
        t = i / sample_rate
        # 线性包络：前 / 后 fade_n 帧做淡入淡出。
        if i < fade_n:
            envelope = i / fade_n
        elif i >= total - fade_n:
            envelope = (total - 1 - i) / fade_n
        else:
            envelope = 1.0
        value = AMP * envelope * math.sin(2 * math.pi * FREQ * t)
        samples.append(int(max(-1.0, min(1.0, value)) * 32767))
    return samples


def main() -> int:
    """写 wav 文件到 ``assets/alert.wav``。"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    samples = build_samples()
    with wave.open(str(OUT), "wb") as wav:
        wav.setnchannels(1)       # 单声道
        wav.setsampwidth(2)       # 16-bit
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    print(f"已生成 {OUT}（{len(samples)} 帧，{SAMPLE_RATE}Hz）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
