import cv2
import numpy as np
import tempfile
import os
import time
from typing import Optional, Callable
from rembg import remove, new_session
from core.video_processor import extract_frames, compose_video

_rembg_session = None

# 抠图处理的最大边长（缩小后处理，大幅提速）
MATTING_MAX_SIZE = 640
# 跳帧间隔：每N帧做一次完整抠图，中间帧用mask插值
SKIP_FRAMES = 3


def _get_providers():
    """自动检测GPU，优先使用CUDA"""
    import onnxruntime as ort
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        providers = _get_providers()
        _rembg_session = new_session("u2netp", providers=providers)
    return _rembg_session


def get_alpha_mask(frame: np.ndarray) -> np.ndarray:
    """对缩小后的帧做抠图，返回原始尺寸的alpha mask (H, W) uint8"""
    h, w = frame.shape[:2]
    # 缩小到 MATTING_MAX_SIZE
    scale = min(MATTING_MAX_SIZE / max(h, w), 1.0)
    if scale < 1.0:
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        small = frame

    session = get_rembg_session()
    result = remove(small, session=session, only_mask=True, post_process_mask=True)

    # 放大mask回原始尺寸
    if scale < 1.0:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)

    return result


def composite_frame(frame: np.ndarray, alpha: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """用alpha mask将前景合成到背景上"""
    h, w = frame.shape[:2]
    bg_resized = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    a = alpha[:, :, np.newaxis].astype(np.float32) / 255.0
    result = frame.astype(np.float32) * a + bg_resized.astype(np.float32) * (1.0 - a)
    return result.astype(np.uint8)


def interpolate_masks(mask_a: np.ndarray, mask_b: np.ndarray, t: float) -> np.ndarray:
    """在两个mask之间线性插值, t in [0,1]"""
    return (mask_a.astype(np.float32) * (1 - t) + mask_b.astype(np.float32) * t).astype(np.uint8)


def process_video_background(
    video_path: str,
    bg_image_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    if progress_callback:
        progress_callback(0.0, "正在读取视频...")

    frames, fps, audio_path = extract_frames(video_path)
    total_frames = len(frames)

    if progress_callback:
        progress_callback(0.05, f"共 {total_frames} 帧，加载背景...")

    bg_image = cv2.imread(bg_image_path)
    if bg_image is None:
        raise ValueError(f"无法读取背景图片: {bg_image_path}")

    # 第一阶段：计算关键帧的alpha mask（每隔SKIP_FRAMES帧）
    keyframe_indices = list(range(0, total_frames, SKIP_FRAMES))
    if keyframe_indices[-1] != total_frames - 1:
        keyframe_indices.append(total_frames - 1)

    keyframe_masks = {}
    start_time = time.time()

    for idx, ki in enumerate(keyframe_indices):
        keyframe_masks[ki] = get_alpha_mask(frames[ki])

        if progress_callback:
            elapsed = time.time() - start_time
            speed = (idx + 1) / max(elapsed, 0.1)
            remaining = (len(keyframe_indices) - idx - 1) / max(speed, 0.01)
            mins, secs = divmod(int(remaining), 60)
            progress = 0.05 + 0.75 * ((idx + 1) / len(keyframe_indices))
            progress_callback(
                progress,
                f"抠图: {idx + 1}/{len(keyframe_indices)} 关键帧 "
                f"(跳帧x{SKIP_FRAMES}) 预计剩余: {mins}分{secs}秒",
            )

    # 第二阶段：插值生成所有帧的mask并合成
    if progress_callback:
        progress_callback(0.80, "正在合成所有帧...")

    processed_frames = []
    for i in range(total_frames):
        if i in keyframe_masks:
            alpha = keyframe_masks[i]
        else:
            # 找前后两个关键帧插值
            prev_ki = (i // SKIP_FRAMES) * SKIP_FRAMES
            next_ki = min(prev_ki + SKIP_FRAMES, total_frames - 1)
            if next_ki not in keyframe_masks:
                next_ki = prev_ki
            if prev_ki == next_ki:
                alpha = keyframe_masks[prev_ki]
            else:
                t = (i - prev_ki) / (next_ki - prev_ki)
                alpha = interpolate_masks(keyframe_masks[prev_ki], keyframe_masks[next_ki], t)

        result = composite_frame(frames[i], alpha, bg_image)
        processed_frames.append(result)

    if progress_callback:
        progress_callback(0.90, "正在编码视频...")

    output_path = tempfile.mktemp(suffix=".mp4")
    compose_video(processed_frames, fps, audio_path, output_path)

    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

    total_time = time.time() - start_time
    mins, secs = divmod(int(total_time), 60)
    if progress_callback:
        progress_callback(1.0, f"处理完成！总耗时: {mins}分{secs}秒")

    return output_path
