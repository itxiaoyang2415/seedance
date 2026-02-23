import cv2
import numpy as np
import tempfile
import os
import time
from typing import Optional, Callable
from rembg import remove, new_session
from core.video_processor import extract_frames, compose_video

_rembg_session = None


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
        _rembg_session = new_session("isnet-general-use", providers=providers)
    return _rembg_session


def remove_background(frame: np.ndarray) -> np.ndarray:
    """移除单帧背景，返回BGRA图像 (H, W, 4)"""
    session = get_rembg_session()
    result = remove(frame, session=session, post_process_mask=True)
    return result


def replace_background(frame_rgba: np.ndarray, bg_image: np.ndarray) -> np.ndarray:
    """将BGRA前景合成到新背景上，返回BGR图像"""
    h, w = frame_rgba.shape[:2]
    bg_resized = cv2.resize(bg_image, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha = frame_rgba[:, :, 3:4].astype(np.float32) / 255.0
    fg = frame_rgba[:, :, :3].astype(np.float32)
    bg = bg_resized.astype(np.float32)

    result = fg * alpha + bg * (1.0 - alpha)
    return result.astype(np.uint8)


def process_video_background(
    video_path: str,
    bg_image_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """逐帧处理视频背景替换，每一帧都完整抠图替换"""
    if progress_callback:
        progress_callback(0.0, "正在读取视频...")

    frames, fps, audio_path = extract_frames(video_path)
    total_frames = len(frames)

    if progress_callback:
        progress_callback(0.05, f"共 {total_frames} 帧，开始处理...")

    bg_image = cv2.imread(bg_image_path)
    if bg_image is None:
        raise ValueError(f"无法读取背景图片: {bg_image_path}")

    processed_frames = []
    start_time = time.time()

    for i, frame in enumerate(frames):
        frame_rgba = remove_background(frame)
        result_frame = replace_background(frame_rgba, bg_image)
        processed_frames.append(result_frame)

        if progress_callback:
            elapsed = time.time() - start_time
            speed = (i + 1) / max(elapsed, 0.1)
            remaining = (total_frames - i - 1) / max(speed, 0.01)
            mins, secs = divmod(int(remaining), 60)
            progress = 0.05 + 0.85 * ((i + 1) / total_frames)
            progress_callback(
                progress,
                f"处理中: {i + 1}/{total_frames} 帧 预计剩余: {mins}分{secs}秒",
            )

    if progress_callback:
        progress_callback(0.90, "正在合成视频...")

    output_path = tempfile.mktemp(suffix=".mp4")
    compose_video(processed_frames, fps, audio_path, output_path)

    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

    total_time = time.time() - start_time
    mins, secs = divmod(int(total_time), 60)
    if progress_callback:
        progress_callback(1.0, f"处理完成！总耗时: {mins}分{secs}秒")

    return output_path
