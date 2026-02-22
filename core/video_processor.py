import cv2
import subprocess
import tempfile
import os
import numpy as np
from typing import List, Tuple, Optional


def extract_frames(video_path: str) -> Tuple[List[np.ndarray], float, Optional[str]]:
    """
    从视频中提取所有帧和音频轨道。

    Returns:
        (frames, fps, audio_path)
        - frames: BGR numpy数组列表
        - fps: 帧率
        - audio_path: 音频临时文件路径，无音频则为None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("视频中没有读取到任何帧")

    # 用 FFmpeg 提取音频
    audio_path = None
    try:
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            video_path,
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)

        if result.stdout.strip():
            audio_path = tempfile.mktemp(suffix=".aac")
            extract_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "aac",
                "-b:a", "128k",
                audio_path,
            ]
            subprocess.run(extract_cmd, capture_output=True, timeout=60, check=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        audio_path = None

    return frames, fps, audio_path


def compose_video(
    frames: List[np.ndarray],
    fps: float,
    audio_path: Optional[str],
    output_path: Optional[str] = None,
) -> str:
    """将帧序列合成为视频，并合并音频。返回输出视频路径。"""
    if not frames:
        raise ValueError("帧列表为空")

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".mp4")

    h, w = frames[0].shape[:2]

    # 用 OpenCV 写入临时无音频视频
    temp_video = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError("无法创建视频写入器，请检查编解码器")

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        writer.write(frame)

    writer.release()

    # 用 FFmpeg 转码为 H.264 并合并音频
    if audio_path and os.path.exists(audio_path):
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            "-an",
            output_path,
        ]

    try:
        subprocess.run(merge_cmd, capture_output=True, timeout=300, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"FFmpeg合成失败: {e.stderr.decode() if e.stderr else '未知错误'}"
        )
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)

    return output_path


def get_video_info(video_path: str) -> dict:
    """获取视频基本信息"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps,
    }
    cap.release()
    return info
