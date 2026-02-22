import cv2
import numpy as np
import os
import tempfile
from typing import Optional, Callable, Tuple
import insightface
from insightface.app import FaceAnalysis
from core.video_processor import extract_frames, compose_video

# 模块级缓存
_face_analyzer = None
_face_swapper = None

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
SWAPPER_MODEL_PATH = os.path.join(MODELS_DIR, "inswapper_128.onnx")
SWAPPER_DOWNLOAD_URL = (
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
)


def _ensure_model_exists():
    """确保 inswapper_128.onnx 模型文件存在"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(SWAPPER_MODEL_PATH):
        raise FileNotFoundError(
            f"未找到换脸模型文件，请手动下载：\n"
            f"  wget -O {SWAPPER_MODEL_PATH} \\\n"
            f"    {SWAPPER_DOWNLOAD_URL}\n"
            f"或用浏览器下载 inswapper_128.onnx 放到 {MODELS_DIR}/ 目录"
        )


def _get_providers():
    """自动检测GPU，优先使用CUDA"""
    import onnxruntime as ort
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def init_face_analyzer() -> FaceAnalysis:
    """初始化人脸检测分析器（单例）"""
    global _face_analyzer
    if _face_analyzer is None:
        providers = _get_providers()
        _face_analyzer = FaceAnalysis(name="buffalo_l", providers=providers)
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        _face_analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return _face_analyzer
    return _face_analyzer


def init_swapper():
    """初始化换脸模型（单例）"""
    global _face_swapper
    if _face_swapper is None:
        _ensure_model_exists()
        _face_swapper = insightface.model_zoo.get_model(
            SWAPPER_MODEL_PATH, download=False, download_zip=False
        )
    return _face_swapper


def get_source_face(face_image: np.ndarray):
    """从人脸图片中检测并提取人脸特征，返回 Face 对象"""
    analyzer = init_face_analyzer()
    faces = analyzer.get(face_image)

    if len(faces) == 0:
        raise ValueError("未检测到人脸，请使用正面清晰的人脸照片")

    # 取面积最大的脸
    if len(faces) > 1:
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
    return faces[0]


def swap_face_in_frame(
    frame: np.ndarray, source_face, analyzer: FaceAnalysis, swapper
) -> Tuple[np.ndarray, bool]:
    """对单帧执行换脸，返回 (处理后帧, 是否检测到人脸)"""
    target_faces = analyzer.get(frame)

    if len(target_faces) == 0:
        return frame.copy(), False

    result = frame.copy()
    for target_face in target_faces:
        result = swapper.get(result, target_face, source_face, paste_back=True)

    return result, True


def process_video_faceswap(
    video_path: str,
    face_image_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """处理整个视频的人脸替换，返回输出视频路径"""
    if progress_callback:
        progress_callback(0.0, "正在初始化模型...")

    analyzer = init_face_analyzer()
    swapper = init_swapper()

    if progress_callback:
        progress_callback(0.02, "正在分析源人脸...")

    face_image = cv2.imread(face_image_path)
    if face_image is None:
        raise ValueError(f"无法读取人脸图片: {face_image_path}")

    source_face = get_source_face(face_image)

    if progress_callback:
        progress_callback(0.05, "正在读取视频...")

    frames, fps, audio_path = extract_frames(video_path)
    total_frames = len(frames)

    if progress_callback:
        progress_callback(0.08, f"共 {total_frames} 帧，开始换脸...")

    processed_frames = []
    faces_detected_count = 0

    for i, frame in enumerate(frames):
        result_frame, detected = swap_face_in_frame(
            frame, source_face, analyzer, swapper
        )
        processed_frames.append(result_frame)
        if detected:
            faces_detected_count += 1

        if progress_callback:
            progress = 0.08 + 0.82 * ((i + 1) / total_frames)
            progress_callback(
                progress,
                f"换脸处理: {i + 1}/{total_frames} 帧 (检测到人脸: {faces_detected_count} 帧)",
            )

    if progress_callback:
        progress_callback(0.90, "正在合成视频...")

    output_path = tempfile.mktemp(suffix=".mp4")
    compose_video(processed_frames, fps, audio_path, output_path)

    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)

    if progress_callback:
        rate = faces_detected_count / total_frames * 100
        progress_callback(1.0, f"处理完成！人脸检测率: {rate:.1f}%")

    return output_path
