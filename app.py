import gradio as gr
import os
import tempfile
import cv2
from core.background import process_video_background
from core.face_swap import process_video_faceswap
from core.video_processor import get_video_info

def validate_video(video_path):
    if video_path is None:
        return "请上传视频文件"
    try:
        info = get_video_info(video_path)
    except Exception as e:
        return f"无法读取视频: {e}"
    if info["frame_count"] == 0:
        return "视频没有帧数据"
    return ""


def save_image_to_temp(image):
    """将Gradio传入的numpy图片保存为临时文件，返回路径"""
    if image is None:
        return None
    if isinstance(image, str):
        return image
    path = tempfile.mktemp(suffix=".png")
    bg_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bg_bgr)
    return path


# ===== Tab 1: 视频换背景 =====
def handle_background_replacement(video_path, bg_image, progress=gr.Progress()):
    err = validate_video(video_path)
    if err:
        return None, err
    if bg_image is None:
        return None, "请上传背景图片"

    try:
        bg_path = save_image_to_temp(bg_image)

        def progress_cb(p, msg):
            progress(p, desc=msg)

        output_path = process_video_background(video_path, bg_path, progress_cb)

        info = get_video_info(video_path)
        status = (
            f"处理完成！\n"
            f"原视频: {info['width']}x{info['height']}, "
            f"{info['fps']:.1f}fps, {info['frame_count']}帧, "
            f"{info['duration']:.1f}秒"
        )
        return output_path, status
    except Exception as e:
        return None, f"处理失败: {e}"


# ===== Tab 2: AI换脸 =====
def handle_face_swap(video_path, face_image, progress=gr.Progress()):
    err = validate_video(video_path)
    if err:
        return None, err
    if face_image is None:
        return None, "请上传人脸照片"

    try:
        face_path = save_image_to_temp(face_image)

        def progress_cb(p, msg):
            progress(p, desc=msg)

        output_path = process_video_faceswap(video_path, face_path, progress_cb)
        return output_path, "换脸处理完成！"
    except FileNotFoundError as e:
        return None, str(e)
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"处理失败: {e}"


# ===== Tab 3: 组合处理 =====
def handle_combined(video_path, bg_image, face_image, progress=gr.Progress()):
    err = validate_video(video_path)
    if err:
        return None, err
    if bg_image is None:
        return None, "请上传背景图片"
    if face_image is None:
        return None, "请上传人脸照片"

    try:
        bg_path = save_image_to_temp(bg_image)
        face_path = save_image_to_temp(face_image)

        # 阶段1：换背景
        def progress_cb_bg(p, msg):
            progress(p * 0.5, desc=f"[1/2 换背景] {msg}")

        bg_output = process_video_background(video_path, bg_path, progress_cb_bg)

        # 阶段2：换脸
        def progress_cb_face(p, msg):
            progress(0.5 + p * 0.5, desc=f"[2/2 换脸] {msg}")

        final_output = process_video_faceswap(bg_output, face_path, progress_cb_face)

        if os.path.exists(bg_output):
            os.remove(bg_output)

        return final_output, "组合处理完成！（换背景 → 换脸）"
    except Exception as e:
        return None, f"处理失败: {e}"


# ===== 构建界面 =====
def create_app() -> gr.Blocks:
    with gr.Blocks(title="Seedance - 视频处理工具") as app:

        gr.Markdown("# 🎬 Seedance 视频处理工具")
        gr.Markdown("支持视频换背景、AI换脸、以及两者组合处理")

        with gr.Tabs():
            # Tab 1: 视频换背景
            with gr.Tab("视频换背景"):
                with gr.Row():
                    with gr.Column():
                        bg_video_in = gr.Video(label="上传视频", sources=["upload"])
                        bg_image_in = gr.Image(label="上传背景图片", type="numpy")
                        bg_btn = gr.Button("开始处理", variant="primary")
                    with gr.Column():
                        bg_video_out = gr.Video(label="处理结果")
                        bg_status = gr.Textbox(label="状态", interactive=False)

                bg_btn.click(
                    fn=handle_background_replacement,
                    inputs=[bg_video_in, bg_image_in],
                    outputs=[bg_video_out, bg_status],
                )

            # Tab 2: AI换脸
            with gr.Tab("AI换脸"):
                with gr.Row():
                    with gr.Column():
                        fs_video_in = gr.Video(label="上传视频", sources=["upload"])
                        fs_face_in = gr.Image(
                            label="上传人脸照片（正面清晰）", type="numpy"
                        )
                        fs_btn = gr.Button("开始处理", variant="primary")
                    with gr.Column():
                        fs_video_out = gr.Video(label="处理结果")
                        fs_status = gr.Textbox(label="状态", interactive=False)

                fs_btn.click(
                    fn=handle_face_swap,
                    inputs=[fs_video_in, fs_face_in],
                    outputs=[fs_video_out, fs_status],
                )

            # Tab 3: 组合处理
            with gr.Tab("组合处理"):
                with gr.Row():
                    with gr.Column():
                        cb_video_in = gr.Video(label="上传视频", sources=["upload"])
                        cb_bg_in = gr.Image(label="上传背景图片", type="numpy")
                        cb_face_in = gr.Image(label="上传人脸照片", type="numpy")
                        cb_btn = gr.Button("开始处理", variant="primary")
                    with gr.Column():
                        cb_video_out = gr.Video(label="处理结果")
                        cb_status = gr.Textbox(label="状态", interactive=False)

                cb_btn.click(
                    fn=handle_combined,
                    inputs=[cb_video_in, cb_bg_in, cb_face_in],
                    outputs=[cb_video_out, cb_status],
                )

        gr.Markdown(
            "---\n"
            "**提示：** 首次运行会自动下载AI模型（约500MB），请耐心等待。"
            "视频时长限制2分钟。处理速度取决于硬件性能。"
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
