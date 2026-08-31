from __future__ import annotations

import shutil
import subprocess
import json
from pathlib import Path

import cv2


def build_preview_video(source: str | Path, output: str | Path) -> Path:
    """Build a small, seek-friendly calibration/player proxy.

    The source files can be multiple gigabytes and have long GOPs. Browsers must
    decode from the previous keyframe for every timeline seek, which looks like
    play/pause/play. The proxy keeps the same timeline but inserts a keyframe
    every second and uses 15 fps at 960 px for responsive UI preview.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法生成流畅预览视频")
    if not source.is_file():
        raise RuntimeError("预览视频源文件不存在")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.building{output.suffix}")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-map", "0:v:0", "-vf", "scale=960:-2,fps=15", "-an",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
        "-g", "15", "-keyint_min", "15", "-sc_threshold", "0",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        temporary.replace(output)
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"预览视频生成失败：{detail[-1000:]}") from exc
    return output


def ensure_browser_video(source: str | Path, output: str | Path) -> Path:
    """Return a seek-friendly H.264/AAC MP4, reusing a valid cached copy.

    OpenCV's ``mp4v`` output is an MP4 container but is not a codec Chrome is
    required to decode.  This adapter fixes both codec compatibility and slow
    seeking (fast-start metadata plus one-second keyframes).
    """
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.is_file() and output.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return output
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg:
        return source
    browser_ready = False
    if ffprobe:
        try:
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt", "-of", "json", str(source)],
                check=True, capture_output=True, text=True,
            )
            stream = (json.loads(probe.stdout).get("streams") or [{}])[0]
            browser_ready = stream.get("codec_name") in {"h264", "av1", "vp9"} and stream.get("pix_fmt") in {"yuv420p", "yuvj420p"}
        except Exception:
            browser_ready = False
    # Still remux browser-ready files when a distinct cache path was requested,
    # placing the moov atom first for prompt metadata and range seeking.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.building{output.suffix}")
    temporary.unlink(missing_ok=True)
    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source), "-map", "0:v:0", "-map", "0:a?"]
    if browser_ready:
        command += ["-c", "copy"]
    else:
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0", "-c:a", "aac", "-b:a", "128k"]
    command += ["-movflags", "+faststart", str(temporary)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        temporary.replace(output)
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"浏览器视频转换失败：{detail[-1000:]}") from exc
    return output


def build_player_compilation(source: str | Path, intervals: list[tuple[float, float]], output: str | Path) -> Path:
    """Concatenate all visible player intervals into one cached review video."""
    source = Path(source).resolve(); output = Path(output).resolve()
    if output.is_file() and output.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return output
    if not intervals:
        raise RuntimeError("该球员没有可用于生成总视频的可见时段")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法生成球员总视频")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.building{output.suffix}")
    filter_path = output.with_suffix(".filter.txt")
    temporary.unlink(missing_ok=True)
    parts = []
    labels = []
    for index, (start, end) in enumerate(intervals):
        parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}]")
        labels.append(f"[v{index}]")
    parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=1:a=0,scale=960:-2,fps=15[outv]")
    filter_path.write_text(";\n".join(parts), encoding="utf-8")
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-filter_complex_script", str(filter_path), "-map", "[outv]", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p",
        "-g", "15", "-keyint_min", "15", "-sc_threshold", "0", "-movflags", "+faststart", str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        temporary.replace(output)
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"球员总视频生成失败：{detail[-1000:]}") from exc
    finally:
        filter_path.unlink(missing_ok=True)
    return output


def probe_video(path: str | Path) -> dict:
    path = Path(path).resolve()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("无法打开上传的视频")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    if fps <= 0 or width <= 0 or height <= 0 or frames <= 0:
        raise RuntimeError(f"视频元数据异常: fps={fps}, size={width}x{height}, frames={frames}")
    return {
        "filename": path.name,
        "path": str(path),
        "fps": round(fps, 6),
        "width": width,
        "height": height,
        "frame_count": frames,
        "duration_seconds": round(frames / fps, 3),
        "size_bytes": path.stat().st_size,
    }


def read_frame_jpeg(path: str | Path, frame_index: int, max_width: int = 1440) -> tuple[bytes, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("无法打开视频")
    total = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_index = max(0, min(int(frame_index), max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"无法读取第 {frame_index} 帧")
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("帧编码失败")
    return data.tobytes(), w, h
