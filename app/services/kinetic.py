"""Word-by-word kinetic captions for MPT videos.

Renders platform-style captions (bold, word pop-in, bottom-center) from the
existing subtitle timing (subtitle.srt) as an ASS subtitle file, then burns
it into the finished video with ffmpeg. Word timings are distributed evenly
across each SRT segment; the plain moviepy subtitle path stays the default
and is only replaced when --kinetic-captions is set.

Never raises: a failed caption burn keeps the uncaptioned video.
"""

import os
import re
import subprocess
from typing import Optional

from loguru import logger

# Warm yellow accent for the currently spoken word (ASS &HAABBGG&).
ACCENT_COLOR = "&H0018F0&"
ASS_FONT_SIZE = 72


def _srt_timestamp_to_seconds(ts: str) -> float:
    hours, minutes, rest = ts.strip().split(":")
    seconds = rest.replace(",", ".")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_srt(path: str) -> list[dict]:
    """Parse an SRT file into [{start, end, text}]. Pure function."""
    segments: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            content = f.read()
    except OSError:
        return segments
    blocks = re.split(r"\r?\n\r?\n", content.strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # First line may be the numeric index; find the timestamp line.
        ts_line = next(
            (ln for ln in lines if "-->" in ln),
            "",
        )
        if not ts_line:
            continue
        try:
            start_ts, end_ts = [p.strip() for p in ts_line.split("-->")]
            start = _srt_timestamp_to_seconds(start_ts)
            end = _srt_timestamp_to_seconds(end_ts)
        except (ValueError, IndexError):
            continue
        text = " ".join(lines[lines.index(ts_line) + 1 :]).strip()
        if text and end > start:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def _ass_timestamp(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    hours, rem = divmod(total_cs, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def build_kinetic_ass(
    srt_path: str,
    ass_path: str,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_size: int = ASS_FONT_SIZE,
) -> bool:
    """Build a word-by-word pop-in ASS file from an SRT file.

    Each word gets its own event: all words so far are shown, the current
    word highlighted bold in the accent color. Returns True on success.
    """
    segments = parse_srt(srt_path)
    if not segments:
        logger.warning(f"kinetic captions: no segments parsed from {srt_path}")
        return False

    events: list[str] = []
    for seg in segments:
        words = [w for w in seg["text"].split() if w]
        if not words:
            continue
        span = seg["end"] - seg["start"]
        step = span / len(words)
        for i, word in enumerate(words):
            start = seg["start"] + i * step
            end = seg["start"] + (i + 1) * step if i + 1 < len(words) else seg["end"]
            shown = " ".join(_escape_ass_text(w) for w in words[:i])
            current = _escape_ass_text(word)
            if shown:
                text = f"{shown} {{\\b1\\c{ACCENT_COLOR}}}{current}{{\\r}}"
            else:
                text = f"{{\\b1\\c{ACCENT_COLOR}}}{current}{{\\r}}"
            events.append(
                f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)},"
                f"Kinetic,,0,0,0,,{text}"
            )

    ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Kinetic,DejaVu Sans,{font_size},&H00FFFFFF,&H000000FF,&H80000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,1,2,40,40,140,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass += "\n".join(events) + "\n"
    try:
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass)
    except OSError as e:
        logger.warning(f"kinetic captions: cannot write {ass_path}: {e}")
        return False
    logger.info(
        f"kinetic captions: {len(events)} word events -> {ass_path}"
    )
    return True


def _escape_filter_path(path: str) -> str:
    # ffmpeg filter argument escaping for the ass= filter.
    return (
        path.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace(",", "\\,")
    )


def _ffmpeg_binary() -> str:
    from app.utils.utils import get_ffmpeg_binary

    return get_ffmpeg_binary()


def burn_kinetic_captions(
    video_path: str,
    srt_path: str,
    ass_path: Optional[str] = None,
) -> bool:
    """Burn word-by-word kinetic captions into a finished video file.

    Re-encodes video (libx264, crf 18), copies audio. The original file is
    only replaced after the burn succeeds. Never raises.
    """
    if not video_path or not os.path.isfile(video_path):
        logger.warning(f"kinetic captions skipped, video not found: {video_path}")
        return False
    if not srt_path or not os.path.isfile(srt_path):
        logger.warning(f"kinetic captions skipped, srt not found: {srt_path}")
        return False

    own_ass = ass_path is None
    if own_ass:
        ass_path = f"{video_path}.kinetic.ass"
    assert ass_path is not None
    if not build_kinetic_ass(srt_path, ass_path):
        return False

    burned_path = f"{video_path}.captioned.mp4"
    cmd = [
        _ffmpeg_binary(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"ass='{_escape_filter_path(ass_path)}'",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "copy",
        burned_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=20 * 60)
        os.replace(burned_path, video_path)
    except Exception as e:  # noqa: BLE001 - never break the pipeline
        logger.warning(
            "kinetic caption burn failed, keeping video without kinetic "
            f"captions: {type(e).__name__}: {e}"
        )
        try:
            if os.path.exists(burned_path):
                os.remove(burned_path)
        except OSError:
            pass
        return False
    finally:
        if own_ass:
            try:
                if os.path.exists(ass_path):
                    os.remove(ass_path)
            except OSError:
                pass
    logger.success(f"kinetic captions burned -> {video_path}")
    return True
