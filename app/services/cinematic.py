"""Cinematic looks for the MPT Veo pipeline (customer mode).

Each look is a two-part grade:
  (a) ``ffmpeg_filter``: a post-production filter chain applied to the FINAL
      assembled video, so even cheap Lite 720p footage looks premium;
  (b) ``prompt_phrase``: a short style phrase appended to the Veo
      shot-planner style block, so generated footage matches the grade.

The customer picks a look BEFORE generation (--cinematic-look); the tool
applies it, the customer never touches ffmpeg or prompt engineering.
"""

import os
import subprocess
from typing import Any, Mapping, Optional

from loguru import logger

LOOKS: dict[str, dict[str, str]] = {
    "golden-hour": {
        "label": "Golden Hour",
        "description": (
            "Warm sunset grade: amber highlights, rich cinematic warmth, "
            "gentle vignette. The default; flatters food, people, products."
        ),
        "ffmpeg_filter": (
            "colorbalance=rs=0.14:gs=0.05:bs=-0.12:"
            "rm=0.10:gm=0.03:bm=-0.08,"
            "eq=saturation=1.18:contrast=1.06,"
            "vignette=angle=PI/4.5,"
            "noise=alls=3:allf=t"
        ),
        "prompt_phrase": (
            "warm golden-hour grade, amber highlights, rich cinematic warmth"
        ),
    },
    "neon-noir": {
        "label": "Neon Noir",
        "description": (
            "Moody night grade: deep teal shadows, electric magenta accents, "
            "high contrast. For tech, nightlife, drama."
        ),
        "ffmpeg_filter": (
            "colorbalance=rs=-0.08:gs=-0.02:bs=0.16:"
            "rm=-0.06:gm=0.0:bm=0.12,"
            "eq=saturation=1.32:contrast=1.16:brightness=-0.02,"
            "vignette=angle=PI/3.8,"
            "noise=alls=5:allf=t"
        ),
        "prompt_phrase": (
            "neon-noir grade, deep teal shadows, electric magenta accents, "
            "moody high contrast"
        ),
    },
    "vintage-film": {
        "label": "Vintage Film",
        "description": (
            "Analog 35mm grade: warm faded tones, soft halation, visible "
            "grain. For nostalgia, storytelling, lifestyle."
        ),
        "ffmpeg_filter": (
            "curves=all='0/0.02 0.5/0.46 1/0.94',"
            "eq=saturation=0.85:contrast=0.97:brightness=0.02,"
            "vignette=angle=PI/4,"
            "noise=alls=8:allf=t"
        ),
        "prompt_phrase": (
            "vintage 35mm film grade, warm faded tones, soft halation, "
            "fine film grain"
        ),
    },
    "clean-modern": {
        "label": "Clean Modern",
        "description": (
            "Neutral commercial grade: balanced color, crisp detail, gentle "
            "contrast. For explainers, product demos, corporate."
        ),
        "ffmpeg_filter": (
            "eq=saturation=1.06:contrast=1.04,"
            "unsharp=5:5:0.5:5:5:0.0,"
            "noise=alls=2:allf=t"
        ),
        "prompt_phrase": (
            "clean modern grade, neutral balanced color, crisp detail, "
            "gentle contrast"
        ),
    },
}

LOOK_NAMES = tuple(LOOKS)
DEFAULT_LOOK = "golden-hour"


def normalize_look(name: Any) -> str:
    """Coerce any look input to a valid look name; unknown -> default."""
    look = str(name or "").strip().lower()
    return look if look in LOOKS else DEFAULT_LOOK


def get_look(name: Any) -> dict[str, str]:
    """Return the look dict for a name (falls back to the default look)."""
    return LOOKS[normalize_look(name)]


def prompt_phrase(name: Any) -> str:
    """Short style phrase for the Veo shot-planner style block."""
    return get_look(name)["prompt_phrase"]


def ffmpeg_filter(name: Any) -> str:
    """The ffmpeg -vf chain for a look."""
    return get_look(name)["ffmpeg_filter"]


def _ffmpeg_binary() -> str:
    from app.utils.utils import get_ffmpeg_binary

    return get_ffmpeg_binary()


def apply_look(
    video_path: str,
    look_name: Any = "",
    settings: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Grade a finished video file in place with the chosen look.

    Re-encodes video (libx264, crf 18) and copies audio untouched.
    Returns True on success; never raises -- a failed grade must not
    destroy an already-paid-for video. The original file is only
    replaced after the graded encode succeeds.
    """
    look_key = normalize_look(
        look_name
        if str(look_name or "").strip()
        else (settings or {}).get("cinematic_look", DEFAULT_LOOK)
    )
    look = LOOKS[look_key]
    if not video_path or not os.path.isfile(video_path):
        logger.warning(f"cinematic grade skipped, video not found: {video_path}")
        return False

    graded_path = f"{video_path}.graded.mp4"
    cmd = [
        _ffmpeg_binary(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        look["ffmpeg_filter"],
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "copy",
        graded_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=20 * 60)
    except Exception as e:  # noqa: BLE001 - never break the pipeline
        logger.warning(
            f"cinematic grade '{look_key}' failed, keeping ungraded video: "
            f"{type(e).__name__}: {e}"
        )
        try:
            if os.path.exists(graded_path):
                os.remove(graded_path)
        except OSError:
            pass
        return False
    try:
        os.replace(graded_path, video_path)
    except OSError as e:
        logger.warning(f"could not replace video with graded copy: {e}")
        return False
    logger.success(
        f"cinematic grade applied: {look['label']} -> {video_path}"
    )
    return True
