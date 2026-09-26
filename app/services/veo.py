"""Google Veo text-to-video provider for MoneyPrinterTurbo.

Generates brand-new cinematic clips per script beat through the Gemini API
Veo models, then hands them to the standard on-demand material flow in
``app/services/material.py``. Paid-generation semantics match the WaveSpeed /
Seedance / MiniMax providers: generate on demand, stop as soon as the
voiceover duration is covered, never cache or reuse across tasks.

Auth: prefers explicit ``veo_api_keys`` from config.toml (upstream-friendly,
same convention as every other provider). When the config list is empty and
the Secure Vault ``custom.gemini`` surrogate is available on this machine,
it is used instead, so the key never lives in a file.
"""

import hashlib
import base64
import json
import os
import re
import sys
import time
from typing import Any, Mapping, Optional

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect

VEO_API_HOST = "generativelanguage.googleapis.com"
VEO_VAULT_CREDENTIAL = "custom.gemini"

# Model fallback order: fast preview first (Phase 0 verified), then GA ids.
# NOTE: FALLBACK_MODELS is the legacy pre-tier order. New code resolves
# models per customer tier via VEO_TIER_MODELS / tier_models() below;
# a tier never falls back into a more expensive tier.
FALLBACK_MODELS = [
    "veo-3.1-fast-generate-preview",
    "veo-3.1-fast-generate-001",
    "veo-3.1-lite-generate-preview",
]

# Customer-facing quality tiers. "lite" is the suggested default: cheapest
# true video generation, and the grade in app/services/cinematic.py makes
# even 720p Lite output look premium.
VEO_TIERS = ("lite", "fast", "standard")
DEFAULT_VEO_TIER = "lite"
SUGGESTED_VEO_TIER = "lite"

# Per-tier model lists. 404/429 fallbacks walk WITHIN the chosen tier's
# list only; the pipeline never silently upgrades the customer to a
# pricier tier.
VEO_TIER_MODELS = {
    "lite": ["veo-3.1-lite-generate-preview"],
    "fast": ["veo-3.1-fast-generate-preview", "veo-3.1-fast-generate-001"],
    "standard": ["veo-3.1-generate-preview", "veo-3.1-generate-001"],
}

# Published USD/sec rates (upper bound for planning; the invoice is truth).
VEO_RATES = {
    "veo-3.1-fast-generate-preview": {"720p": 0.10, "1080p": 0.10},
    "veo-3.1-fast-generate-001": {"720p": 0.10, "1080p": 0.10},
    "veo-3.1-lite-generate-preview": {"720p": 0.05, "1080p": 0.08},
    "veo-3.1-generate-preview": {"720p": 0.40, "1080p": 0.40},
    "veo-3.1-generate-001": {"720p": 0.40, "1080p": 0.40},
}

# Veo 3.1 generates 4, 6 or 8 second clips.
SUPPORTED_DURATIONS = (4, 6, 8)
DEFAULT_MIN_DURATION = 4
DEFAULT_MAX_DURATION = 8

MAX_PROMPT_LENGTH = 2000
POLL_DEADLINE_SECONDS = 12 * 60
MAX_DOWNLOAD_RETRIES = 2
DOWNLOAD_RETRY_BASE_SECONDS = 5

# The locked style block every shot prompt must carry so clips from
# separate generations feel shot by one crew.
DEFAULT_STYLE_BLOCK = (
    "Cinematic commercial film look: 35mm film, rich warm color grade, "
    "shallow depth of field, soft practical lighting, smooth professional "
    "camera movement. Photorealistic. No on-screen text, no watermarks."
)

CAMERA_MOVES = [
    "Slow cinematic push-in on the subject",
    "Gentle lateral dolly across the scene",
    "Slow crane descent revealing the subject",
    "Smooth arc moving around the subject",
    "Slow pull-back revealing the wider scene",
    "Subtle drifting handheld feel, subject stays sharp",
]

# Alternate opening directions for hook variants (index 0 = the planned
# opening, used as-is; the rest get an alternate-angle instruction).
HOOK_VARIANT_DIRECTIONS = (
    "",
    "Alternate opening angle: open on an extreme close-up product detail, "
    "then pull back to reveal the scene.",
    "Alternate opening angle: sweeping high establishing move that dives "
    "into the subject.",
)
MAX_HOOK_VARIANTS = 3

# Reference ("ingredients") images: up to 3, asset type only on Veo 3.1.
MAX_REFERENCE_IMAGES = 3


def hook_variant_prompt(base_prompt: str, variant_index: int) -> str:
    """Append an alternate-opening direction for hook variant N (1-based)."""
    direction = HOOK_VARIANT_DIRECTIONS[variant_index % len(HOOK_VARIANT_DIRECTIONS)]
    base = str(base_prompt or "").strip()
    if not direction:
        return base
    return f"{base} {direction}"


def _load_reference_images(paths: list[str]) -> list[dict]:
    """Validate and base64-encode reference images for the Veo payload.

    Raises VeoError when a file is missing/unreadable or more than
    MAX_REFERENCE_IMAGES are given. Shape follows the documented
    ``instances[].referenceImages`` ingredients format
    (referenceType "asset", the only type Veo 3.1 accepts).
    """
    refs: list[dict] = []
    for raw in paths:
        path = str(raw or "").strip()
        if not path:
            continue
        if len(refs) >= MAX_REFERENCE_IMAGES:
            raise VeoError(
                f"at most {MAX_REFERENCE_IMAGES} reference images are supported"
            )
        if not os.path.isfile(path):
            raise VeoError(f"reference image not found: {path}")
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
        except OSError as e:
            raise VeoError(f"cannot read reference image {path}: {e}") from e
        ext = os.path.splitext(path)[1].lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(ext, "image/png")
        refs.append(
            {
                "image": {"bytesBase64Encoded": data, "mimeType": mime},
                "referenceType": "asset",
            }
        )
    return refs


class VeoError(RuntimeError):
    """Base error for the Veo provider."""


class VeoUnconfirmedTaskError(VeoError):
    """A paid generation was submitted but its final state is unknown.

    The remote operation may still complete and bill. Callers must stop
    submitting new generations (same contract as WaveSpeed's
    WaveSpeedUnconfirmedTaskError) and surface the operation name so the
    user can recover the clip from the Google AI Studio console.
    """

    def __init__(self, message: str, operation_name: str = ""):
        super().__init__(message)
        self.operation_name = operation_name


class VeoDownloadError(VeoError):
    """The clip finished generating but could not be downloaded."""


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

_surrogate_available: Optional[bool] = None


def _surrogate_url(url: str) -> str:
    """Attach the Secure Vault surrogate for custom.gemini to a URL.

    Raises VeoError when the surrogate helper is not importable (i.e. not
    running on the provisioned machine); callers then tell the user to set
    veo_api_keys instead.
    """
    global _surrogate_available
    if _surrogate_available is False:
        raise VeoError(
            "Veo needs an API key: set veo_api_keys in config.toml "
            "(Secure Vault surrogate is not available on this machine)"
        )
    try:
        sys.path.insert(0, "/opt/hatch/skills/skill-creator/bin")
        from dynamic_credentials import url_with_surrogate_query_param

        _surrogate_available = True
        return url_with_surrogate_query_param(
            url, VEO_VAULT_CREDENTIAL, allowed_hosts=[VEO_API_HOST]
        )
    except ImportError:
        _surrogate_available = False
        raise VeoError(
            "Veo needs an API key: set veo_api_keys in config.toml "
            "(Secure Vault surrogate is not available on this machine)"
        )


def _redact_key(text: str) -> str:
    return re.sub(r"([?&]key=)[^&\s]+", r"\1***", str(text))


def get_api_key(settings: Mapping[str, Any] | None = None) -> Optional[str]:
    """Return a configured Veo API key, or None to use the Vault surrogate."""
    settings = config.app if settings is None else settings
    api_keys = settings.get("veo_api_keys")
    if not api_keys:
        return None
    if isinstance(api_keys, str):
        return api_keys
    return api_keys[0]


def is_enabled(settings: Mapping[str, Any] | None = None) -> bool:
    """True when a config key exists or the Vault surrogate may be usable."""
    if get_api_key(settings):
        return True
    try:
        _surrogate_url(f"https://{VEO_API_HOST}/")
        return True
    except VeoError:
        return False


def _authed_url(url: str, api_key: Optional[str]) -> str:
    if api_key:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}key={api_key}"
    return _surrogate_url(url)


def _model_id(settings: Mapping[str, Any] | None = None) -> str:
    settings = config.app if settings is None else settings
    model = str(settings.get("veo_model", "") or "").strip()
    if model:
        return model
    return FALLBACK_MODELS[0]


def normalize_tier(tier: Any) -> str:
    """Coerce any tier input to a valid tier name; unknown -> lite."""
    t = str(tier or "").strip().lower()
    return t if t in VEO_TIER_MODELS else DEFAULT_VEO_TIER


def tier_models(
    tier: Any = "", settings: Mapping[str, Any] | None = None
) -> list[str]:
    """Model list for a customer tier.

    An explicit ``veo_model`` in config still wins (operator override).
    Otherwise the tier's own model list is returned, so 404/429 fallbacks
    stay inside the tier the customer chose and priced.
    """
    settings = config.app if settings is None else settings
    explicit = str(settings.get("veo_model", "") or "").strip()
    if explicit:
        return [explicit]
    t = normalize_tier(tier if str(tier or "").strip() else settings.get("veo_tier"))
    return list(VEO_TIER_MODELS[t])


def tier_rate_per_s(tier: Any, resolution: str = "1080p") -> float:
    """Published USD/sec for a tier's primary model (upper bound)."""
    model = tier_models(tier)[0]
    rates = VEO_RATES.get(model, VEO_RATES[FALLBACK_MODELS[0]])
    return rates.get(resolution, rates["1080p"])


def pricing_table(
    num_clips: int, duration_s: int, resolution: str = "1080p"
) -> list[dict]:
    """One row per tier: model, per-second rate, estimated run cost.

    Estimates use 1080p rates as the documented upper bound; the invoice
    is truth. ``suggested`` marks the default cheap tier for the customer.
    """
    rows = []
    for t in VEO_TIERS:
        model = VEO_TIER_MODELS[t][0]
        rate = tier_rate_per_s(t, resolution)
        rows.append(
            {
                "tier": t,
                "model": model,
                "rate_per_s": rate,
                "estimated_usd": round(num_clips * duration_s * rate, 2),
                "suggested": t == SUGGESTED_VEO_TIER,
            }
        )
    return rows


def format_pricing_table(
    num_clips: int, duration_s: int, resolution: str = "1080p"
) -> str:
    """Human-readable pricing table for the pre-generation log."""
    lines = [
        "Veo pricing for this run "
        f"({num_clips} clips x {duration_s}s, {resolution} rates, upper bound):"
    ]
    for row in pricing_table(num_clips, duration_s, resolution):
        marker = " <-- suggested (cheapest)" if row["suggested"] else ""
        lines.append(
            f"  {row['tier']:<8} {row['model']:<32} "
            f"${row['rate_per_s']:.2f}/s  ~= ${row['estimated_usd']:.2f}{marker}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Phase 1: shot planner
# --------------------------------------------------------------------------

def _template_shot_prompts(
    search_terms: list[str], style_block: str
) -> list[str]:
    """Deterministic fallback: term + rotating camera move + style lock."""
    prompts = []
    for i, term in enumerate(search_terms):
        term_text = str(term or "").strip().rstrip(".")
        move = CAMERA_MOVES[i % len(CAMERA_MOVES)]
        prompts.append(f"{term_text}. {move}. {style_block}")
    return prompts


def _llm_shot_prompts(
    search_terms: list[str],
    video_subject: str,
    style_block: str,
    app_config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Ask the configured LLM to write director-style prompts, one per beat."""
    from app.services import llm as llm_service

    beats = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(search_terms))
    prompt = (
        "You are a commercial film director writing text-to-video prompts "
        "for Google Veo.\n"
        f"Ad subject: {video_subject or '(see beats below)'}\n"
        f"Write ONE shot prompt per story beat below, in order ({len(search_terms)} beats):\n"
        f"{beats}\n\n"
        "Rules:\n"
        "- Each prompt describes ONE continuous shot: subject, action, camera movement, lighting.\n"
        "- Vary the camera movement across shots (push-in, lateral dolly, crane, arc, pull-back).\n"
        "- Every prompt MUST end with this exact style block:\n"
        f'"{style_block}"\n'
        "- Each prompt under 400 characters. Photorealistic; no on-screen text or dialogue "
        "unless the beat needs it.\n"
        "Return ONLY a JSON array of strings. No markdown, no commentary."
    )
    raw = llm_service._generate_response(prompt, app_config)
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    prompts = json.loads(text)
    if not isinstance(prompts, list) or not prompts:
        raise VeoError("shot planner LLM did not return a prompt list")
    cleaned = [str(p).strip() for p in prompts if str(p).strip()]
    if len(cleaned) != len(search_terms):
        raise VeoError(
            f"shot planner returned {len(cleaned)} prompts for "
            f"{len(search_terms)} beats"
        )
    return cleaned


def build_shot_prompts(
    search_terms: list[str],
    video_subject: str = "",
    style_block: str = "",
    app_config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Expand each script beat into a director-style Veo prompt.

    All prompts share one locked style block so separately generated clips
    feel shot by one crew. Prefers the LLM; falls back to the deterministic
    template when the LLM is unavailable or returns garbage (paid Veo calls
    must never depend on a flaky planner).
    """
    terms = [str(t).strip() for t in (search_terms or []) if str(t).strip()]
    if not terms:
        raise VeoError("shot planner needs at least one script beat")
    block = (style_block or "").strip() or DEFAULT_STYLE_BLOCK
    try:
        prompts = _llm_shot_prompts(terms, video_subject, block, app_config)
        logger.info(f"veo shot planner: LLM wrote {len(prompts)} shot prompts")
        return prompts
    except Exception as e:  # noqa: BLE001 - planner must never block generation
        logger.warning(
            "veo shot planner LLM failed, using template fallback: "
            f"{type(e).__name__}: {e}"
        )
        return _template_shot_prompts(terms, block)


# --------------------------------------------------------------------------
# Phase 4: cost estimation
# --------------------------------------------------------------------------

def estimate_cost(
    num_clips: int,
    duration_s: int,
    model: str = "",
    resolution: str = "1080p",
) -> float:
    """Estimated USD for a planned generation run (upper bound)."""
    model_id = model or FALLBACK_MODELS[0]
    rates = VEO_RATES.get(model_id, VEO_RATES[FALLBACK_MODELS[0]])
    rate = rates.get(resolution, rates["1080p"])
    return round(num_clips * duration_s * rate, 4)


# --------------------------------------------------------------------------
# Phase 2: generation
# --------------------------------------------------------------------------

def _normalize_duration(
    minimum_duration: int, settings: Mapping[str, Any] | None = None
) -> int:
    settings = config.app if settings is None else settings

    def read_bound(key: str, fallback: int) -> int:
        try:
            return max(1, int(settings.get(key, fallback)))
        except (TypeError, ValueError):
            return fallback

    lo = read_bound("veo_min_duration", DEFAULT_MIN_DURATION)
    hi = read_bound("veo_max_duration", DEFAULT_MAX_DURATION)
    if lo > hi:
        lo, hi = hi, lo
    clamped = max(lo, min(hi, int(minimum_duration or lo)))
    for supported in SUPPORTED_DURATIONS:
        if clamped <= supported:
            return supported
    return SUPPORTED_DURATIONS[-1]


def _aspect_ratio(video_aspect: VideoAspect) -> str:
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.portrait:
        return "9:16"
    if aspect == VideoAspect.landscape:
        return "16:9"
    return "1:1"


def _wait_for_operation(
    operation_name: str, api_key: Optional[str], timeout_s: int = POLL_DEADLINE_SECONDS
) -> dict:
    """Poll a predictLongRunning operation until done. Never logs the key."""
    deadline = time.time() + timeout_s
    delay = 10
    op_url = f"https://{VEO_API_HOST}/v1beta/{operation_name}"
    while time.time() < deadline:
        time.sleep(delay)
        try:
            resp = requests.get(
                _authed_url(op_url, api_key),
                proxies=config.proxy,
                timeout=120,
            )
            resp.raise_for_status()
            op = resp.json()
        except Exception as e:  # noqa: BLE001 - poll again; timeout decides
            logger.warning(
                "veo operation poll failed, will retry: "
                f"{type(e).__name__}: {_redact_key(e)}"
            )
            delay = min(delay * 1.3, 30)
            continue
        if op.get("done"):
            return op
        delay = min(delay * 1.3, 30)
    raise VeoUnconfirmedTaskError(
        "timed out waiting for the Veo operation; it may still complete and "
        "bill. Recover the clip from the Google AI Studio console.",
        operation_name=operation_name,
    )


def generate_videos(
    prompt: str,
    minimum_duration: int,
    video_aspect: VideoAspect = VideoAspect.portrait,
    settings: Mapping[str, Any] | None = None,
    tier: Any = "",
    reference_images: list[str] | None = None,
) -> list[MaterialInfo]:
    """Submit one Veo text-to-video generation and wait for the clip.

    Returns a one-item list of MaterialInfo whose url is the (authed)
    Veo video URI; callers download it with download_video_with_retry.
    Raises VeoUnconfirmedTaskError when the submit or poll outcome is
    unknown, so the caller stops submitting further paid generations.

    ``tier`` selects the customer tier's model list (404/429 fallbacks
    stay inside the tier; an explicit ``veo_model`` in config still wins).
    ``reference_images`` are local image paths (max 3) sent as Veo 3.1
    "ingredients" reference images so a product/character/logo stays
    identical across shots; missing files raise VeoError before any
    paid call.
    """
    settings = config.app if settings is None else settings
    api_key = get_api_key(settings)

    text = str(prompt or "").strip()
    if not text:
        # Empty prompts on a paid endpoint only produce billed garbage.
        raise VeoError("Veo prompt must not be empty")
    if len(text) > MAX_PROMPT_LENGTH:
        raise VeoError(f"Veo prompt exceeds {MAX_PROMPT_LENGTH} characters")

    duration = _normalize_duration(minimum_duration, settings)
    aspect = _aspect_ratio(video_aspect)

    # Tier-scoped models: never silently spend the customer into a
    # pricier tier when a model 404s or its quota bucket is drained.
    models = tier_models(tier, settings)

    # Reference ("ingredients") images: validated before any paid call.
    # Ingredients mode generates at 8s, so snap the duration up.
    refs = _load_reference_images(list(reference_images or []))
    if refs:
        duration = 8
        logger.info(f"veo reference images: {len(refs)} attached (ingredients mode, 8s)")

    instance: dict[str, Any] = {"prompt": text}
    if refs:
        instance["referenceImages"] = refs
    payload = {
        "instances": [instance],
        "parameters": {"aspectRatio": aspect, "durationSeconds": duration},
    }

    last_error: Optional[Exception] = None
    for model in models:
        submit_url = f"https://{VEO_API_HOST}/v1beta/models/{model}:predictLongRunning"
        logger.info(
            "generating video with Veo: "
            f"model={model}, aspect={aspect}, duration={duration}s, "
            f"prompt_length={len(text)}, reference_images={len(refs)}"
        )
        # A POST timeout / 5xx may still have created a billed operation:
        # never auto-retry the submit; surface it as unconfirmed.
        try:
            resp = requests.post(
                _authed_url(submit_url, api_key),
                json=payload,
                headers={"Content-Type": "application/json"},
                proxies=config.proxy,
                timeout=120,
            )
        except Exception as e:  # noqa: BLE001 - submit outcome unknown
            raise VeoUnconfirmedTaskError(
                f"Veo submit request failed with unknown outcome: "
                f"{type(e).__name__}: {_redact_key(e)}"
            )
        if resp.status_code == 404:
            last_error = VeoError(f"model not found: {model}")
            logger.warning(f"veo model 404, trying next fallback: {model}")
            continue
        if resp.status_code == 429:
            # Quota buckets are per-model: a drained fast-preview bucket
            # does not mean lite is drained. Fall through to the next
            # model instead of failing the clip outright.
            last_error = VeoError(f"quota exhausted on {model} (HTTP 429)")
            logger.warning(f"veo model 429 quota, trying next fallback: {model}")
            continue
        try:
            resp.raise_for_status()
        except Exception as e:  # noqa: BLE001 - keep message, drop key material
            raise VeoError(
                f"Veo rejected the request (HTTP {resp.status_code}): "
                f"{_redact_key(resp.text[:500])}"
            ) from e
        body = resp.json()
        operation_name = str(body.get("name", "")).strip()
        if not operation_name:
            raise VeoError("Veo accepted the request but returned no operation name")
        logger.info(f"veo operation started: {operation_name} (model={model})")

        op = _wait_for_operation(operation_name, api_key)
        if op.get("error"):
            raise VeoError(f"Veo operation failed: {op['error']}")
        try:
            video_uri = op["response"]["generateVideoResponse"]["generatedSamples"][0][
                "video"
            ]["uri"]
        except (KeyError, IndexError, TypeError) as e:
            raise VeoError(f"unexpected Veo operation response shape: {e}") from e

        logger.success(f"veo clip ready: model={model}, duration={duration}s")
        return [
            MaterialInfo(
                provider="veo",
                url=video_uri,
                duration=duration,
            )
        ]

    raise VeoError(f"no Veo model accepted the request: {last_error}")


def download_video_with_retry(
    video_uri: str, save_dir: str = "", api_key: Optional[str] = None
) -> str:
    """Download a finished Veo clip (authed URI) with retry.

    Re-generating a billed clip costs another generation, so download
    jitter retries the same URI first, mirroring material.py's
    _save_generated_video_with_retry semantics. Returns the local path,
    or "" when all retries are exhausted.
    """
    from app.utils import utils

    if not save_dir:
        save_dir = utils.storage_dir("cache_videos")
    os.makedirs(save_dir, exist_ok=True)

    uri_hash = hashlib.md5(video_uri.split("?")[0].encode()).hexdigest()
    video_path = os.path.join(save_dir, f"vid-veo-{uri_hash}.mp4")
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        logger.info(f"veo video already exists: {video_path}")
        return video_path

    if api_key is None:
        api_key = get_api_key()

    failure_detail = "empty result"
    for attempt in range(MAX_DOWNLOAD_RETRIES + 1):
        try:
            with requests.get(
                _authed_url(video_uri, api_key),
                proxies=config.proxy,
                timeout=(60, 240),
                stream=True,
            ) as resp:
                resp.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            f.write(chunk)
            if os.path.getsize(video_path) > 0:
                return video_path
            failure_detail = "empty result"
        except Exception as e:  # noqa: BLE001
            failure_detail = (
                f"error={type(e).__name__}, detail={_redact_key(e)}"
            )
        if attempt >= MAX_DOWNLOAD_RETRIES:
            break
        delay = DOWNLOAD_RETRY_BASE_SECONDS * (attempt + 1)
        logger.warning(
            "failed to download veo clip, retry the same uri: "
            f"attempt={attempt + 1}/{MAX_DOWNLOAD_RETRIES}, "
            f"{failure_detail}, retry_in={delay}s"
        )
        time.sleep(delay)
        # A partial file must not masquerade as a finished clip.
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except OSError:
            pass
    logger.error(
        "failed to download veo clip after "
        f"{MAX_DOWNLOAD_RETRIES + 1} attempts: {failure_detail}"
    )
    return ""
