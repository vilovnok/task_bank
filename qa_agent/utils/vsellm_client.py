from __future__ import annotations

import os
import threading
import time
import json
import base64
import tempfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger
from PIL import Image

TAGME_LLM_PROXY_URL: str = (os.getenv("TAGME_LLM_PROXY_URL") or "").strip() or \
                           "https://tagme.sberdevices.ru/x/ai/llm/v1/chat/completions"

TAGME_DIRECT_URL: str = (os.getenv("TAGME_DIRECT_URL") or "").strip() or \
                        "https://tagme.sberdevices.ru/ai/llm/v1/chat/completions"

DEFAULT_PROXY_CFG_PATH: str = os.path.expanduser(
    (os.getenv("TAGME_PROXY_CFG_PATH") or "").strip() or "~/configs/.ml_config.cfg"
)

GIGAHF_PROXY_CFG_PATH: str = os.path.expanduser(
    (os.getenv("GIGAHF_PROXY_CFG_PATH") or "").strip() or "~/configs/.ml_config.cfg"
)

VIDEO_PROXY_MAX_JSON_MB = float(os.getenv("VIDEO_PROXY_MAX_JSON_MB", "0.5"))
VIDEO_PROXY_MAX_JSON_BYTES = int(VIDEO_PROXY_MAX_JSON_MB * 1024 * 1024)

VIDEO_DOWNLOAD_TIMEOUT_SEC = float(os.getenv("VIDEO_DOWNLOAD_TIMEOUT_SEC", "60"))

IMG_DOWNLOAD_TIMEOUT_SEC = float(os.getenv("IMG_DOWNLOAD_TIMEOUT_SEC", "20"))
IMG_MAX_SIDE = int(os.getenv("IMG_MAX_SIDE", "512"))
IMG_JPEG_QUALITY = int(os.getenv("IMG_JPEG_QUALITY", "60"))

GEMINI_VIDEO_MAX_HEIGHT = int(os.getenv("GEMINI_VIDEO_MAX_HEIGHT", "350"))
GEMINI_VIDEO_TARGET_FPS = int(os.getenv("GEMINI_VIDEO_TARGET_FPS", "5"))

GIGAHF_QWEN_MODELS = {
    "Qwen3-VL-235B-A22B-Instruct",
    "Qwen2.5-VL-72B-Instruct-AWQ",
    "Qwen3-VL-235B-A22B-Instruct-AWQ"
}

GIGAHF_TEXT_MODELS = {
    "DeepSeek-V3.2",
    "Qwen3-Next-80B-A3B-Thinking",
    "Qwen3-235B-A22B-Instruct-2507",
    "gpt-oss-120b"
}

TAGME_MODELS = {
    "DeepSeek-R1",
    "GigaChat",
    "GigaChat-2",
    "GigaChat-3-Ultra",
    "GigaChat-3-Lightning",
    "GigaChat-2-Pro",
    "GigaChat-Pro",
    "GigaChat-2-Max",
    "GigaChat-3-Pro",
    "GigaChat-Max",
    "Qwen"
}


def _is_gemini_model(model: str) -> bool:
    m = (model or "").strip().lower()
    return "gemini" in m


def get_proxy_token(proxy_cfg_path: str) -> str:
    try:
        from crowd_sdk.tagme import TagmeClientAdvanced
    except Exception as e:
        raise RuntimeError(
            "crowd_sdk.tagme.TagmeClientAdvanced недоступен — не могу получить proxy token"
        ) from e

    client = TagmeClientAdvanced(proxy_cfg_path)
    token = getattr(getattr(client, "client", None), "token", None)
    if not token:
        raise RuntimeError("Не получилось получить proxy token: client.client.token пустой")
    return str(token)


def _extract_usage_from_json(j: dict[str, Any]) -> Dict[str, int]:
    usage = j.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _extract_text_from_json(j: dict[str, Any]) -> str:
    try:
        choice = (j.get("choices") or [None])[0] or {}
        msg = choice.get("message") or {}
        content = msg.get("content")

        if isinstance(content, list):
            parts: List[str] = []
            for it in content:
                if isinstance(it, dict) and it.get("text"):
                    parts.append(str(it["text"]))
            return "\n".join(parts).strip()

        return (content or "").strip()
    except Exception:
        return ""


def _extract_text_from_gigahf_json(j: dict[str, Any]) -> str:
    """
    Формат GigaHF (aihf):
    {
      "choices": [
        {
          "index": 0,
          "messages": [
            {
              "role": "assistant",
              "content": [
                {"type": "text", "text": "..."}
              ]
            }
          ],
          ...
        }
      ],
      ...
    }
    """
    try:
        choices = j.get("choices") or []
        if not choices:
            return ""
        choice0 = choices[0] or {}

        messages = choice0.get("messages") or []
        if not messages:
            return ""
        msg0 = messages[0] or {}

        content = msg0.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for it in content:
                if isinstance(it, dict) and it.get("text"):
                    parts.append(str(it["text"]))
            return "\n".join(parts).strip()

        if isinstance(content, str):
            return content.strip()

        return ""
    except Exception:
        return ""


def _post_proxy(*, proxy_url: str, token: str, payload: dict[str, Any], timeout_sec: float) -> requests.Response:
    return requests.post(
        proxy_url,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        verify=False,
        timeout=(60.0, float(timeout_sec)),
    )


def _retry_call(n: int, fn, *, label: str, delay_sec: float = 0.0):
    last_exc: Optional[BaseException] = None
    for i in range(1, n + 1):
        try:
            logger.info("{} attempt {}/{}", label, i, n)
            return fn()
        except Exception as e:
            last_exc = e
            logger.warning("{} attempt {}/{} failed: {}", label, i, n, e)
            if delay_sec and i < n:
                time.sleep(delay_sec)
    assert last_exc is not None
    raise last_exc


def _clean_media_url(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _call_proxy_with_refresh(
        *,
        proxy_url: str,
        proxy_token: str,
        proxy_cfg_path: str,
        payload: dict[str, Any],
        timeout_sec: float,
) -> tuple[dict[str, Any], str]:
    r = _post_proxy(proxy_url=proxy_url, token=proxy_token, payload=payload, timeout_sec=timeout_sec)

    if r.status_code in (401, 403):
        logger.warning(
            "Proxy token expired (status={}), refreshing token using cfg_path={}",
            r.status_code,
            proxy_cfg_path,
        )
        proxy_token = get_proxy_token(proxy_cfg_path)
        logger.info("Proxy token refreshed (len={})", len(proxy_token))

        r = _post_proxy(proxy_url=proxy_url, token=proxy_token, payload=payload, timeout_sec=timeout_sec)

    r.raise_for_status()
    return r.json(), proxy_token


def _json_size_bytes(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def _download_image_bytes(url: str, timeout_sec: float = IMG_DOWNLOAD_TIMEOUT_SEC) -> bytes:
    r = requests.get(url, timeout=float(timeout_sec), verify=False)
    r.raise_for_status()
    return r.content or b""


def _resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return img.resize((nw, nh))


def _image_to_data_url_jpeg(img: Image.Image, quality: int) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality), optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _download_image_as_data_url_max_compress(image_url: str) -> str:
    b = _download_image_bytes(image_url, timeout_sec=IMG_DOWNLOAD_TIMEOUT_SEC)
    if not b:
        raise RuntimeError("Downloaded image is empty")
    img = Image.open(BytesIO(b))
    img = _resize_keep_aspect(img, IMG_MAX_SIDE)
    return _image_to_data_url_jpeg(img, IMG_JPEG_QUALITY)


def _data_image_url_to_data_url_max_compress(data_url: str) -> str:
    s = (data_url or "").strip()
    if not s.startswith("data:image/") or ";base64," not in s:
        raise ValueError("Not a base64 data:image URL")

    b64 = s.split(";base64,", 1)[1]
    raw = base64.b64decode(b64)

    img = Image.open(BytesIO(raw))
    img = _resize_keep_aspect(img, IMG_MAX_SIDE)
    return _image_to_data_url_jpeg(img, IMG_JPEG_QUALITY)


def _download_video_bytes(url: str, timeout_sec: float = VIDEO_DOWNLOAD_TIMEOUT_SEC) -> bytes:
    r = requests.get(url, timeout=float(timeout_sec), verify=False)
    r.raise_for_status()
    return r.content or b""


def _video_bytes_to_data_uri_mp4(video_bytes: bytes) -> str:
    b64 = base64.b64encode(video_bytes).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


def _compress_video_bytes_opencv(
        video_bytes: bytes,
        *,
        max_height: int,
        target_fps: int,
) -> bytes:
    try:
        import cv2 as cv
    except Exception as e:
        raise RuntimeError("OpenCV (cv2) is required for video compression fallback") from e

    in_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            in_path = f.name
            f.write(video_bytes)

        out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
        os.close(out_fd)

        cap = cv.VideoCapture(in_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open input video for compression")

        fps_in = cap.get(cv.CAP_PROP_FPS) or 0
        w_in = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 0)
        h_in = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 0)
        if w_in <= 0 or h_in <= 0:
            cap.release()
            raise RuntimeError("Failed to read input video dimensions")

        if h_in > max_height:
            scale = max_height / float(h_in)
            w_out = max(2, int(w_in * scale))
            h_out = max(2, int(h_in * scale))
        else:
            w_out, h_out = w_in, h_in

        if w_out % 2:
            w_out -= 1
        if h_out % 2:
            h_out -= 1
        w_out = max(2, w_out)
        h_out = max(2, h_out)

        fps_out = max(1, int(target_fps) if target_fps > 0 else (int(fps_in) if fps_in else 4))

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(out_path, fourcc, float(fps_out), (w_out, h_out))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Failed to open output writer for compression")

        keep_every = 1
        if fps_in and fps_out and fps_in > fps_out:
            keep_every = max(1, int(round(float(fps_in) / float(fps_out))))

        frame_idx = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if keep_every > 1 and (frame_idx % keep_every) != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            if (w_out, h_out) != (w_in, h_in):
                frame = cv.resize(frame, (w_out, h_out), interpolation=cv.INTER_AREA)

            writer.write(frame)
            written += 1

        cap.release()
        writer.release()

        if written <= 0:
            raise RuntimeError("No frames written during compression")

        with open(out_path, "rb") as f:
            return f.read()

    finally:
        for p in (in_path, out_path):
            if p:
                try:
                    os.remove(p)
                except Exception:
                    pass


class VseLLMProxyClient:
    def __init__(
            self,
            *,
            proxy_cfg_path: str | None = None,
            proxy_url: str | None = None,
            proxy_token: str | None = None,
            request_timeout: float = 120.0,
    ):
        self.proxy_cfg_path = (proxy_cfg_path or DEFAULT_PROXY_CFG_PATH).strip()
        self.proxy_url = (proxy_url or TAGME_LLM_PROXY_URL).strip()
        self.request_timeout = float(request_timeout)

        self._proxy_token = (proxy_token or "").strip() or get_proxy_token(self.proxy_cfg_path)
        self._lock = threading.Lock()

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            j, self._proxy_token = _call_proxy_with_refresh(
                proxy_url=self.proxy_url,
                proxy_token=self._proxy_token,
                proxy_cfg_path=self.proxy_cfg_path,
                payload=payload,
                timeout_sec=self.request_timeout,
            )
        return j

    def call_text(
            self,
            *,
            model: str,
            messages: List[Dict[str, Any]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            api_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, int], Any]:
        payload: Dict[str, Any] = {"model": model, "messages": messages}

        if temperature is not None:
            payload["temperature"] = float(temperature)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if top_p is not None:
            payload["top_p"] = float(top_p)

        if api_kwargs:
            for k, v in api_kwargs.items():
                if v is not None:
                    payload[k] = v

        j = self._call(payload)
        usage = _extract_usage_from_json(j)
        text = _extract_text_from_json(j)
        return (text or None), usage, j

    def call_image(
            self,
            *,
            model: str,
            system_prompt: Optional[str],
            image_url: str,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            extra_text: Optional[str] = None,
            api_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, int], Any]:

        user_text = "Проанализируй изображение согласно инструкции."
        if extra_text:
            user_text += "\n\nВот дополнительные текстовые данные для анализа:\n" + extra_text

        is_gemini = _is_gemini_model(model)

        def _build_messages_with_image_url(img_ref: str) -> List[Dict[str, Any]]:
            msgs: List[Dict[str, Any]] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": img_ref}},
                    ],
                }
            )
            return msgs

        def _build_payload(msgs: List[Dict[str, Any]]) -> Dict[str, Any]:
            payload: Dict[str, Any] = {"model": model, "messages": msgs}

            if temperature is not None:
                payload["temperature"] = float(temperature)
            if max_tokens is not None:
                payload["max_tokens"] = int(max_tokens)
            if top_p is not None:
                payload["top_p"] = float(top_p)

            if api_kwargs:
                for k, v in api_kwargs.items():
                    if v is not None:
                        payload[k] = v

            return payload

        # === ДИНАМИЧЕСКОЕ СЖАТИЕ ИЗОБРАЖЕНИЙ ===
        def _dynamic_image_data_url_from_bytes(raw_bytes: bytes) -> str:
            """
            1) Пытаемся отправить картинку без уменьшения размера (orig side, качество ~95).
            2) Если payload >= 0.5MB, пробуем последовательно более агрессивные варианты:
               - max_side=700, quality=60 (как ты просил),
               - далее ступенчато уменьшаем max_side/quality,
               пока не влезем < 0.5MB или не исчерпаем пресеты.
            Если ни один не влез — берём тот, у которого payload минимальный.
            """
            img = Image.open(BytesIO(raw_bytes))

            presets: List[Tuple[int, int]] = [
                (max(img.size), 95),  # "почти без сжатия", первый тест
                (700, 60),  # явный requested preset
                (512, 60),
                (384, 60),
                (384, 40),
                (256, 40),
            ]

            best_data_url: Optional[str] = None
            best_mb: float = 1e9

            for max_side, quality in presets:
                logger.info("IMAGE dynamic compression try: max_side={} quality={}", max_side, quality)

                if max(img.size) > max_side:
                    img_resized = _resize_keep_aspect(img, max_side)
                else:
                    img_resized = img

                data_url = _image_to_data_url_jpeg(img_resized, quality)
                tmp_msgs = _build_messages_with_image_url(data_url)
                tmp_payload = _build_payload(tmp_msgs)
                mb_size = _json_size_bytes(tmp_payload) / 1024.0 / 1024.0

                logger.info(
                    "IMAGE payload size with max_side={} quality={} -> {:.3f}MB",
                    max_side, quality, mb_size
                )

                if mb_size < best_mb:
                    best_mb = mb_size
                    best_data_url = data_url

                if mb_size < float(VIDEO_PROXY_MAX_JSON_MB):
                    logger.info(
                        "IMAGE dynamic compression fit: max_side={} quality={} payload={:.3f}MB",
                        max_side, quality, mb_size
                    )
                    return data_url

            if best_data_url is None:
                raise RuntimeError("Dynamic image compression failed to produce any data_url")

            logger.warning(
                "Cannot fit image payload under {:.3f}MB; best={:.3f}MB (will use best)",
                float(VIDEO_PROXY_MAX_JSON_MB), best_mb
            )
            return best_data_url

        if is_gemini:
            logger.info(
                "Gemini image: dynamic IMAGE_URL base64 with payload<= {:.2f}MB",
                VIDEO_PROXY_MAX_JSON_MB
            )

            # поддержка data:image/...;base64,... без requests.get
            if isinstance(image_url, str) and image_url.strip().startswith("data:image/"):
                s = image_url.strip()
                if not s.startswith("data:image/") or ";base64," not in s:
                    raise ValueError("Not a base64 data:image URL")
                b64 = s.split(";base64,", 1)[1]
                raw = base64.b64decode(b64)
            else:
                raw = _download_image_bytes(image_url, timeout_sec=IMG_DOWNLOAD_TIMEOUT_SEC)

            if not raw:
                raise RuntimeError("Downloaded image is empty")

            data_url = _dynamic_image_data_url_from_bytes(raw)
            payload = _build_payload(_build_messages_with_image_url(data_url))

            j = _retry_call(3, lambda: self._call(payload), label="GEMINI IMAGE (dynamic b64 image_url)")
            usage = _extract_usage_from_json(j)
            text = _extract_text_from_json(j)
            return (text or None), usage, j

        payload_url = _build_payload(_build_messages_with_image_url(image_url))

        try:
            j = _retry_call(3, lambda: self._call(payload_url), label="IMAGE URL")
            usage = _extract_usage_from_json(j)
            text = _extract_text_from_json(j)
            return (text or None), usage, j
        except Exception as e1:
            logger.warning("IMAGE URL failed, switching to dynamic base64: {}", e1)

        if isinstance(image_url, str) and image_url.strip().startswith("data:image/"):
            s = image_url.strip()
            if not s.startswith("data:image/") or ";base64," not in s:
                raise ValueError("Not a base64 data:image URL")
            b64 = s.split(";base64,", 1)[1]
            raw = base64.b64decode(b64)
        else:
            raw = _download_image_bytes(image_url, timeout_sec=IMG_DOWNLOAD_TIMEOUT_SEC)

        if not raw:
            raise RuntimeError("Downloaded image is empty")

        data_url = _dynamic_image_data_url_from_bytes(raw)
        payload_b64 = _build_payload(_build_messages_with_image_url(data_url))

        j = _retry_call(3, lambda: self._call(payload_b64), label="IMAGE BASE64 (dynamic image_url)")
        usage = _extract_usage_from_json(j)
        text = _extract_text_from_json(j)
        return (text or None), usage, j

    def call_video_full(
            self,
            *,
            model: str,
            video_url: str,
            prompt_text: str,
            temperature: Optional[float] = None,
            timeout_sec: int = 120,
            api_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict[str, int], Any]:
        prompt = (prompt_text or "").strip() or "Опиши, что происходит в этом видео. Будь конкретным."

        is_gemini = _is_gemini_model(model)

        def _build_payload_url(video_ref: str) -> Dict[str, Any]:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "video_url", "video_url": {"url": video_ref}},
                    ],
                }
            ]
            payload: Dict[str, Any] = {"model": model, "messages": messages}
            payload["temperature"] = float(temperature if temperature is not None else 0.2)

            if api_kwargs:
                for k, v in api_kwargs.items():
                    if v is not None:
                        payload[k] = v

            return payload

        def _build_payload_file(data_uri: str) -> Dict[str, Any]:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "file", "file": {"file_id": data_uri}},
                    ],
                }
            ]
            payload: Dict[str, Any] = {"model": model, "messages": messages}
            payload["temperature"] = float(temperature if temperature is not None else 0.2)

            if api_kwargs:
                for k, v in api_kwargs.items():
                    if v is not None:
                        payload[k] = v

            return payload

        def _payload_size_mb(obj: Any) -> float:
            try:
                return len(json.dumps(obj, ensure_ascii=False).encode("utf-8")) / 1024 / 1024
            except Exception:
                return 0.0

        def _compress_video_to_fit_payload_mb(video_bytes: bytes, *, limit_mb: float) -> str:
            HEIGHTS = [120]
            FPS = [2, 1]

            best_data_uri: Optional[str] = None
            best_mb = 1e9

            for height in HEIGHTS:
                for fps in FPS:
                    logger.info("FAST ladder try: height={} fps={}", height, fps)

                    comp_bytes = _compress_video_bytes_opencv(
                        video_bytes,
                        max_height=height,
                        target_fps=fps,
                    )
                    data_uri = _video_bytes_to_data_uri_mp4(comp_bytes)

                    payload_try = _build_payload_file(data_uri)
                    mb_size = _payload_size_mb(payload_try)

                    if mb_size < best_mb:
                        best_mb = mb_size
                        best_data_uri = data_uri

                    if mb_size <= float(limit_mb):
                        logger.info(
                            "Gemini video fit: height={} fps={} payload={:.3f}MB",
                            height, fps, mb_size
                        )
                        return data_uri

            if best_data_uri is None:
                raise RuntimeError("Gemini video compression failed")

            raise RuntimeError(
                f"Cannot fit payload under {limit_mb:.3f}MB; best={best_mb:.3f}MB"
            )

        old_timeout = self.request_timeout
        self.request_timeout = float(timeout_sec)

        try:

            if not is_gemini:
                payload = _build_payload_url(video_url)
                j = _retry_call(3, lambda: self._call(payload), label="VIDEO URL")
                usage = _extract_usage_from_json(j)
                text = _extract_text_from_json(j)
                return (text or None), usage, j

            logger.warning(
                "Gemini video: FILE base64 with adaptive compression to payload <= {:.2f}MB",
                VIDEO_PROXY_MAX_JSON_MB
            )

            if isinstance(video_url, str) and video_url.strip().startswith("data:video/"):
                logger.info("VIDEO is already base64 data URL, skipping download")

                data_uri = video_url.strip()

                def _one_call_direct():
                    payload = _build_payload_file(data_uri)
                    return self._call(payload)

                j = _retry_call(3, _one_call_direct, label="GEMINI VIDEO (data_url direct)")
                usage = _extract_usage_from_json(j)
                text = _extract_text_from_json(j)
                return (text or None), usage, j

            raw_bytes = _download_video_bytes(video_url, timeout_sec=VIDEO_DOWNLOAD_TIMEOUT_SEC)
            if not raw_bytes:
                raise RuntimeError("Downloaded video is empty")

            def _one_call():
                data_uri = _compress_video_to_fit_payload_mb(raw_bytes, limit_mb=VIDEO_PROXY_MAX_JSON_MB)
                payload = _build_payload_file(data_uri)

                mb_size = _payload_size_mb(payload)
                if mb_size > float(VIDEO_PROXY_MAX_JSON_MB):
                    raise RuntimeError(
                        f"Payload still too large: {mb_size:.3f}MB > {float(VIDEO_PROXY_MAX_JSON_MB):.3f}MB")

                return self._call(payload)

            j = _retry_call(3, _one_call, label="GEMINI VIDEO (file b64 <=0.5MB payload)")
            usage = _extract_usage_from_json(j)
            text = _extract_text_from_json(j)
            return (text or None), usage, j

        finally:
            self.request_timeout = old_timeout


_default_client: Optional[VseLLMProxyClient] = None
try:
    _default_client = VseLLMProxyClient(
        proxy_cfg_path=DEFAULT_PROXY_CFG_PATH,
        proxy_url=TAGME_LLM_PROXY_URL,
        request_timeout=180.0,
    )
except Exception as e:
    logger.warning("VseLLMProxyClient init failed: {}", e)
    _default_client = None


def _get_client(
        *,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,
        request_timeout: float = 180.0,
) -> VseLLMProxyClient:
    if _default_client is not None and proxy_cfg_path is None and proxy_url is None:
        return _default_client
    return VseLLMProxyClient(
        proxy_cfg_path=proxy_cfg_path or DEFAULT_PROXY_CFG_PATH,
        proxy_url=proxy_url or TAGME_LLM_PROXY_URL,
        request_timeout=request_timeout,
    )


def call_vsellm_text(
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        api_kwargs: Optional[Dict[str, Any]] = None,
        request_timeout: float = 180.0,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,
) -> Tuple[Optional[str], Dict[str, int], Any]:
    client = _get_client(proxy_cfg_path=proxy_cfg_path, proxy_url=proxy_url, request_timeout=request_timeout)
    return client.call_text(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        api_kwargs=api_kwargs,
    )


def call_vsellm_image(
        *,
        model: str,
        system_prompt: Optional[str],
        image_url: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        extra_text: Optional[str] = None,
        api_kwargs: Optional[Dict[str, Any]] = None,
        request_timeout: float = 180.0,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,
) -> Tuple[Optional[str], Dict[str, int], Any]:
    client = _get_client(proxy_cfg_path=proxy_cfg_path, proxy_url=proxy_url, request_timeout=request_timeout)
    return client.call_image(
        model=model,
        system_prompt=system_prompt,
        image_url=image_url,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        extra_text=extra_text,
        api_kwargs=api_kwargs,
    )


def call_vsellm_video_full(
        *,
        model: str,
        video_url: str,
        prompt_text: str,
        temperature: Optional[float] = None,
        timeout_sec: int = 180,
        api_kwargs: Optional[Dict[str, Any]] = None,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,
) -> Tuple[Optional[str], Dict[str, int], Any]:
    client = _get_client(proxy_cfg_path=proxy_cfg_path, proxy_url=proxy_url, request_timeout=float(timeout_sec))
    return client.call_video_full(
        model=model,
        video_url=video_url,
        prompt_text=prompt_text,
        temperature=temperature,
        timeout_sec=timeout_sec,
        api_kwargs=api_kwargs,
    )


def _call_gigahf_qwen_image(
        *,
        model: str,
        system_prompt: Optional[str],
        image_url: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_text: Optional[str] = None,
        timeout_sec: float = 180.0,
) -> Dict[str, Any]:
    """
    Специальный путь для Qwen*-VL через GigaHF:
    https://tagme.pd.sberdevices.ru/x/aihf/llm/v1/chat/completions
    """
    base_url = os.getenv("TAGME_GIGAHF_BASE_URL", "https://tagme.sberdevices.ru").rstrip("/")
    url = f"https://tagme.sberdevices.ru/x/aihf/llm/v1/chat/completions"

    proxy_cfg_path = GIGAHF_PROXY_CFG_PATH
    token = get_proxy_token(proxy_cfg_path)

    if isinstance(image_url, str) and image_url.strip().startswith("data:image/"):
        data_url = image_url.strip()
    else:
        data_url = _download_image_as_data_url_max_compress(image_url)

    user_text = "Опиши, что изображено на фотографии. Отвечай кратко и по делу."
    if extra_text:
        user_text += "\n\nДополнительный текст:\n" + extra_text

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    )

    payload: Dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    r = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=float(timeout_sec),
        verify=False,
    )
    r.raise_for_status()
    j = r.json()

    text = _extract_text_from_gigahf_json(j)
    usage = _extract_usage_from_json(j)

    return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": j}


def _call_gigahf_text(
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_sec: float = 180.0,
) -> Dict[str, Any]:
    base_url = os.getenv("TAGME_GIGAHF_BASE_URL", "https://tagme.sberdevices.ru").rstrip("/")
    url = f"{base_url}/x/aihf/llm/v1/chat/completions"

    proxy_cfg_path = GIGAHF_PROXY_CFG_PATH
    token = get_proxy_token(proxy_cfg_path)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    if temperature is not None:
        payload["temperature"] = float(temperature)

    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    r = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=float(timeout_sec),
        verify=False,
    )

    r.raise_for_status()
    j = r.json()

    text = _extract_text_from_gigahf_json(j)
    usage = _extract_usage_from_json(j)

    return {
        "raw_text": text or "",
        "answer": text or "",
        "usage": usage,
        "raw": j,
    }


def call_vsellm(
        *,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        user_text: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        image: Optional[bool] = None,
        video: Optional[bool] = None,
        image_url: Optional[str] = None,
        video_url: Optional[str] = None,
        extra_text: Optional[str] = None,
        timeout_sec: int = 180,
        api_kwargs: Optional[Dict[str, Any]] = None,
        proxy_cfg_path: str | None = None,
        proxy_url: str | None = None,
) -> Dict[str, Any]:
    image_url = _clean_media_url(image_url)
    video_url = _clean_media_url(video_url)

    if model in TAGME_MODELS:
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text or ""})

        if api_kwargs is None:
            api_kwargs = {}
        api_kwargs["profanity_check"] = False

        text, usage, raw = call_vsellm_text(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            api_kwargs=api_kwargs,
            request_timeout=float(timeout_sec),
            proxy_cfg_path=proxy_cfg_path,
            proxy_url=TAGME_DIRECT_URL
        )
        logger.info("TAGME MODEL")
        return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": raw}

    if video:
        if not video_url:
            raise ValueError("call_vsellm(video=True): video_url is required")

        prompt_text = (system_prompt or "").strip()
        if extra_text:
            prompt_text = (prompt_text + "\n\n" if prompt_text else "") + extra_text

        text, usage, raw = call_vsellm_video_full(
            model=model,
            video_url=video_url,
            prompt_text=prompt_text,
            temperature=temperature,
            timeout_sec=timeout_sec,
            api_kwargs=api_kwargs,
            proxy_cfg_path=proxy_cfg_path,
            proxy_url=proxy_url,
        )
        return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": raw}

    if image:
        if not image_url:
            raise ValueError("call_vsellm(image=True): image_url is required")

        if model in GIGAHF_QWEN_MODELS:
            result = _call_gigahf_qwen_image(
                model=model,
                system_prompt=system_prompt,
                image_url=image_url,
                temperature=temperature,
                max_tokens=10000,
                extra_text=extra_text,
                timeout_sec=float(timeout_sec),
            )
            return result

        if "qwen" in model:
            # Для Qwen через VseLLM: отправляем image + text как для Gemini
            if isinstance(image_url, str) and image_url.strip().startswith("data:image/"):
                data_url = image_url.strip()
            else:
                data_url = _download_image_as_data_url_max_compress(image_url)

            user_text = "Проанализируй изображение согласно инструкции."
            if extra_text:
                user_text += "\n\nВот дополнительные текстовые данные для анализа:\n" + extra_text

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            )

            text, usage, raw = call_vsellm_text(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                api_kwargs=api_kwargs,
                request_timeout=float(timeout_sec),
                proxy_cfg_path=proxy_cfg_path,
                proxy_url=proxy_url,
            )
            return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": raw}

        text, usage, raw = call_vsellm_image(
            model=model,
            system_prompt=system_prompt,
            image_url=image_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            extra_text=extra_text,
            api_kwargs=api_kwargs,
            request_timeout=float(timeout_sec),
            proxy_cfg_path=proxy_cfg_path,
            proxy_url=proxy_url,
        )
        return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": raw}

    if messages is None:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text or ""})

    if model in GIGAHF_TEXT_MODELS:

        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text or ""})

        result = _call_gigahf_text(
            model=model,
            messages=messages,
            temperature=temperature,
            # max_tokens=max_tokens,
            timeout_sec=float(timeout_sec),
        )

        return result

    text, usage, raw = call_vsellm_text(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        api_kwargs=api_kwargs,
        request_timeout=float(timeout_sec),
        proxy_cfg_path=proxy_cfg_path,
        proxy_url=proxy_url,
    )
    return {"raw_text": text or "", "answer": text or "", "usage": usage, "raw": raw}


def _add_usage(acc: dict, delta: dict) -> dict:
    """Складываем usage-словари по ключам."""
    if acc is None:
        acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        acc[k] = acc.get(k, 0) + (delta.get(k, 0) or 0)
    return acc