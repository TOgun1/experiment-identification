"""Thin wrapper around the Meta Model API with a disk cache.

Both models are reached through the OpenAI SDK pointed at the Model API base URL,
following the official cookbook (github.com/meta-models/meta-model-cookbook):

- Muse Spark: Chat Completions (`client.chat.completions.create`).
- SAM 3.1: streamed Responses API (`client.responses.create(stream=True)`) with a
  short noun-phrase concept plus an `input_image` (data URL) or an `input_video`
  (Files API id, uploaded with `purpose="user_data"`). The streamed segmentation
  grammar is decoded by the official `meta_sam_parser` package; we never parse it
  ourselves.

Every successful response is cached on disk, keyed by a hash of the model, the
media content hash and the request parameters, so re-runs cost nothing. For SAM we
cache the raw stream events and re-run the official parser on replay, so parser
upgrades apply to cached data too. Failed or incomplete responses are not cached.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from meta_sam_parser import (
    ImageSegmentationResult,
    SegmentationResult,
    VideoSegmentationResult,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)

from physlab.config import Settings, load_settings

log = logging.getLogger(__name__)


class CacheMissError(RuntimeError):
    """Raised in offline mode when a request is not in the cache."""


class SegmentationError(RuntimeError):
    """SAM returned an incomplete response or records the parser rejected."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DiskCache:
    """JSON files under `root/<namespace>/<key[:2]>/<key>.json`."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def key(payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, namespace: str, key: str) -> Path:
        return self.root / namespace / key[:2] / f"{key}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        path = self._path(namespace, key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        tmp.replace(path)


def _event_dict(event: Any) -> dict[str, Any]:
    return event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)


def _normalise_events(events: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Synthesize `response.output_text.done` from `response.content_part.done`.

    The API streams `content_part.done` but not `output_text.done`, which the parser
    needs to close a text lane (ported from the cookbook's `sam_api.stream_events`).
    """
    out: list[Mapping[str, Any]] = []
    completed_lanes: set[tuple[object, object, object]] = set()
    for data in events:
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if data.get("type") == "response.output_text.done":
            if lane not in completed_lanes:
                completed_lanes.add(lane)
                out.append(data)
            continue
        out.append(data)
        part = data.get("part")
        if (
            data.get("type") == "response.content_part.done"
            and lane not in completed_lanes
            and isinstance(part, Mapping)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            completed_lanes.add(lane)
            out.append(
                {
                    "type": "response.output_text.done",
                    "item_id": data.get("item_id"),
                    "output_index": data.get("output_index"),
                    "content_index": data.get("content_index"),
                    "text": part["text"],
                }
            )
    return out


def _run(coro: Any) -> Any:
    """Run a coroutine to completion, even if called from inside an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # re-raised in the caller's thread
            result["error"] = exc

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def parse_segmentation_events(
    events: Iterable[Mapping[str, Any]], media: str
) -> SegmentationResult:
    """Decode recorded Responses events with the official SAM parser."""
    fmt = image_segmentation_format() if media == "image" else video_segmentation_format()
    normalised = _normalise_events(events)

    async def source():
        for event in normalised:
            yield event

    async def parse():
        return await parse_responses_stream(source(), fmt).final_result()

    return _run(parse())


def validate_segmentation(result: SegmentationResult) -> None:
    if result.outcome.status != "completed":
        reason = getattr(result.outcome, "reason", "unknown")
        detail = getattr(result.outcome, "detail", None)
        raise SegmentationError(f"Segmentation was incomplete: {reason} ({detail})")
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        messages = "; ".join(f"line {d.line}: {d.message}" for d in errors)
        raise SegmentationError(f"The parser rejected records: {messages}")


class MetaClient:
    """Cached access to Muse Spark (chat) and SAM 3.1 (image/video segmentation)."""

    def __init__(self, settings: Settings | None = None, api: Any = None) -> None:
        self.settings = settings or load_settings()
        self.cache = DiskCache(self.settings.cache_dir)
        self._api = api
        self.stats = {"hits": 0, "misses": 0}

    @property
    def api(self) -> Any:
        if self._api is None:
            if not self.settings.api_key:
                raise RuntimeError("Set MODEL_API_KEY in .env (see .env.example).")
            from openai import OpenAI

            self._api = OpenAI(base_url=self.settings.base_url, api_key=self.settings.api_key)
        return self._api

    def _lookup(self, namespace: str, key: str) -> Any | None:
        cached = self.cache.get(namespace, key)
        if cached is not None:
            self.stats["hits"] += 1
            log.debug("cache hit %s/%s", namespace, key[:12])
            return cached
        if self.settings.offline:
            raise CacheMissError(f"{namespace}/{key[:12]} not cached and PHYSLAB_OFFLINE is set")
        self.stats["misses"] += 1
        log.info("cache miss %s/%s: calling the Meta Model API", namespace, key[:12])
        return None

    # ---- Muse Spark -------------------------------------------------------------

    def chat_completion(
        self, messages: list[dict[str, Any]], model: str | None = None, **params: Any
    ) -> dict[str, Any]:
        """Return the full Chat Completions response as a dict (cached)."""
        model = model or self.settings.muse_model
        key = DiskCache.key({"model": model, "messages": messages, "params": params})
        cached = self._lookup("chat", key)
        if cached is not None:
            return cached
        response = self.api.chat.completions.create(model=model, messages=messages, **params)
        data = _event_dict(response)
        self.cache.put("chat", key, data)
        return data

    def chat(self, messages: list[dict[str, Any]], model: str | None = None, **params: Any) -> str:
        """Return just the assistant text of a Chat Completions call (cached)."""
        data = self.chat_completion(messages, model=model, **params)
        return data["choices"][0]["message"]["content"] or ""

    # ---- SAM 3.1 ----------------------------------------------------------------

    @staticmethod
    def _sam_input(concept: str, media: dict[str, str]) -> list[dict[str, Any]]:
        phrase = concept.strip()
        if not phrase:
            raise ValueError("The SAM concept must be a short noun phrase.")
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": phrase}, media],
            }
        ]

    def _stream_events(self, model: str, input_: list[dict[str, Any]]) -> list[dict[str, Any]]:
        stream = self.api.responses.create(model=model, stream=True, input=input_)
        try:
            return [_event_dict(event) for event in stream]
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

    def _segment(
        self, namespace: str, key: str, media: str, model: str, input_factory: Any
    ) -> SegmentationResult:
        events = self._lookup(namespace, key)
        if events is not None:
            return parse_segmentation_events(events, media)
        events = self._stream_events(model, input_factory())
        result = parse_segmentation_events(events, media)
        validate_segmentation(result)  # raises before caching a bad response
        self.cache.put(namespace, key, events)
        return result

    def segment_image(
        self, image: Path | str, concept: str, model: str | None = None
    ) -> ImageSegmentationResult:
        """Segment every instance of `concept` in one image file."""
        path = Path(image)
        model = model or self.settings.sam_model
        mime, _ = mimetypes.guess_type(path.name)
        if mime is None or not mime.startswith("image/"):
            raise ValueError(f"Could not infer an image media type for {path}")
        key = DiskCache.key(
            {"model": model, "image_sha256": sha256_file(path), "concept": concept.strip()}
        )

        def input_factory() -> list[dict[str, Any]]:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            url = f"data:{mime};base64,{encoded}"
            return self._sam_input(concept, {"type": "input_image", "image_url": url})

        return self._segment("sam_image", key, "image", model, input_factory)

    def upload_video(self, video: Path | str) -> str:
        """Upload a video to the Files API once per (content, endpoint); return its id."""
        path = Path(video)
        key = DiskCache.key({"base_url": self.settings.base_url, "video_sha256": sha256_file(path)})
        cached = self._lookup("files", key)
        if cached is not None:
            return cached["file_id"]
        mime, _ = mimetypes.guess_type(path.name)
        uploaded = self.api.files.create(
            file=(path.name, path.read_bytes(), mime or "video/mp4"), purpose="user_data"
        )
        file_id = uploaded.id
        if not isinstance(file_id, str) or not file_id.startswith("file-"):
            raise ValueError("The Files API returned an invalid file ID.")
        self.cache.put("files", key, {"file_id": file_id, "name": path.name})
        return file_id

    def segment_video(
        self, video: Path | str, concept: str, model: str | None = None
    ) -> VideoSegmentationResult:
        """Segment and track `concept` through a whole video (SAM propagates identities)."""
        path = Path(video)
        model = model or self.settings.sam_model
        key = DiskCache.key(
            {"model": model, "video_sha256": sha256_file(path), "concept": concept.strip()}
        )

        def input_factory() -> list[dict[str, Any]]:
            file_id = self.upload_video(path)
            return self._sam_input(concept, {"type": "input_video", "file_id": file_id})

        return self._segment("sam_video", key, "video", model, input_factory)
