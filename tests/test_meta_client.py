"""Offline tests for MetaClient: caching, offline mode and SAM stream decoding.

The fake API emits events shaped like the real streamed Responses API, which sends
`response.content_part.done` but never `response.output_text.done`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from physlab.config import Settings
from physlab.meta_client import CacheMissError, MetaClient, SegmentationError

# One 5x5 one_bit mask (payload from the meta_sam_parser README) in a 64x48 image.
SAM_LINE = (
    "<{frame}f>0<|box;x1=10;y1=20;x2=14;y2=24;w=64;h=48;c=0.93|>"
    "<|mask;x=0;y=0;data=5,5,!!!!!(QO(0lu8?|>\n"
)


def sam_events(text: str, status: str = "completed") -> list[dict]:
    lane = {"item_id": "msg_1", "output_index": 0, "content_index": 0}
    return [
        {"type": "response.created", "response": {"status": "in_progress"}},
        {"type": "response.output_text.delta", **lane, "delta": text},
        {"type": "response.content_part.done", **lane, "part": {"type": "output_text", "text": text}},
        {"type": f"response.{status}", "response": {"status": status}},
    ]


class FakeAPI:
    def __init__(self, sam_text: str = SAM_LINE.format(frame=0), sam_status: str = "completed"):
        self.calls = {"chat": 0, "responses": 0, "files": 0}
        self.sam_text = sam_text
        self.sam_status = sam_status
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.responses = SimpleNamespace(create=self._responses)
        self.files = SimpleNamespace(create=self._files)

    def _chat(self, model, messages, **params):
        self.calls["chat"] += 1
        data = {"model": model, "choices": [{"message": {"role": "assistant", "content": "ready"}}]}
        return SimpleNamespace(model_dump=lambda mode="python": data)

    def _responses(self, model, stream, input):
        assert stream is True
        assert input[0]["content"][0] == {"type": "input_text", "text": "orange ball"}
        self.calls["responses"] += 1
        return iter(sam_events(self.sam_text, self.sam_status))

    def _files(self, file, purpose):
        assert purpose == "user_data"
        self.calls["files"] += 1
        return SimpleNamespace(id="file-abc123")


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "frame.png"
    path.write_bytes(b"\x89PNG fake bytes")
    return path


def make_client(tmp_path: Path, api: FakeAPI, offline: bool = False) -> MetaClient:
    settings = Settings(api_key="test", cache_dir=tmp_path / "cache", offline=offline)
    return MetaClient(settings=settings, api=api)


def test_chat_second_call_hits_cache(tmp_path):
    api = FakeAPI()
    messages = [{"role": "user", "content": "Reply with the single word: ready"}]
    first = make_client(tmp_path, api)
    assert first.chat(messages) == "ready"
    second = make_client(tmp_path, api)
    assert second.chat(messages) == "ready"
    assert api.calls["chat"] == 1
    assert second.stats == {"hits": 1, "misses": 0}


def test_chat_params_are_part_of_the_key(tmp_path):
    api = FakeAPI()
    client = make_client(tmp_path, api)
    messages = [{"role": "user", "content": "hi"}]
    client.chat(messages)
    client.chat(messages, temperature=0)
    assert api.calls["chat"] == 2


def test_segment_image_decodes_and_caches(tmp_path, image):
    api = FakeAPI()
    result = make_client(tmp_path, api).segment_image(image, "orange ball")
    assert result.outcome.status == "completed"
    (box,) = [r for r in result.records if r.kind == "box"]
    (mask,) = [r for r in result.records if r.kind == "mask"]
    assert box.confidence == pytest.approx(0.93)
    assert (mask.bounds.left, mask.bounds.top, mask.bounds.right, mask.bounds.bottom) == (10, 20, 15, 25)

    replay = make_client(tmp_path, api, offline=True).segment_image(image, "orange ball")
    assert api.calls["responses"] == 1
    assert replay.records == result.records


def test_offline_cache_miss_raises(tmp_path, image):
    api = FakeAPI()
    with pytest.raises(CacheMissError):
        make_client(tmp_path, api, offline=True).segment_image(image, "orange ball")
    assert api.calls["responses"] == 0


def test_incomplete_segmentation_is_not_cached(tmp_path, image):
    api = FakeAPI(sam_status="incomplete")
    with pytest.raises(SegmentationError):
        make_client(tmp_path, api).segment_image(image, "orange ball")
    with pytest.raises(CacheMissError):
        make_client(tmp_path, api, offline=True).segment_image(image, "orange ball")


def test_segment_video_uploads_once_and_tracks_frames(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake mp4")
    api = FakeAPI(sam_text=SAM_LINE.format(frame=0) + SAM_LINE.format(frame=1))
    client = make_client(tmp_path, api)
    result = client.segment_video(video, "orange ball")
    frames = sorted(r.frame.frame_index for r in result.records if r.kind == "mask")
    assert frames == [0, 1]
    client.segment_video(video, "orange ball")
    assert api.calls == {"chat": 0, "responses": 1, "files": 1}


def test_missing_api_key_is_reported(tmp_path):
    client = MetaClient(settings=Settings(api_key=None, cache_dir=tmp_path))
    with pytest.raises(RuntimeError, match="MODEL_API_KEY"):
        client.chat([{"role": "user", "content": "hi"}])
