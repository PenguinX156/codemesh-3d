"""API tests against the real app (store, worker thread, ingest) in a temp jobs dir."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_ingest import make_images, make_video

from codemesh.jobs.models import Job
from codemesh.jobs.store import JobStore
from codemesh.main import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(jobs_dir=tmp_path / "jobs")
    with TestClient(app) as c:
        yield c


def _wait(client: TestClient) -> None:
    assert client.app.state.runner.wait_idle(60)


def _upload_video(client: TestClient, tmp_path: Path, **form) -> dict:
    video = make_video(tmp_path / "clip.mp4", n=48, sharp_every=None)
    res = client.post("/api/jobs", files=[("files[]", ("clip.mp4", video.read_bytes(), "video/mp4"))],
                      data={"mode": "walkaround", "preset": "draft", **form})
    assert res.status_code == 201, res.text
    return res.json()


def test_health_and_capabilities(client: TestClient) -> None:
    assert client.get("/api/health").json()["ok"] is True
    caps = client.get("/api/capabilities").json()
    assert {"geometry_engine", "device", "precision", "blender", "meta_api"} <= caps.keys()
    assert "META_API_KEY" not in str(caps)


def test_video_job_flow(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(*args, **kwargs):
        raise RuntimeError("SAM is offline in tests")

    monkeypatch.setattr("codemesh.meta.sam.segment_video", offline)  # never call the paid API from tests
    job = _upload_video(client, tmp_path)
    assert job["input"]["kind"] == "video" and job["settings"]["preset"] == "draft"
    _wait(client)
    job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "ready_to_select", job
    assert job["stages"][0]["name"] == "ingest" and job["stages"][0]["metrics"]["frames"] == 24
    frames = client.get(f"/api/jobs/{job['id']}/frames").json()
    assert len(frames) == 24 and frames[0]["url"].endswith("/artifacts/frames/0000.jpg")
    img = client.get(frames[0]["url"])
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"
    assert client.get(frames[0]["thumb_url"]).status_code == 200
    assert [j["id"] for j in client.get("/api/jobs").json()] == [job["id"]]

    # Segment (SAM stubbed offline) and reconstruct: accepted, validated, then fail with a clear message.
    assert client.post(f"/api/jobs/{job['id']}/select", json={"frame_index": 99, "text": "mug"}).status_code == 422
    assert client.post(f"/api/jobs/{job['id']}/select", json={"frame_index": 0}).status_code == 422
    res = client.post(f"/api/jobs/{job['id']}/select",
                      json={"frame_index": 0, "points": [{"x": 0.5, "y": 0.5, "label": 1}]})
    assert res.status_code == 200 and res.json()["status"] == "segmenting"
    _wait(client)
    job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "failed" and "SAM is offline" in job["error"]
    assert job["settings"]["selection"]["points"][0]["label"] == 1

    assert client.post(f"/api/jobs/{job['id']}/reconstruct", json={"texture_size": 1000}).status_code == 422
    res = client.post(f"/api/jobs/{job['id']}/reconstruct", json={"preset": "high", "triangle_budget": 20000})
    assert res.status_code == 200
    _wait(client)
    job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "failed" and "geometry" in job["error"]
    assert client.post(f"/api/jobs/{job['id']}/refine", json={}).status_code == 409

    assert client.delete(f"/api/jobs/{job['id']}").json() == {"ok": True}
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert not (client.app.state.jobs_dir / job["id"]).exists()


def test_image_job_and_filename_sanitizing(client: TestClient, tmp_path: Path) -> None:
    paths = make_images(tmp_path / "imgs", n=2)
    files = [("files[]", (f"../../evil{i}.jpg", p.read_bytes(), "image/jpeg")) for i, p in enumerate(paths)]
    res = client.post("/api/jobs", files=files, data={"mode": "turntable"})
    assert res.status_code == 201, res.text
    job = res.json()
    assert job["input"]["files"] == ["000_evil0.jpg", "001_evil1.jpg"] and job["mode"] == "turntable"
    _wait(client)
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "ready_to_select"
    assert not (client.app.state.jobs_dir.parent / "evil0.jpg").exists()


@pytest.mark.parametrize(
    ("files", "status"),
    [
        ([("x.mp4", b"not a video at all")], 415),  # extension lies
        ([("x.jpg", b"\x89PNG garbage")], 400),  # one image is not enough
        ([("a.jpg", b"garbage"), ("b.jpg", b"garbage")], 415),
        ([("tool.exe", b"MZ")], 415),
        ([], 400),
    ],
)
def test_upload_rejections(client: TestClient, files: list, status: int) -> None:
    res = client.post("/api/jobs", files=[("files[]", (n, io.BytesIO(b), "application/octet-stream")) for n, b in files],
                      data={"mode": "walkaround"})
    assert res.status_code == status, res.text
    assert client.get("/api/jobs").json() == []
    assert [p.name for p in client.app.state.jobs_dir.iterdir() if p.is_dir()] == []


def test_upload_size_limit(tmp_path: Path) -> None:
    with TestClient(create_app(jobs_dir=tmp_path / "jobs", max_upload_bytes=1000)) as c:
        res = c.post("/api/jobs", files=[("files[]", ("v.mp4", b"0" * 5000, "video/mp4"))])
        assert res.status_code == 413


def test_bad_mode_and_traversal(client: TestClient, tmp_path: Path) -> None:
    job = _upload_video(client, tmp_path)
    _wait(client)
    assert client.post("/api/jobs", data={"mode": "orbit"}, files=[("files[]", ("v.mp4", b"x", "video/mp4"))]).status_code == 422
    base = f"/api/jobs/{job['id']}/artifacts/"
    for bad in ("..%2Fjobs.db", "%2e%2e/jobs.db", "frames/..%2F..%2Fjobs.db", "C:%2FWindows%2Fwin.ini", "nope.txt"):
        assert client.get(base + bad).status_code == 404, bad
    assert client.get("/api/jobs/not-a-job").status_code == 404
    assert client.get("/api/jobs/..%2F..%2Fx/frames").status_code == 404


def test_startup_marks_interrupted_jobs_failed(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    store = JobStore(jobs_dir / "jobs.db")
    job = store.create(Job(status="reconstructing"))
    store.close()
    with TestClient(create_app(jobs_dir=jobs_dir)) as c:
        assert c.get(f"/api/jobs/{job.id}").json()["status"] == "failed"
