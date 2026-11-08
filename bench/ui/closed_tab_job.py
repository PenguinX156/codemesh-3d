"""Start a real job on the hosted app, then disconnect as if the user closed the tab.

    python bench/ui/closed_tab_job.py <video> "<object text>" [base-url]

Uploads the video, selects the object by text, starts the 3D build and exits immediately. Check later
with `python bench/ui/closed_tab_job.py --status <job_id>`: "done" means the job finished with no client
connected (the server kept itself alive); "failed" with an interrupted error means it was shut down mid-job.
Costs one real job (Meta API about $0.01, Modal GPU/CPU time).
"""

import sys
import time
from pathlib import Path

import httpx

BASE = "https://fjziemski--codemesh-web.us-east.modal.run"


def wait_for(c: httpx.Client, job_id: str, statuses: set[str], timeout: float = 600) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = c.get(f"/api/jobs/{job_id}").raise_for_status().json()
        if job["status"] in statuses:
            return job
        if job["status"] == "failed":
            sys.exit(f"job failed: {job.get('error')}")
        time.sleep(3)
    sys.exit(f"timed out waiting for {statuses}")


def main() -> None:
    if sys.argv[1] == "--status":
        base = sys.argv[3] if len(sys.argv) > 3 else BASE
        job = httpx.get(f"{base}/api/jobs/{sys.argv[2]}", timeout=120).raise_for_status().json()
        print(job["status"], job.get("stage"), job.get("error") or "", job.get("metrics", {}).get("iou_mean"))
        return
    video, text = Path(sys.argv[1]), sys.argv[2]
    base = sys.argv[3] if len(sys.argv) > 3 else BASE
    with httpx.Client(base_url=base, timeout=600) as c:
        with video.open("rb") as f:
            files = {"files[]": (video.name, f, "video/mp4")}
            job = c.post("/api/jobs", files=files, data={"mode": "walkaround", "preset": "standard"}).raise_for_status().json()
        job_id = job["id"]
        print("job", job_id, flush=True)
        wait_for(c, job_id, {"ready_to_select"})
        frames = c.get(f"/api/jobs/{job_id}/frames").raise_for_status().json()
        best = next(f["index"] for f in frames if f["best"])
        c.post(f"/api/jobs/{job_id}/select", json={"frame_index": best, "text": text}).raise_for_status()
        wait_for(c, job_id, {"ready_to_reconstruct"})
        c.post(f"/api/jobs/{job_id}/reconstruct", json={"preset": "standard"}).raise_for_status()
    print("build started at", time.strftime("%H:%M:%S"), "- disconnected (tab closed)", flush=True)


if __name__ == "__main__":
    main()
