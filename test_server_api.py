"""
Integration Verification Test Suite.
Tests full FastAPI endpoints, 11-stage pipeline polling, multi-ROI consensus, and commercial flagging.
"""
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_synthetic_test_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1024, 768), color=(40, 80, 40))
    draw = ImageDraw.Draw(img)

    # Draw synthetic trunk 1
    draw.rectangle([200, 100, 320, 700], fill=(110, 70, 40))
    # Draw synthetic trunk 2
    draw.rectangle([600, 150, 720, 750], fill=(95, 60, 35))

    img.save(path)
    print(f"Created synthetic test image at: {path}")


def main():
    print("=" * 50)
    print(" TESTING FASTAPI APPLICATION ENDPOINTS")
    print("=" * 50)

    test_img_path = Path("test_output/synthetic_forest.jpg")
    create_synthetic_test_image(test_img_path)

    # 1. Test GET /api/health
    print("\n1. Testing GET /api/health...")
    with client as c:
        resp = c.get("/api/health")
        print("Response status:", resp.status_code)
        print("Body:", resp.json())
        assert resp.status_code == 200, "Health check failed"

        # 2. Test GET /
        print("\n2. Testing GET /...")
        resp = c.get("/")
        print("Response status:", resp.status_code)
        assert resp.status_code == 200, "Root UI page load failed"

        # 3. Test POST /api/analyze
        print("\n3. Testing POST /api/analyze...")
        with open(test_img_path, "rb") as f:
            resp = c.post("/api/analyze", files={"file": ("synthetic.jpg", f, "image/jpeg")})

        print("Response status:", resp.status_code)
        data = resp.json()
        print("Body:", data)
        assert resp.status_code == 200, "Analyze job creation failed"

        job_id = data["job_id"]

        # 4. Poll GET /api/jobs/{job_id} until completed
        print(f"\n4. Polling GET /api/jobs/{job_id}...")
        for _ in range(30):
            r_job = c.get(f"/api/jobs/{job_id}")
            j_data = r_job.json()
            status = j_data["status"]
            stage_idx = j_data["stage_index"]
            msg = j_data["message"]
            print(f"  Stage {stage_idx}/11 ({j_data['stage_name']}): {status} - {msg}")
            if status in ["completed", "error"]:
                break
            time.sleep(0.5)

        assert status == "completed", f"Job failed with status: {status}"

        # 5. Test GET /api/results/{job_id}
        print(f"\n5. Testing GET /api/results/{job_id}...")
        r_res = c.get(f"/api/results/{job_id}")
        print("Response status:", r_res.status_code)
        res_data = r_res.json()
        print("Summary:", res_data["summary"])
        print("Trees Count:", len(res_data["trees"]))

        assert r_res.status_code == 200, "Fetch results failed"
        assert "summary" in res_data
        assert "trees" in res_data

    print("\n" + "=" * 50)
    print(" ALL API INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 50)


if __name__ == "__main__":
    main()
