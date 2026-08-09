# Deploying to Modal

This app runs entirely as one FastAPI process (`app/main.py`). `modal_app.py` wraps
that same app to run on [Modal](https://modal.com), a serverless GPU/CPU hosting
platform, and gives it a public HTTPS URL. Nothing about the application code is
Modal-specific — `python app.py` still works locally exactly as before.

## What you get

- A public URL (e.g. `https://<you>--pathfinders-fyp-fastapi-app.modal.run`) serving
  every tab: Tree Detection, Tree Classification, Detection+Classification, Maturity
  Assessment, the full pipeline, Timber Grading, Grading+Bid Price, and Auction Bid
  Price.
- Runs on a T4 GPU.
- Scales to zero when idle (no traffic = no cost), and spins back up on the next
  request.
- Uploaded photos, annotated results, maturity run outputs, and the auction
  prediction history/PDF reports persist across restarts via three Modal Volumes.

## 1. Create a Modal account

Go to [modal.com](https://modal.com) and sign up (free tier is enough to deploy and
test this). No credit card is required to start; GPU usage is billed per-second once
you add billing details.

## 2. Install and authenticate the Modal CLI

From the project's virtual environment (or any Python environment on this machine):

```powershell
pip install modal
modal setup
```

`modal setup` opens your browser to log in and links this machine to your account —
it writes a token to `~/.modal.toml`. You only need to do this once per machine.

## 3. (Recommended) Add a Hugging Face token for the image build

Several models (`facebook/dinov2-small`, UniDepthV2) download from Hugging Face Hub
the first time they're used. Local testing already logged an "unauthenticated
requests" warning from the Hub — adding a free token avoids hitting anonymous rate
limits while `modal_app.py`'s image build pre-downloads these weights.

1. Create a free account at [huggingface.co](https://huggingface.co) if you don't
   have one.
2. Create a **read-only** access token: Settings → Access Tokens → New token.
3. Store it as a Modal secret:
   ```powershell
   modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```
4. In `modal_app.py`, change:
   ```python
   .run_function(_prewarm_models)
   ```
   to:
   ```python
   .run_function(_prewarm_models, secrets=[modal.Secret.from_name("huggingface-secret")])
   ```

You can skip this step and deploy without it — the build will still work using
anonymous Hugging Face access; add the secret later if a build ever fails with a
rate-limit error.

## 4. Deploy

From the project root (`D:\pathfinders-fyp`):

```powershell
modal deploy modal_app.py
```

The **first** deploy will take a while — it installs `torch`, `transformers`,
`ultralytics`, `catboost`, `xgboost`, and everything else in `requirements.txt` into
the image, copies the app code and ~255MB of model weights, and pre-downloads the
Hugging Face/Ultralytics models. Subsequent deploys after code-only changes are much
faster (Modal caches unchanged image layers).

The three volumes (`pathfinders-uploads`, `pathfinders-results`,
`pathfinders-outputs`) are created automatically on first deploy
(`create_if_missing=True` in `modal_app.py`) — no separate `modal volume create` step
needed.

When it finishes, Modal prints your public URL, e.g.:

```
✓ Created web endpoint for fastapi_app => https://your-username--pathfinders-fyp-fastapi-app.modal.run
```

## 5. Verify it's working

```powershell
curl https://<your-url>/api/health
```

should return `{"status":"online", ..., "models_loaded": true}`. Then open the URL
in a browser and try a couple of tabs — expect the **first** request after a cold
start to take a while (loading all the models onto a fresh GPU container), and
subsequent requests within the same warm container to be fast.

## Iterating

- **Redeploy after changes**: `modal deploy modal_app.py` again.
- **Tail logs**: `modal app logs pathfinders-fyp`
- **List deployed apps**: `modal app list`
- **Inspect persisted data**: `modal volume ls pathfinders-results` (etc. for the
  other two volumes)
- **Stop the app** (stops serving, keeps volumes/secrets): `modal app stop pathfinders-fyp`

## Cost/behavior notes

- **Scale-to-zero**: the container shuts down after ~5 minutes with no requests
  (`scaledown_window=300` in `modal_app.py`). The next request after that pays a
  cold-start cost (re-loading all models onto a fresh GPU container, roughly
  15-30+ seconds) before it starts responding. If you want instant responses at all
  times instead, change `min_containers=0` to `min_containers=1` in
  `modal_app.py` and redeploy — this keeps one container always running, which
  means continuous GPU billing even with zero traffic.
- **GPU billing**: you're billed per-second only while a container is actually
  running (loading models or serving a request) — not during scale-to-zero idle
  time.
- **Timeout**: each request can run up to 600 seconds (`timeout=600`) before Modal
  cancels it — generous enough for the slower pipelines (e.g. the full maturity
  assessment) even under cold-start load.
