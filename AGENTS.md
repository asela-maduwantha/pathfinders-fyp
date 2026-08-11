# Repository Guidelines

## Project Structure & Module Organization

The FastAPI application lives in `app/`. `app/main.py` assembles routes and serves the UI from `app/templates/` and `app/static/`. Domain modules are grouped by feature: `app/inference/` handles tree detection and species classification, while `app/maturity/`, `app/timber/`, and `app/auction/` contain routes, configuration, and service layers for their respective pipelines. Deployment artifacts belong in `models/`; reference data belongs in `data/`. Keep research utilities in `tools/`, exploratory work in `notebooks/`, and generated results in the existing `outputs/`, `reports/`, or `logs/` directories.

## Build, Test, and Development Commands

Create and activate a Windows environment with `python -m venv .venv` and `.venv\Scripts\activate`, then install dependencies using `pip install -r requirements.txt`.

- `.venv\Scripts\python.exe verify_models.py` checks required checkpoint files and metadata.
- `.venv\Scripts\python.exe app.py` starts the local server at `http://127.0.0.1:8000`.
- `.venv\Scripts\python.exe test_pipeline.py` runs the end-to-end image pipeline smoke test.
- `.venv\Scripts\python.exe test_server_api.py` exercises server API endpoints.
- `modal deploy modal_app.py` deploys the production service to Modal.

Some maturity models download on first use, so initial setup and tests may require network access and substantial disk space.

## Coding Style & Naming Conventions

Follow standard Python conventions: four-space indentation, `snake_case` for modules, functions, and variables, and `PascalCase` for classes. Add type hints to public service interfaces where practical. Keep route handlers thin; place inference, image processing, and business logic in the relevant `services/` or `app/inference/` module. Use `UPPER_SNAKE_CASE` for constants and environment variables such as `SMARTTIMBER_DEVICE`. No formatter is currently configured, so match nearby code and keep imports organized.

## Testing Guidelines

Tests are executable Python smoke-test scripts rather than a configured pytest suite. Name new test files `test_<feature>.py`, make failures return a non-zero exit status, and avoid overwriting checked-in samples under `test_output/`. Validate the smallest relevant module first, then run both root test scripts for cross-pipeline changes.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Add core app and tooling`. Keep each commit focused and avoid committing generated uploads, results, logs, or downloaded weights. Pull requests should explain the affected pipeline, list verification commands, link related issues, and include screenshots for UI changes or representative metrics for model-behavior changes. Do not modify trained model artifacts unless the change explicitly updates a model release.
