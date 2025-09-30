### Deploy to Render

Production deployment is configured for Render using Gunicorn and a blueprint file.

---

### Prerequisites
- GitHub repository containing this project
- Render account (`https://render.com`)

---

### Included Files for Render
- `render.yaml` — service definition (env, build/start commands, env vars)
- `Procfile` — Gunicorn start command
- `runtime.txt` — pins Python version
- `requirements.txt` — uses `opencv-python-headless` for servers

---

### Deploy Steps
1) Push all files to a GitHub repository.
2) On Render, click New → Blueprint and connect the repo.
3) Render detects `render.yaml` and creates a Web Service.
4) Confirm and create resources; the build will:
   - Install deps: `pip install -r requirements.txt`
   - Start with Gunicorn: `gunicorn app:app --workers=1 --threads=1 --timeout=120 --bind 0.0.0.0:$PORT`
5) Open the service URL once live.

---

### Model File (`model1_best.h5`)
- Must be available at the repository root at runtime.
- If large, host it and download during build. Example in `render.yaml`:
```yaml
buildCommand: |
  pip install -r requirements.txt
  curl -L -o model1_best.h5 "https://YOUR_HOSTED_URL/model1_best.h5"
```

---

### Environment
From `render.yaml`:
- `PYTHON_VERSION=3.11.9`
- `FLASK_DEBUG=0`

The app reads the port from `PORT` (automatically provided by Render). Do not hardcode ports in production.

---

### Health Check
- Use `GET /test` to verify the model loads and inference works. Response includes a synthetic prediction and per-class probabilities.

---

### Notes & Tips
- Threads/workers are intentionally conservative for CPU-only instances.
- `opencv-python-headless` avoids issues on headless Linux servers.
- If you need more concurrency, scale vertically or adjust `workers`/`threads` after monitoring memory/CPU.

---

### Scaling
- Default plan is `free`. You can change plan in `render.yaml` or in the Render dashboard. Increase resources if you see timeouts or OOM.
