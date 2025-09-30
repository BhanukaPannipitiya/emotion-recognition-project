# Deploy to Render

This project is configured to deploy a Flask + TensorFlow app on Render.

## Prerequisites
- A GitHub repository containing this project
- A Render account (`https://render.com`)

## Files added for Render
- `Procfile`: Gunicorn start command
- `render.yaml`: Render service definition
- `runtime.txt`: Pin Python version
- `requirements.txt`: Uses `opencv-python-headless`

## Steps
1. Push all files to a GitHub repository.
2. On Render, click New > Blueprint and connect your repo.
3. Render will detect `render.yaml` and create a Web Service.
4. Confirm the defaults and click Create Resources.
5. Wait for build & deploy to finish.
6. Open the service URL shown by Render.

## Notes
- The app reads the port from the `PORT` environment variable (Render sets it).
- Model file `model1_best.h5` must be in the repository root. If it is large, consider storing it on a file host and downloading it during `buildCommand`:

```yaml
buildCommand: |
  pip install -r requirements.txt
  curl -L -o model1_best.h5 "https://YOUR_HOSTED_URL/model1_best.h5"
```

- If OpenCV causes issues on headless servers, ensure you are using `opencv-python-headless`.

## Environment variables
- `PYTHON_VERSION=3.11.9`
- `FLASK_DEBUG=0`

## Health check
- The server serves `/test` endpoint returning a JSON indicating the model is working.

## Scaling
- Default plan is `free`. You can change plan in `render.yaml` or in the Render dashboard.
