# ClimateVision Deployment Guide

This guide covers deploying ClimateVision to **Fly.io** at `https://climatevision.green` with real Google Earth Engine (GEE) satellite data.

---

## Prerequisites

- A Fly.io account: https://fly.io
- The Fly CLI (`flyctl`) installed and authenticated
- A Google Cloud project with Earth Engine enabled
- This repository cloned and dependencies installed

---

## 1. Google Earth Engine credentials

Real satellite imagery requires a GEE service account.

1. Go to https://console.cloud.google.com/ and select your project.
2. Enable the **Earth Engine API**.
3. Navigate to **IAM & Admin → Service Accounts**.
4. Create a service account with the role **Earth Engine Resource Writer**.
5. Create a JSON key and download it.
6. Place the key at `secrets/gee-key.json` (the `secrets/` directory is gitignored).
7. Register the service account in the Earth Engine console: https://code.earthengine.google.com/
8. Copy `.env.example` to `.env` and fill in:
   ```bash
   GEE_SERVICE_ACCOUNT=your-service-account@your-project.iam.gserviceaccount.com
   GEE_SERVICE_ACCOUNT_KEY=secrets/gee-key.json
   GEE_PROJECT_ID=your-gee-project-id
   ```

> **Never commit `secrets/` or `.env` to Git.**

---

## 2. Local production-like run (Docker)

Build and run the full stack locally:

```bash
# Make sure .env is populated and secrets/gee-key.json exists
docker compose up --build
```

The API will be available at http://localhost:8000 and the dashboard at http://localhost:8000/.

---

## 3. Fly.io setup

### 3.1 Create the app and provision secrets

```bash
# Create the Fly app
fly apps create climatevision-green

# Set all secrets from .env
./scripts/setup_fly_secrets.sh

# Create a persistent volume for SQLite and outputs (1 GB is plenty for demos)
fly volumes create climatevision_data --region lhr --size 1
```

### 3.2 Custom domain

```bash
# Request a certificate for your domain
fly certs create climatevision.green
```

Add the DNS records shown by `fly certs show` to your DNS provider. Once propagated, Fly will serve the app at `https://climatevision.green`.

### 3.3 Deploy

```bash
# Manual deploy
fly deploy --remote-only

# Or push to main — GitHub Actions will deploy automatically
# once FLY_API_TOKEN is configured (see section 4).
```

---

## 4. GitHub Actions CI/CD

The repository includes two workflows:

- `.github/workflows/ci.yml` — runs on every PR/push to `main` or `develop`.
- `.github/workflows/deploy.yml` — deploys to Fly.io on every push to `main`.

### Required repository secrets

Add these in **GitHub → Settings → Secrets and variables → Actions**:

| Secret | Value |
|--------|-------|
| `FLY_API_TOKEN` | A Fly.io access token from https://fly.io/user/personal_access_tokens |

GEE credentials are **not** stored in GitHub; they are set directly on Fly via `scripts/setup_fly_secrets.sh`.

---

## 5. Verify the deployment

Check the health endpoint:

```bash
curl https://climatevision.green/api/health
```

Check which models have trained weights:

```bash
curl https://climatevision.green/api/health/models
```

Open the dashboard:

```bash
open https://climatevision.green
```

---

## 6. Model status for investor demos

As of this deployment, only the **flooding** analysis ships with trained weights (`models/flooding_20260513_124208/model.onnx`).

Deforestation and arctic-ice analyses run using synthetic fallback tiles and untrained weights. The API reports this transparently via `/api/health/models` and the `is_synthetic` flag in prediction responses.

To add production models:

1. Train or obtain `models/unet_deforestation.pth` and `models/unet_ice.pth`.
2. Upload them to the `models/` directory.
3. Redeploy.

---

## 7. Alert delivery

The alert generator emits notifications to pluggable channels. By default only the `log` channel is active.

To enable email or webhook alerts, register a delivery function in `src/climatevision/inference/alert_generator.py` or implement the alert worker (see open issue #10).

For email, configure an SMTP provider or a transactional email service such as Mailgun or SendGrid, then set the relevant secrets via `fly secrets set`.

---

## 8. Security checklist

- [ ] `CLIMATEVISION_ALLOW_DEV_KEY` is `0` in production.
- [ ] `API_SECRET_KEY` is a strong, unique secret.
- [ ] The GEE service-account key is stored only in `secrets/` or Fly secrets, never in Git.
- [ ] Hopelyn’s old PAT has been revoked in GitHub.
- [ ] The Fly app uses HTTPS via `fly certs`.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `GEE tile download failed` | Missing or invalid GEE credentials | Check `.env` / Fly secrets and service-account registration |
| Frontend shows blank page | `frontend/dist` not built or mounted | Re-run `npm run build` in `frontend/` or redeploy |
| CORS errors | Origin not allowed | Add domain to `CLIMATEVISION_CORS_ORIGINS` |
| SQLite data lost on deploy | No Fly volume mounted | Create and mount `climatevision_data` volume on `/app/outputs` |
