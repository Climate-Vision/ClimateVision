# ClimateVision Deployment Guide

This guide covers deploying ClimateVision at `https://climatevision.green` with real Google Earth Engine (GEE) satellite data.

The primary path uses **Render.com** because it offers a reliable free tier and does not require a paid account to launch. A **Fly.io** path is also documented for users with active billing.

---

## Prerequisites

- A Google Cloud project with Earth Engine enabled
- This repository cloned and dependencies installed
- One of the following platforms:
  - **Render.com** account: https://render.com
  - **Fly.io** account with active billing: https://fly.io

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

## 3. Deploy to Render.com (recommended, free tier)

### 3.1 Install the Render CLI

https://render.com/docs/cli

### 3.2 Create secrets locally

Make sure `.env` and `secrets/gee-key.json` are populated as described in section 1.

### 3.3 Push secrets to Render

```bash
./scripts/setup_render_secrets.sh
```

This uploads:
- `GEE_SERVICE_ACCOUNT`
- `GEE_PROJECT_ID`
- `GEE_SERVICE_ACCOUNT_KEY_JSON` (the contents of your JSON key)
- `API_SECRET_KEY`
- `CLIMATEVISION_ALLOW_DEV_KEY`
- `CLIMATEVISION_CORS_ORIGINS`
- `DATABASE_URL`

### 3.4 Deploy via Blueprint

1. Push `render.yaml` to your default branch (`main`).
2. In the Render dashboard, go to **Blueprints**.
3. Connect your GitHub repository.
4. Render will read `render.yaml` and create the `climatevision-green` web service.

### 3.5 Custom domain

1. In the Render dashboard, open the `climatevision-green` service.
2. Go to **Settings → Custom Domains**.
3. Add `climatevision.green` and `www.climatevision.green`.
4. Render will give you DNS records to add in Namecheap.
5. In Namecheap, add the provided CNAME or A records.

### 3.6 Limitations of the free tier

- The service spins down after 15 minutes of inactivity. The first request after spin-down will have a ~30-second cold start.
- SQLite data lives on ephemeral disk. It will survive restarts but not full redeploys. For production persistence, switch to a managed PostgreSQL database.
- Memory is limited; large model inference may be slow.

---

## 4. Deploy to Fly.io (requires active billing)

Use this path only if your Fly.io account has an up-to-date payment method.

```bash
# Create the Fly app
fly apps create climatevision-green

# Create a persistent volume for SQLite
fly volumes create climatevision_data --region lhr --size 1

# Upload secrets
./scripts/setup_fly_secrets.sh

# Set up SSL
fly certs create climatevision.green

# Add DNS records in Namecheap using the output of:
fly certs show climatevision.green

# Add FLY_API_TOKEN to GitHub Actions secrets, then deploy:
git push origin main
```

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

For email, configure an SMTP provider or a transactional email service such as Mailgun or SendGrid, then set the relevant secrets via your platform’s secret manager.

---

## 8. Security checklist

- [ ] `CLIMATEVISION_ALLOW_DEV_KEY` is `0` in production.
- [ ] `API_SECRET_KEY` is a strong, unique secret.
- [ ] The GEE service-account key is stored only in `secrets/` or your platform’s secret store, never in Git.
- [ ] Old or exposed service-account keys have been deleted in Google Cloud Console.
- [ ] The production domain serves HTTPS.
