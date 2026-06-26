# ClimateVision — Go-Live Checklist (climatevision.green)

Work top to bottom. Anything in **🔴 Blocker** must be done before the site is
publicly announced. **🟠 Important** should be done launch day. **🟢 Nice-to-have**
can follow.

---

## 🔴 Blockers — do these first

- [ ] **Rotate the leaked GEE key.** The `climatevision-gee@kinos-473422`
      service-account private key was shared in plaintext. In Google Cloud
      Console → IAM → Service Accounts → Keys: delete that key, create a new one.
- [ ] **Set secrets as runtime env vars, never in the repo.** Run
      `./scripts/setup_render_secrets.sh` (Render) or `./scripts/setup_fly_secrets.sh`
      (Fly). Confirm none of these appear in any committed file:
      `GEE_SERVICE_ACCOUNT`, `GEE_SERVICE_ACCOUNT_KEY_JSON`, `GEE_PROJECT_ID`,
      `ANTHROPIC_API_KEY`, `API_SECRET_KEY`.
- [ ] **Confirm `CLIMATEVISION_ALLOW_DEV_KEY=0` in production.** It is set in
      `render.yaml`; verify it on Fly too. A live dev key bypasses API-key auth.
- [ ] **Decide the truth-in-labeling posture.** Until trained weights are
      deployed, predictions come from untrained/synthetic models. Either:
      (a) label the product a **technical preview** in the UI and API responses, or
      (b) deploy trained weights first (see `notebooks/TRAINING_PIPELINE.md`).
      Do **not** ship unlabeled demo output to NGOs/agencies as real monitoring.
- [ ] **Rotate the teammate Personal Access Tokens** stored in `.git/config`
      (franchaise, olufemi, and previously hopelyn). They were printed in plaintext.

---

## 🟠 Important — launch day

- [ ] **CORS matches the domain.** `render.yaml` sets
      `CLIMATEVISION_CORS_ORIGINS=https://climatevision.green,https://www.climatevision.green`.
      Confirm the deployed env has it and that the frontend is built with the
      right `VITE_API_BASE_URL` (empty = same-origin, which is correct here).
- [ ] **DNS → host.** Point `climatevision.green` (and `www`) at Render/Fly and
      verify HTTPS (`force_https = true` is set in `fly.toml`).
- [ ] **Persistent storage is mounted.** `fly.toml` mounts `climatevision_data`
      at `/app/outputs` and `DATABASE_URL=sqlite:///app/outputs/...`. Confirm the
      volume exists so run history + the NGO/alerts DB survive restarts.
      (On Render free tier, confirm the disk is attached — otherwise SQLite is
      ephemeral and you lose registrations on redeploy.)
- [ ] **Health checks green.** Hit `/api/health` and `/api/health/models` on the
      live URL. `/api/health/models` will tell you whether real weights loaded.
- [ ] **Smoke test the core flow.** One `POST /api/predict` and one dashboard
      load against the live site. Confirm `/docs` renders.
- [ ] **Merge the flood-detection PR** (committed under your name) so the live
      build includes it.

---

## 🟢 Nice-to-have — this week

- [ ] Deploy real trained weights for flooding first (best dataset fit:
      Sen1Floods11), then deforestation and ice. Follow `TRAINING_PIPELINE.md`.
- [ ] Fill in the README benchmark table (IoU / F1) once a run passes the
      governance gate — replaces the "in progress" placeholders.
- [ ] Add a status/uptime check and basic error monitoring.
- [ ] Record dataset `provenance.json` + model cards for each deployed model
      (credibility for government adoption).
- [ ] Push the 21 unpushed `main` commits and confirm CI is green on `main`.

---

## Quick verification commands (run after deploy)

```bash
curl -s https://climatevision.green/api/health | jq
curl -s https://climatevision.green/api/health/models | jq   # are real weights loaded?
curl -s https://climatevision.green/api/analysis-types | jq
```

If `/api/health/models` shows untrained/demo, you are still in preview mode —
which is fine, as long as the UI says so.
