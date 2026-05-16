# Security Policy

## Supported Versions

ClimateVision is under active development. Security fixes are applied to the latest release on the `main` branch.

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

Instead, please report them privately using GitHub's built-in Security Advisory system:

- Go to the [Security tab](https://github.com/Climate-Vision/ClimateVision/security) of this repository.
- Click **"Report a vulnerability"**.
- Fill out the form with a description of the issue, steps to reproduce, and (if known) a suggested fix.

You should receive an initial response within **5 business days**. If the issue is confirmed, we will work on a fix and coordinate disclosure with you.

## Scope

**In scope:**

- Vulnerabilities in the ClimateVision API (`src/climatevision/api/`)
- Vulnerabilities in the React dashboard (`frontend/`)
- Vulnerabilities in the data pipeline, model inference, or authentication flow
- Dependency vulnerabilities not already tracked by Dependabot

**Out of scope:**

- Issues in third-party services (Google Earth Engine, MLflow, etc.) — please report those upstream
- Self-inflicted issues from running with debug or development configuration in production
- Missing security best-practices without a demonstrated exploit

## Disclosure Policy

We follow a coordinated disclosure model. After a fix is released, we will publish a GitHub Security Advisory crediting the reporter (unless anonymity is requested).

Thank you for helping keep ClimateVision and its users safe.
