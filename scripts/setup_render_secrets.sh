#!/bin/bash
# Upload ClimateVision secrets to Render.com.
# Usage: ./scripts/setup_render_secrets.sh

set -e

cd "$(dirname "$0")/.."

SERVICE_NAME="climatevision-green"

if ! command -v render &> /dev/null; then
    echo "Error: Render CLI not found. Install it from https://render.com/docs/cli"
    exit 1
fi

if [ ! -f .env ]; then
    echo "Error: .env file not found. Copy .env.example to .env and fill it in."
    exit 1
fi

if [ ! -f secrets/gee-key.json ]; then
    echo "Error: secrets/gee-key.json not found. Place your GEE service-account key there."
    exit 1
fi

echo "Uploading secrets to Render service: $SERVICE_NAME"

# Read values from .env and upload
while IFS= read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    case "$line" in
        \#*|""|*\ #*) continue ;;
    esac

    case "$line" in
        *=*) ;;
        *) continue ;;
    esac

    key="${line%%=*}"
    value="${line#*=}"
    value=$(echo "$value" | sed -e 's/^["'"'"']//' -e 's/["'"'"']$//')

    [ -z "$value" ] && continue

    # Skip the file-path key; we will upload the file contents separately
    if [ "$key" = "GEE_SERVICE_ACCOUNT_KEY" ]; then
        continue
    fi

    echo "Setting $key"
    render env set "$SERVICE_NAME" "$key=$value" --yes

done < .env

# Upload the GEE key JSON contents as GEE_SERVICE_ACCOUNT_KEY_JSON
GEE_KEY_JSON=$(cat secrets/gee-key.json)
echo "Setting GEE_SERVICE_ACCOUNT_KEY_JSON"
render env set "$SERVICE_NAME" "GEE_SERVICE_ACCOUNT_KEY_JSON=$GEE_KEY_JSON" --yes

echo "Done. Verify with: render env list $SERVICE_NAME"
