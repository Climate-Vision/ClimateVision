#!/bin/bash
# Load environment variables from .env and set them as Fly.io secrets.
# Usage: ./scripts/setup_fly_secrets.sh

set -e

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "Error: .env file not found. Copy .env.example to .env and fill in your credentials."
    exit 1
fi

# Ensure the GEE key file exists if referenced
GEE_KEY_PATH=$(grep '^GEE_SERVICE_ACCOUNT_KEY=' .env | cut -d '=' -f2- | tr -d '"' | tr -d "'")
if [ -n "$GEE_KEY_PATH" ] && [ ! -f "$GEE_KEY_PATH" ]; then
    echo "Warning: GEE_SERVICE_ACCOUNT_KEY points to a missing file: $GEE_KEY_PATH"
fi

# Read non-empty, non-comment lines and build a single fly secrets set command
SECRETS=""
while IFS= read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    case "$line" in
        \#*|""|*\ #*) continue ;;
    esac

    # Skip lines without an equals sign
    case "$line" in
        *=*) ;;
        *) continue ;;
    esac

    key="${line%%=*}"
    value="${line#*=}"

    # Strip surrounding quotes if present
    value=$(echo "$value" | sed -e 's/^["'"'"']//' -e 's/["'"'"']$//')

    # Skip empty values
    [ -z "$value" ] && continue

    SECRETS="$SECRETS $key=$value"
done < .env

if [ -z "$SECRETS" ]; then
    echo "No non-empty secrets found in .env."
    exit 0
fi

echo "Setting Fly.io secrets..."
# shellcheck disable=SC2086
fly secrets set $SECRETS

echo "Done. Verify with: fly secrets list"
