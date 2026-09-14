#!/usr/bin/env sh
set -eu
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if ! command -v docker >/dev/null 2>&1; then
  case "$(uname -s)" in
    Darwin)
      command -v brew >/dev/null 2>&1 || { echo "Homebrew is required for automatic Docker Desktop installation; Python is not required." >&2; exit 1; }
      brew install --cask docker
      open -a Docker
      echo "Docker Desktop installed. Finish its first-run setup, then rerun this file." >&2
      exit 1
      ;;
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y docker.io docker-compose-v2 || sudo apt-get install -y docker.io docker-compose-plugin
      elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y docker docker-compose-plugin
      else
        echo "No supported package manager found for automatic Docker installation; Python is not required." >&2
        exit 1
      fi
      sudo systemctl enable --now docker
      ;;
  esac
fi
docker info >/dev/null 2>&1 || { echo "Docker is not running." >&2; exit 1; }
ENV_FILE="$ROOT_DIR/deploy/.env"
if [ ! -f "$ENV_FILE" ]; then
  seed="$(date +%s)-$$-${RANDOM:-0}"
  pg="pg-$(printf '%s-pg' "$seed" | cksum | awk '{print $1}')-$(printf '%s' "$seed" | cksum | awk '{print $1}')"
  redis="redis-$(printf '%s-redis' "$seed" | cksum | awk '{print $1}')-$(printf '%s-x' "$seed" | cksum | awk '{print $1}')"
  minio="minio-$(printf '%s-minio' "$seed" | cksum | awk '{print $1}')-$(printf '%s-y' "$seed" | cksum | awk '{print $1}')"
  sed -e "s/change-this-postgres-password/$pg/" -e "s/change-this-redis-password/$redis/" -e "s/change-this-minio-password/$minio/" "$ROOT_DIR/deploy/.env.example" > "$ENV_FILE"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  docker compose --env-file "$ENV_FILE" -f "$ROOT_DIR/deploy/compose.yaml" -f "$ROOT_DIR/deploy/compose.gpu.yaml" up -d --build
else
  docker compose --env-file "$ENV_FILE" -f "$ROOT_DIR/deploy/compose.yaml" up -d --build
fi
echo "Football Insight is available at http://localhost:8000"
