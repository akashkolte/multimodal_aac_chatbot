#!/bin/bash
set -e

# ── Config — edit these before running ────────────────────────────────────────
RESOURCE_GROUP="aac-chatbot"
LOCATION="eastus"
REGISTRY_NAME="aacchatbotregistry"
APP_NAME="aac-chatbot"
ENV_NAME="aac-env"
IMAGE="$REGISTRY_NAME.azurecr.io/$APP_NAME:latest"

# Load from .env if present, else require them to be set in shell
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

if [ -z "$PRIMARY_API_KEY" ]; then
  echo "ERROR: PRIMARY_API_KEY not set. Add it to .env or export it."
  exit 1
fi

PRIMARY_MODEL="${PRIMARY_MODEL:-gemini-3-flash-preview}"
PRIMARY_BASE_URL="${PRIMARY_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
FALLBACK_MODEL="${FALLBACK_MODEL:-$PRIMARY_MODEL}"
FALLBACK_BASE_URL="${FALLBACK_BASE_URL:-$PRIMARY_BASE_URL}"
FALLBACK_API_KEY="${FALLBACK_API_KEY:-$PRIMARY_API_KEY}"

echo "==> Logging in to Azure..."
az login --only-show-errors

echo "==> Registering providers (safe to re-run)..."
az provider register -n Microsoft.ContainerRegistry --wait
az provider register -n Microsoft.OperationalInsights --wait
az provider register -n Microsoft.App --wait

echo "==> Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION --only-show-errors

echo "==> Creating container registry..."
az acr create \
  --name $REGISTRY_NAME \
  --resource-group $RESOURCE_GROUP \
  --sku Basic \
  --admin-enabled true \
  --only-show-errors

echo "==> Logging in to container registry..."
az acr login --name $REGISTRY_NAME

echo "==> Building and pushing image (linux/amd64)..."
docker buildx create --use --name aac-builder 2>/dev/null || docker buildx use aac-builder
docker buildx build \
  --platform linux/amd64 \
  --push \
  -t $IMAGE .

echo "==> Creating Container Apps environment..."
az containerapp env create \
  --name $ENV_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --only-show-errors

echo "==> Deploying container app..."
az containerapp create \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME \
  --image $IMAGE \
  --registry-server $REGISTRY_NAME.azurecr.io \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --only-show-errors \
  --env-vars \
    PRIMARY_MODEL="$PRIMARY_MODEL" \
    PRIMARY_BASE_URL="$PRIMARY_BASE_URL" \
    PRIMARY_API_KEY="$PRIMARY_API_KEY" \
    FALLBACK_MODEL="$FALLBACK_MODEL" \
    FALLBACK_BASE_URL="$FALLBACK_BASE_URL" \
    FALLBACK_API_KEY="$FALLBACK_API_KEY"

echo ""
echo "==> Done! Your app is live at:"
az containerapp show \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.configuration.ingress.fqdn \
  --output tsv | sed 's/^/https:\/\//'

# delete everything
# az group delete --name aac-chatbot --yes
