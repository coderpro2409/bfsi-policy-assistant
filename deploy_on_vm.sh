#!/usr/bin/env bash
# One-shot deploy for the BFSI app + OCR app on a fresh Ubuntu 22.04 VM
# (e.g. an Oracle Cloud Always Free ARM instance). Self-hosted, Ollama only.
#
# Usage (on the VM, after `ssh ubuntu@<vm-ip>`):
#   curl -fsSL https://raw.githubusercontent.com/coderpro2409/bfsi-policy-assistant/main/deploy_on_vm.sh | bash
# or copy this file over and: bash deploy_on_vm.sh
set -euo pipefail

# Small, CPU-friendly model. Bump to llama3 (8B) if the VM has >= 16GB RAM.
LLM_MODEL="${LLM_MODEL:-llama3.2:3b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

echo "==> Installing Docker"
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi

echo "==> Opening firewall ports 8000 (bfsi) and 7860 (ocr)"
# Oracle Ubuntu images block ports in iptables by default; open them.
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT || true
sudo iptables -I INPUT -p tcp --dport 7860 -j ACCEPT || true
sudo netfilter-persistent save 2>/dev/null || true

echo "==> Cloning repos"
cd "$HOME"
[ -d bfsi-policy-assistant ] || git clone https://github.com/coderpro2409/bfsi-policy-assistant.git
[ -d ocr ] || git clone https://github.com/coderpro2409/ocr.git

echo "==> Starting BFSI (Ollama + app) via docker compose"
cd "$HOME/bfsi-policy-assistant"
cat > .env <<EOF
LLM_MODEL=${LLM_MODEL}
EMBEDDING_MODEL=${EMBEDDING_MODEL}
EOF
sudo docker compose up -d --build

echo "==> Pulling Ollama models (this downloads a few GB, please wait)"
sudo docker compose exec -T ollama ollama pull "${LLM_MODEL}"
sudo docker compose exec -T ollama ollama pull "${EMBEDDING_MODEL}"

echo "==> Starting OCR app"
cd "$HOME/ocr"
sudo docker build -t ocr-app .
sudo docker rm -f ocr-app 2>/dev/null || true
sudo docker run -d --name ocr-app --restart unless-stopped -p 7860:7860 ocr-app

IP="$(curl -fsSL ifconfig.me 2>/dev/null || echo '<vm-ip>')"
echo
echo "==> Done."
echo "    BFSI Policy Assistant:  http://${IP}:8000"
echo "    OCR app:                http://${IP}:7860"
echo
echo "Note: first answers from BFSI are slow (CPU model load). That is normal."
