#!/usr/bin/env bash
# download-weights.sh — Setup for downloading SAM3 model weights
#
# Usage:
#   ./download-weights.sh                        # downloads model weights
#   ./download-weights.sh --yes                  # run in non-interactive mode assuming yes for all options
#
# Environment variables (alternative to flags):
#   SAM3_MODEL_DIR=model/sam3         where to place model weights
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR_ROOT="${SAM3_MODEL_DIR:-$REPO_DIR}"
MODEL_DIR=$MODEL_DIR_ROOT/sam3

AUTO_YES=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --yes)            AUTO_YES=true ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── Colours ───────────────────────────────────────────────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[1;34m'; NC='\033[0m'
step()  { echo -e "\n${B}══ $* ${NC}"; }
info()  { echo -e "  ${G}✓${NC} $*"; }
warn()  { echo -e "  ${Y}⚠${NC}  $*"; }
die()   { echo -e "  ${R}✗${NC}  $*"; exit 1; }

echo -e "${G}"
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║  Downloading SAM3 Model Weights                  ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "  Model dir : $MODEL_DIR"
echo ""

cd "$REPO_DIR"

WEIGHT_FILE="$MODEL_DIR/model.safetensors"

if [[ -f "$WEIGHT_FILE" ]]; then
    SIZE=$(du -sh "$WEIGHT_FILE" | cut -f1)
    info "Weights already present ($WEIGHT_FILE, $SIZE)"
else
    echo "  Could not find model weights file: $WEIGHT_FILE"
    echo "  SAM3 model weights (~3.3 GB) are required."
    echo "  Config/tokenizer files are already included."
    echo ""
    # huggingface_hub >=1.0 ships the new `hf` CLI and removes huggingface-cli.
    # Older versions only ship huggingface-cli. Pick whichever is available.
    if command -v hf >/dev/null 2>&1; then HF=hf; else HF=huggingface-cli; fi

    echo "  Option A — Official (requires HuggingFace account + accepted terms):"
    echo "    https://huggingface.co/facebook/sam3"
    echo "    $HF download facebook/sam3 model.safetensors --local-dir $MODEL_DIR"
    echo ""
    echo "  Option B — Community mirror (no account needed, same weights):"
    echo "    $HF download 1038lab/sam3 sam3.safetensors --local-dir $MODEL_DIR"
    echo "    mv $MODEL_DIR/sam3.safetensors $MODEL_DIR/model.safetensors"
    echo ""
    if $AUTO_YES; then
        yn="Y"
        echo "  Download via Option B now? [Y/n] Y  (auto-yes)"
    else
        read -rp "  Download via Option B now? [Y/n] " yn
        yn="${yn:-Y}"
    fi
    if [[ "$yn" =~ ^[Yy] ]]; then
        mkdir -p "$MODEL_DIR"
        cp $REPO_DIR/model-configs/* $MODEL_DIR
        
        # --local-dir-use-symlinks was removed in huggingface_hub 1.0; only pass
        # to old huggingface-cli.
        if [[ "$HF" == "huggingface-cli" ]]; then
            "$HF" download 1038lab/sam3 sam3.safetensors \
                --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
        else
            "$HF" download 1038lab/sam3 sam3.safetensors --local-dir "$MODEL_DIR"
        fi
        mv "$MODEL_DIR/sam3.safetensors" "$MODEL_DIR/model.safetensors"
        info "Weights downloaded → $WEIGHT_FILE"
    else
        warn "Skipping weights — place model.safetensors in $MODEL_DIR then re-run"
        exit 0
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Weights downloaded
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "Model weights downloaded!"
echo ""
