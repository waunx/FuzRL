#!/usr/bin/env bash

set -e

# Generic launcher for ppolag / fuzppolag / cvarppolag

# Ensure we run from project root so "python scripts/train.py" and imports work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

ALGORITHM="${ALGORITHM:-ppolag}"
ENVIRONMENT="${ENVIRONMENT:-SafetyPointCircle1-v0}"
EPOCHS="${EPOCHS:-1000}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
LEVEL="${LEVEL:-11}"
EPS="${EPS:-1.0}"
TRAIN_EPS="${TRAIN_EPS:-0.5}"
DISTURB_PART="${DISTURB_PART:-multi_source}"
DISTURB_TYPE="${DISTURB_TYPE:-white_noise}"
ALPHA="${ALPHA:-0.1}"
NUM_CPU="${NUM_CPU:-1}"
USE_SWANLAB="${USE_SWANLAB:-true}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-Fuz-RL}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWANLAB_EXPERIMENT="${SWANLAB_EXPERIMENT:-${ALGORITHM}-${ENVIRONMENT}-level${LEVEL}-${TIMESTAMP}}"

show_help() {
    echo "Fuz-RL training launcher"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options (can also be overridden via env vars):"
    echo "  -a, --algorithm ALG     Algorithm (ppolag|fuzppolag|cvarppolag)"
    echo "  -e, --env ENV            Environment name (default: SafetyPointCircle1-v0)"
    echo "      --epochs N           Training epochs (default: 1000)"
    echo "  -g, --gpu ID             GPU id (default: 0)"
    echo "  -s, --seed SEED          Random seed (default: 42)"
    echo "      --level L            Fuzzy level (default: 11)"
    echo "      --eps V              Perturbation magnitude (default: 1.0)"
    echo "      --train-eps V        Training perturbation (default: 0.5)"
    echo "      --disturb-part P     Disturbance part (dynamics|observation|action|multi_source)"
    echo "      --disturb-type T     Disturbance type (white_noise|impulse|periodic)"
    echo "      --alpha A            CVaR confidence level (default: 0.1)"
    echo "      --num-cpu N          MPI num CPUs (default: 1)"
    echo "      --no-swanlab         Disable SwanLab logging"
    echo "  -h, --help               Show this help"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--algorithm)      ALGORITHM="$2"; shift 2 ;;
        -e|--env)            ENVIRONMENT="$2"; shift 2 ;;
        --epochs)            EPOCHS="$2"; shift 2 ;;
        -g|--gpu)            GPU="$2"; shift 2 ;;
        -s|--seed)           SEED="$2"; shift 2 ;;
        --level)             LEVEL="$2"; shift 2 ;;
        --eps)               EPS="$2"; shift 2 ;;
        --train-eps)         TRAIN_EPS="$2"; shift 2 ;;
        --disturb-part)      DISTURB_PART="$2"; shift 2 ;;
        --disturb-type)      DISTURB_TYPE="$2"; shift 2 ;;
        --alpha)             ALPHA="$2"; shift 2 ;;
        --num-cpu)           NUM_CPU="$2"; shift 2 ;;
        --no-swanlab)        USE_SWANLAB=false; shift ;;
        --swanlab-project)   SWANLAB_PROJECT="$2"; shift 2 ;;
        --swanlab-experiment) SWANLAB_EXPERIMENT="$2"; shift 2 ;;
        -h|--help)           show_help; exit 0 ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage"
            exit 1
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

echo "====== Starting Fuz-RL training ======"
echo "Algorithm:   $ALGORITHM"
echo "Env:         $ENVIRONMENT"
echo "Epochs:      $EPOCHS"
echo "GPU:         $GPU"
echo "Seed:        $SEED"
echo "Level:       $LEVEL"
echo "EPS:         $EPS"
echo "Train EPS:   $TRAIN_EPS"
echo "DisturbPart: $DISTURB_PART"
echo "DisturbType: $DISTURB_TYPE"
echo "Alpha:       $ALPHA"
echo "Num CPU:     $NUM_CPU"
echo "SwanLab:     $USE_SWANLAB"
echo "Exp Name:    $SWANLAB_EXPERIMENT"
echo "================================"

CMD="python scripts/train.py \
  --algorithm $ALGORITHM \
  --env $ENVIRONMENT \
  --epochs $EPOCHS \
  --device cuda:0 \
  --seed $SEED \
  --num_cpu $NUM_CPU"

if [[ $ALGORITHM == fuz* ]]; then
  CMD="$CMD \
    --level $LEVEL \
    --eps $EPS \
    --train_eps $TRAIN_EPS \
    --disturb_part $DISTURB_PART \
    --disturb_type $DISTURB_TYPE \
    --fuzzy_lr 0.0001 \
    --fuz_freq 1"
fi

if [[ $ALGORITHM == cvar* ]]; then
  CMD="$CMD --alpha $ALPHA"
fi

if [[ "$USE_SWANLAB" == "true" ]]; then
  CMD="$CMD \
    --use_swanlab \
    --swanlab_project $SWANLAB_PROJECT \
    --swanlab_experiment $SWANLAB_EXPERIMENT"
fi

echo "Command:"
echo "  $CMD"
echo

exec $CMD

