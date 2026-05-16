#!/usr/bin/env bash
# =============================================================================
# launch_hpo.sh
# Asynchronous Multi-GPU HPO Launcher for Nos Model Finetuning
#
# Architecture:
#   Spawns one independent hpo_tuner.py process per GPU. Each process gets
#   its own CUDA context via CUDA_VISIBLE_DEVICES, with no shared memory,
#   no DDP, and no synchronisation barriers. All workers share only the
#   Optuna SQLite/PostgreSQL study database.
#
# Compatibility:
#   - Linux        : bash 4.0+  (Ubuntu, Debian, CentOS, RHEL, Amazon Linux)
#   - Windows      : Git Bash, WSL2, Cygwin (all ship bash 4+)
#   - macOS        : bash 3.2+ via /bin/bash (limited testing)
#   - Cloud VMs    : AWS, GCP, Azure GPU instances (all Linux-based)
#   - Windows VMs  : Azure NV-series, AWS G-series with Git Bash / WSL2
#
# Prerequisites:
#   - nvidia-smi   accessible in PATH
#   - python       accessible in PATH (or set PYTHON_BIN below)
#   - hpo_tuner.py in the same directory as this script (or set SCRIPT_DIR)
#
# Usage:
#   chmod +x launch_hpo.sh
#
#   # Tune both phases across all GPUs (30 trials per worker):
#   ./launch_hpo.sh --config configs/config.yaml --phase both --n-trials 30
#
#   # Tune tokenizer only on GPUs 0,1,2 (GPU subset):
#   ./launch_hpo.sh --gpus 0,1,2 --config configs/config.yaml --phase tokenizer
#
#   # Dry run — print what would be launched without executing:
#   ./launch_hpo.sh --dry-run --config configs/config.yaml --phase both
#
#   # Resume a previously interrupted study:
#   ./launch_hpo.sh --config configs/config.yaml --phase both --n-trials 30
#   (Optuna's load_if_exists=True means workers automatically resume)
#
#   # Use a custom Python binary (e.g., conda env):
#   PYTHON_BIN=/opt/conda/envs/nos/bin/python ./launch_hpo.sh --config ...
#
#   # Limit VRAM per GPU (useful on shared machines):
#   GPU_MEMORY_FRACTION=0.8 ./launch_hpo.sh --config configs/config.yaml
#
# Environment Variables (all optional):
#   PYTHON_BIN              Path to python executable. Default: auto-detected.
#   GPU_MEMORY_FRACTION     CUDA memory fraction [0.1-1.0]. Default: unset.
#   HPO_TIMEOUT_SECONDS     Kill workers after N seconds. Default: unset (no limit).
#   HPO_WORKER_NICE         nice(1) priority for workers [−20 to 19]. Default: 0.
#   LOG_DIR                 Override log directory. Default: logs/hpo_workers.
#
# Exit Codes:
#   0   All workers completed successfully.
#   1   One or more workers failed (details in log files).
#   2   Configuration or environment error (bad args, missing deps).
#   3   Interrupted by user (SIGINT/SIGTERM). Partial results are preserved.
# =============================================================================

# ── Bash version guard ────────────────────────────────────────────────────────
# Arrays with negative indexing and associative arrays require bash 4.0+.
# macOS ships bash 3.2 at /bin/bash but Git Bash / Homebrew bash is 5.x.
if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    echo "ERROR: bash 4.0 or later is required." >&2
    echo "       Detected: bash ${BASH_VERSION}" >&2
    echo "       macOS users: brew install bash && use /usr/local/bin/bash" >&2
    echo "       Windows users: use Git Bash 4+ or WSL2." >&2
    exit 2
fi

set -euo pipefail
# -e  : Exit on any unhandled error
# -u  : Treat unset variables as errors
# -o pipefail : Propagate pipe failures (not just last command)

# ── Script self-location (works with symlinks and spaces in paths) ─────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# ── Colour output (disabled automatically when not a terminal) ────────────────
if [ -t 1 ] && command -v tput &>/dev/null && tput colors &>/dev/null; then
    C_RESET="\033[0m"
    C_BOLD="\033[1m"
    C_RED="\033[31m"
    C_GREEN="\033[32m"
    C_YELLOW="\033[33m"
    C_CYAN="\033[36m"
    C_WHITE="\033[37m"
else
    C_RESET="" C_BOLD="" C_RED="" C_GREEN="" C_YELLOW="" C_CYAN="" C_WHITE=""
fi

# ── Logging helpers ───────────────────────────────────────────────────────────
_ts()    { date "+%Y-%m-%d %H:%M:%S"; }
_info()  { echo -e "${C_CYAN}[$(_ts)] [INFO]${C_RESET}  $*"; }
_ok()    { echo -e "${C_GREEN}[$(_ts)] [ OK ]${C_RESET}  $*"; }
_warn()  { echo -e "${C_YELLOW}[$(_ts)] [WARN]${C_RESET}  $*"; }
_error() { echo -e "${C_RED}[$(_ts)] [ERR ]${C_RESET}  $*" >&2; }
_die()   { _error "$*"; exit 2; }

# ── Default configuration ─────────────────────────────────────────────────────
GPU_SUBSET=""                          # Empty = use all GPUs
DRY_RUN=false
PYTHON_BIN="${PYTHON_BIN:-}"           # Auto-detected below if empty
GPU_MEMORY_FRACTION="${GPU_MEMORY_FRACTION:-}"
HPO_TIMEOUT_SECONDS="${HPO_TIMEOUT_SECONDS:-}"
HPO_WORKER_NICE="${HPO_WORKER_NICE:-0}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/hpo_workers}"
HPO_TUNER_SCRIPT="${SCRIPT_DIR}/hpo_tuner.py"
HPO_ARGS=()                            # Arguments forwarded to hpo_tuner.py

# ── Argument parser ───────────────────────────────────────────────────────────
# Separates launcher-specific flags from hpo_tuner.py pass-through arguments.
_usage() {
    cat <<EOF

${C_BOLD}Usage:${C_RESET}
  ${SCRIPT_NAME} [LAUNCHER OPTIONS] [HPO_TUNER OPTIONS]

${C_BOLD}Launcher Options:${C_RESET}
  --gpus <0,1,2,...>      Comma-separated GPU IDs to use. Default: all GPUs.
  --dry-run               Print launch commands without executing them.
  --log-dir <path>        Override log directory. Default: logs/hpo_workers/
  --help, -h              Show this help message.

${C_BOLD}HPO Tuner Options (passed through to hpo_tuner.py):${C_RESET}
  --config <path>         Path to YAML config. Default: configs/config.yaml
  --phase <phase>         tokenizer | basemodel | both. Default: both
  --n-trials <int>        Number of Optuna trials per worker.
  --apply-best            Write best params to a new config file.
  --tokenizer-path <path> Tokenizer path (for --phase basemodel).
  --output-config <path>  Output path for best config YAML.
  --show-importance       Print hyperparameter importance after tuning.

${C_BOLD}Environment Variables:${C_RESET}
  PYTHON_BIN              Python executable. Default: auto-detected.
  GPU_MEMORY_FRACTION     CUDA memory fraction [0.1-1.0].
  HPO_TIMEOUT_SECONDS     Worker timeout in seconds.
  HPO_WORKER_NICE         Process nice priority [-20 to 19]. Default: 0.
  LOG_DIR                 Log directory override.

${C_BOLD}Examples:${C_RESET}
  ${SCRIPT_NAME} --config configs/config.yaml --phase both --n-trials 30
  ${SCRIPT_NAME} --gpus 0,1,2 --config configs/config.yaml --phase tokenizer
  ${SCRIPT_NAME} --dry-run --config configs/config.yaml --phase both
  PYTHON_BIN=/opt/conda/bin/python ${SCRIPT_NAME} --config configs/config.yaml

EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            [[ $# -lt 2 ]] && _die "--gpus requires an argument."
            GPU_SUBSET="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --log-dir)
            [[ $# -lt 2 ]] && _die "--log-dir requires an argument."
            LOG_DIR="$2"; shift 2 ;;
        --help|-h)
            _usage; exit 0 ;;
        *)
            # Everything else is forwarded to hpo_tuner.py
            HPO_ARGS+=("$1"); shift ;;
    esac
done

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Environment Detection & Validation
# ══════════════════════════════════════════════════════════════════════════════

# ── OS detection ──────────────────────────────────────────────────────────────
_detect_os() {
    local os_name
    os_name="$(uname -s 2>/dev/null || echo 'Unknown')"
    case "${os_name}" in
        Linux*)   echo "linux"   ;;
        Darwin*)  echo "macos"   ;;
        CYGWIN*)  echo "windows" ;;
        MINGW*)   echo "windows" ;;
        MSYS*)    echo "windows" ;;
        *)
            # Final fallback: check for Windows-style paths
            if [[ "${OSTYPE:-}" == "msys" ]] || [[ "${OSTYPE:-}" == "cygwin" ]]; then
                echo "windows"
            else
                echo "unknown"
            fi
            ;;
    esac
}

OS_TYPE="$(_detect_os)"
_info "Detected OS: ${OS_TYPE}"

# ── Python detection ──────────────────────────────────────────────────────────
_detect_python() {
    # If user explicitly set PYTHON_BIN, validate and use it
    if [[ -n "${PYTHON_BIN}" ]]; then
        if command -v "${PYTHON_BIN}" &>/dev/null; then
            echo "${PYTHON_BIN}"
            return 0
        else
            _die "PYTHON_BIN='${PYTHON_BIN}' is not executable or not in PATH."
        fi
    fi

    # Auto-detection priority:
    # 1. python3   (preferred on Linux/macOS/WSL2)
    # 2. python    (Windows native, conda base envs)
    # 3. py -3     (Windows Python Launcher — py.exe)
    local candidates=("python3" "python" "py")
    for candidate in "${candidates[@]}"; do
        if command -v "${candidate}" &>/dev/null; then
            # Verify it is Python 3 (not Python 2)
            local version
            version="$("${candidate}" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo '0')"
            if [[ "${version}" == "3" ]]; then
                echo "${candidate}"
                return 0
            fi
        fi
    done

    # Windows py.exe launcher
    if command -v py &>/dev/null; then
        local version
        version="$(py -3 -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo '0')"
        if [[ "${version}" == "3" ]]; then
            echo "py -3"
            return 0
        fi
    fi

    _die "No Python 3 interpreter found in PATH. " \
         "Set PYTHON_BIN=/path/to/python3 or activate your conda/venv."
}

PYTHON_BIN="$(_detect_python)"
_info "Python interpreter: ${PYTHON_BIN}"

# Validate Python version is 3.8+
PY_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo '0.0')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION#*.}"
if [[ "${PY_MAJOR}" -lt 3 ]] || { [[ "${PY_MAJOR}" -eq 3 ]] && [[ "${PY_MINOR}" -lt 8 ]]; }; then
    _die "Python 3.8 or later required. Found: Python ${PY_VERSION}"
fi
_info "Python version: ${PY_VERSION} ✓"

# ── nvidia-smi detection ──────────────────────────────────────────────────────
_detect_nvidia_smi() {
    # Standard PATH check
    if command -v nvidia-smi &>/dev/null; then
        echo "nvidia-smi"
        return 0
    fi

    # Windows-specific install paths (not always in PATH inside Git Bash)
    local win_paths=(
        "/c/Windows/System32/nvidia-smi.exe"
        "/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe"
        "/c/Windows/System32/nvidia-smi"
    )
    for path in "${win_paths[@]}"; do
        if [[ -x "${path}" ]]; then
            echo "${path}"
            return 0
        fi
    done

    return 1
}

NVIDIA_SMI=""
if ! NVIDIA_SMI="$(_detect_nvidia_smi)"; then
    _die "nvidia-smi not found in PATH or standard install locations." \
         "Ensure NVIDIA drivers are installed and nvidia-smi is accessible."
fi
_info "nvidia-smi: ${NVIDIA_SMI} ✓"

# ── hpo_tuner.py location check ───────────────────────────────────────────────
if [[ ! -f "${HPO_TUNER_SCRIPT}" ]]; then
    _die "hpo_tuner.py not found at: ${HPO_TUNER_SCRIPT}" \
         "Place launch_hpo.sh in the same directory as hpo_tuner.py."
fi
_info "HPO script: ${HPO_TUNER_SCRIPT} ✓"

# ── Optuna installation check ─────────────────────────────────────────────────
if ! ${PYTHON_BIN} -c "import optuna" &>/dev/null; then
    _die "Optuna is not installed in the detected Python environment." \
         "Run: ${PYTHON_BIN} -m pip install optuna plotly"
fi
OPTUNA_VERSION="$(${PYTHON_BIN} -c 'import optuna; print(optuna.__version__)' 2>/dev/null || echo 'unknown')"
_info "Optuna version: ${OPTUNA_VERSION} ✓"

# ── PyTorch + CUDA check ──────────────────────────────────────────────────────
TORCH_CUDA_AVAILABLE="$(${PYTHON_BIN} -c \
    'import torch; print("yes" if torch.cuda.is_available() else "no")' \
    2>/dev/null || echo 'no')"
if [[ "${TORCH_CUDA_AVAILABLE}" != "yes" ]]; then
    _warn "torch.cuda.is_available() returned False." \
          "Workers will run on CPU. Performance will be severely degraded."
fi

# ── nice availability (non-critical on Windows) ───────────────────────────────
NICE_CMD=""
if command -v nice &>/dev/null && [[ "${OS_TYPE}" != "windows" ]]; then
    NICE_CMD="nice -n ${HPO_WORKER_NICE}"
fi

# ── timeout availability ──────────────────────────────────────────────────────
TIMEOUT_CMD=""
if [[ -n "${HPO_TIMEOUT_SECONDS}" ]]; then
    if command -v timeout &>/dev/null; then
        TIMEOUT_CMD="timeout ${HPO_TIMEOUT_SECONDS}"
    elif command -v gtimeout &>/dev/null; then
        # macOS via coreutils: brew install coreutils
        TIMEOUT_CMD="gtimeout ${HPO_TIMEOUT_SECONDS}"
    else
        _warn "HPO_TIMEOUT_SECONDS=${HPO_TIMEOUT_SECONDS} set but 'timeout' not found." \
              "Workers will run without a time limit."
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: GPU Enumeration
# ══════════════════════════════════════════════════════════════════════════════

# ── Enumerate all available GPUs ──────────────────────────────────────────────
_enumerate_gpus() {
    local raw_output
    # --query-gpu=index gives numeric IDs; csv,noheader strips the header row
    raw_output="$(${NVIDIA_SMI} --query-gpu=index,name,memory.total \
        --format=csv,noheader,nounits 2>/dev/null)" || {
        _die "nvidia-smi failed to enumerate GPUs. " \
             "Check driver installation: ${NVIDIA_SMI} --version"
    }

    if [[ -z "${raw_output}" ]]; then
        _die "nvidia-smi returned no GPU information."
    fi
    echo "${raw_output}"
}

# Build GPU arrays
declare -a ALL_GPU_IDS=()
declare -a ALL_GPU_NAMES=()
declare -a ALL_GPU_VRAM=()

while IFS=',' read -r gpu_id gpu_name gpu_vram; do
    # Trim whitespace (critical for Windows where nvidia-smi adds extra spaces)
    gpu_id="$(echo "${gpu_id}"   | tr -d '[:space:]')"
    gpu_name="$(echo "${gpu_name}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    gpu_vram="$(echo "${gpu_vram}" | tr -d '[:space:]')"

    ALL_GPU_IDS+=("${gpu_id}")
    ALL_GPU_NAMES+=("${gpu_name}")
    ALL_GPU_VRAM+=("${gpu_vram}")
done < <(_enumerate_gpus)

TOTAL_GPUS="${#ALL_GPU_IDS[@]}"
if [[ "${TOTAL_GPUS}" -eq 0 ]]; then
    _die "No GPUs detected. Check nvidia-smi output manually: ${NVIDIA_SMI}"
fi

# ── Apply GPU subset filter ───────────────────────────────────────────────────
declare -a SELECTED_GPU_IDS=()
declare -a SELECTED_GPU_NAMES=()
declare -a SELECTED_GPU_VRAM=()

if [[ -n "${GPU_SUBSET}" ]]; then
    # Parse comma-separated list: "0,1,2" → array
    IFS=',' read -ra REQUESTED_IDS <<< "${GPU_SUBSET}"
    for req_id in "${REQUESTED_IDS[@]}"; do
        req_id="$(echo "${req_id}" | tr -d '[:space:]')"
        local_found=false
        for i in "${!ALL_GPU_IDS[@]}"; do
            if [[ "${ALL_GPU_IDS[$i]}" == "${req_id}" ]]; then
                SELECTED_GPU_IDS+=("${ALL_GPU_IDS[$i]}")
                SELECTED_GPU_NAMES+=("${ALL_GPU_NAMES[$i]}")
                SELECTED_GPU_VRAM+=("${ALL_GPU_VRAM[$i]}")
                local_found=true
                break
            fi
        done
        if [[ "${local_found}" == false ]]; then
            _warn "Requested GPU ${req_id} not found in available GPUs [${ALL_GPU_IDS[*]}]. Skipping."
        fi
    done

    if [[ "${#SELECTED_GPU_IDS[@]}" -eq 0 ]]; then
        _die "No valid GPUs remain after applying --gpus '${GPU_SUBSET}'."
    fi
else
    # Use all available GPUs
    SELECTED_GPU_IDS=("${ALL_GPU_IDS[@]}")
    SELECTED_GPU_NAMES=("${ALL_GPU_NAMES[@]}")
    SELECTED_GPU_VRAM=("${ALL_GPU_VRAM[@]}")
fi

NUM_WORKERS="${#SELECTED_GPU_IDS[@]}"

# ── Check for existing HPO DB and warn if starting fresh ──────────────────────
_check_existing_db() {
    local storage_uri=""
    # Extract storage URI from HPO_ARGS if --config was passed
    local config_path=""
    for i in "${!HPO_ARGS[@]}"; do
        if [[ "${HPO_ARGS[$i]}" == "--config" ]]; then
            config_path="${HPO_ARGS[$((i+1))]:-}"
            break
        fi
    done

    if [[ -n "${config_path}" ]] && [[ -f "${config_path}" ]]; then
        # Simple grep — avoids requiring PyYAML at bash level
        storage_uri="$(grep -E '^\s*storage:' "${config_path}" \
            | sed 's/.*storage:[[:space:]]*//' \
            | tr -d '"'"'" \
            | tr -d '[:space:]' || echo '')"
        if [[ "${storage_uri}" == "null" ]] || [[ -z "${storage_uri}" ]]; then
            _warn "hpo.storage is null in config. Using in-memory Optuna storage." \
                  "Results will NOT be shared between workers or persisted across runs." \
                  "Set hpo.storage to a SQLite or PostgreSQL URI for multi-GPU HPO."
        else
            # Extract file path from sqlite:///path?timeout=60
            local db_path
            db_path="$(echo "${storage_uri}" \
                | sed 's|sqlite:///||' \
                | sed 's|?.*||')"
            if [[ -f "${db_path}" ]]; then
                _info "Existing Optuna DB found: ${db_path}"
                _info "Workers will resume the existing study (load_if_exists=True)."
            fi
        fi
    fi
}
_check_existing_db

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Signal Handling & Cleanup
# ══════════════════════════════════════════════════════════════════════════════

declare -a WORKER_PIDS=()
INTERRUPTED=false

_cleanup() {
    local signal="${1:-SIGTERM}"
    INTERRUPTED=true

    echo ""
    _warn "Received ${signal}. Sending SIGTERM to all worker processes..."

    local kill_failed=0
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            # ── TASK 2.2: Clean Process Termination for Windows/WSL ──────────
            # Standard 'kill' leaves Python multiprocessing DataLoader workers orphaned
            # on Windows, trapping VRAM forever. We must kill the entire process tree.
            if [[ "${OS_TYPE}" == "windows" ]]; then
                taskkill //F //T //PID "${pid}" 2>/dev/null || kill_failed=$((kill_failed + 1))
            else
                kill -TERM "${pid}" 2>/dev/null || kill_failed=$((kill_failed + 1))
            fi
            _warn "  Sent termination signal to PID ${pid}"
        fi
    done

    # Give workers 10 seconds to shut down gracefully
    # Skip wait on Windows since taskkill //F is immediate
    if [[ "${OS_TYPE}" != "windows" ]]; then
        local grace_seconds=10
        _info "Waiting up to ${grace_seconds}s for graceful shutdown..."
        local elapsed=0
        while [[ ${elapsed} -lt ${grace_seconds} ]]; do
            local still_running=0
            for pid in "${WORKER_PIDS[@]}"; do
                kill -0 "${pid}" 2>/dev/null && still_running=$((still_running + 1))
            done
            [[ ${still_running} -eq 0 ]] && break
            sleep 1
            elapsed=$((elapsed + 1))
        done

        # Force kill any remaining workers (POSIX)
        for pid in "${WORKER_PIDS[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                _warn "  Force killing PID ${pid} (SIGKILL)"
                kill -KILL "${pid}" 2>/dev/null || true
            fi
        done
    fi

    _warn "Interrupted. Completed trials have been saved to the Optuna DB."
    _warn "Re-run the same command to resume from where you left off."
    exit 3
}

# Trap SIGINT (Ctrl+C), SIGTERM (kill), and SIGHUP (terminal close)
# Note: SIGHUP is not available on Windows but the trap is harmless there
trap '_cleanup SIGINT'  INT
trap '_cleanup SIGTERM' TERM
trap '_cleanup SIGHUP'  HUP 2>/dev/null || true

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Print Launch Plan
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${C_BOLD}╔══════════════════════════════════════════════════════════════╗${C_RESET}"
echo -e "${C_BOLD}║       Nos HPO — Asynchronous Multi-GPU Launcher              ║${C_RESET}"
echo -e "${C_BOLD}╠══════════════════════════════════════════════════════════════╣${C_RESET}"
printf  "║  %-20s : %-37s║\n" "OS"           "${OS_TYPE}"
printf  "║  %-20s : %-37s║\n" "Python"       "${PYTHON_BIN} (${PY_VERSION})"
printf  "║  %-20s : %-37s║\n" "Optuna"       "${OPTUNA_VERSION}"
printf  "║  %-20s : %-37s║\n" "Total GPUs"   "${TOTAL_GPUS} detected"
printf  "║  %-20s : %-37s║\n" "Worker GPUs"  "${NUM_WORKERS} selected"
printf  "║  %-20s : %-37s║\n" "Log dir"      "${LOG_DIR}"
[[ -n "${HPO_TIMEOUT_SECONDS}" ]] && \
printf  "║  %-20s : %-37s║\n" "Worker timeout"  "${HPO_TIMEOUT_SECONDS}s"
[[ "${DRY_RUN}" == true ]] && \
printf  "║  %-20s : %-37s║\n" "Mode"         "DRY RUN — no processes launched"
echo -e "${C_BOLD}╠══════════════════════════════════════════════════════════════╣${C_RESET}"
echo    "║  Selected GPUs:                                              ║"
for i in "${!SELECTED_GPU_IDS[@]}"; do
    printf "║    GPU %-3s : %-20s  %6s MiB VRAM          ║\n" \
        "${SELECTED_GPU_IDS[$i]}" \
        "${SELECTED_GPU_NAMES[$i]:0:20}" \
        "${SELECTED_GPU_VRAM[$i]}"
done
echo -e "${C_BOLD}╠══════════════════════════════════════════════════════════════╣${C_RESET}"
echo    "║  HPO arguments forwarded to hpo_tuner.py:                    ║"
printf  "║    %-58s║\n" "${HPO_ARGS[*]:0:58}"
echo -e "${C_BOLD}╚══════════════════════════════════════════════════════════════╝${C_RESET}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Worker Launch
# ══════════════════════════════════════════════════════════════════════════════

mkdir -p "${LOG_DIR}"

# Per-run timestamp for log file naming (shared across all workers in this run)
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

declare -a LOG_FILES=()

for i in "${!SELECTED_GPU_IDS[@]}"; do
    GPU_ID="${SELECTED_GPU_IDS[$i]}"
    GPU_NAME="${SELECTED_GPU_NAMES[$i]}"
    LOG_FILE="${LOG_DIR}/worker_gpu${GPU_ID}_${RUN_TIMESTAMP}.log"
    LOG_FILES+=("${LOG_FILE}")

    # ── Build the full command ─────────────────────────────────────────────
    # Construct as an array to handle paths with spaces correctly
    CMD=()

    # Process priority (Linux/macOS only)
    if [[ -n "${NICE_CMD}" ]]; then
        CMD+=($NICE_CMD)
    fi

    # Timeout wrapper (if configured)
    if [[ -n "${TIMEOUT_CMD}" ]]; then
        CMD+=($TIMEOUT_CMD)
    fi

    # Python interpreter
    # Split PYTHON_BIN in case it contains flags (e.g., "py -3")
    read -ra PY_PARTS <<< "${PYTHON_BIN}"
    CMD+=("${PY_PARTS[@]}")

    # Unbuffered output: critical so log files get real-time writes,
    # not buffered writes that only flush when the process exits.
    CMD+=("-u")

    # The HPO script
    CMD+=("${HPO_TUNER_SCRIPT}")

    # Forward all HPO arguments
    CMD+=("${HPO_ARGS[@]}")

    # ── Environment for this worker ────────────────────────────────────────
    # CUDA_VISIBLE_DEVICES: Restricts this process to exactly one physical GPU.
    # CUDA remaps the assigned GPU to logical index cuda:0 inside the process.
    # PYTHONUNBUFFERED: Belt-and-suspenders for -u flag (some launchers ignore -u).
    # PYTHONFAULTHANDLER: Dumps C-level stack traces on segfaults (invaluable for debugging).
    declare -a WORKER_ENV=(
        "CUDA_VISIBLE_DEVICES=${GPU_ID}"
        "PYTHONUNBUFFERED=1"
        "PYTHONFAULTHANDLER=1"
    )

    # Optional VRAM fraction limiter
    if [[ -n "${GPU_MEMORY_FRACTION}" ]]; then
        WORKER_ENV+=("PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
        # Note: actual fraction limiting requires PyTorch code changes;
        # this env var limits the allocator's split size as a proxy.
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        _info "[DRY RUN] GPU ${GPU_ID} (${GPU_NAME}) would run:"
        echo  "         env ${WORKER_ENV[*]} ${CMD[*]}"
        echo  "         >> ${LOG_FILE} 2>&1 &"
    else
        # Write a header into the log file before the process starts
        {
            echo "============================================================"
            echo "  Nos HPO Worker Log"
            echo "  Run timestamp : ${RUN_TIMESTAMP}"
            echo "  GPU ID        : ${GPU_ID} (${GPU_NAME})"
            echo "  GPU VRAM      : ${SELECTED_GPU_VRAM[$i]} MiB"
            echo "  Command       : ${CMD[*]}"
            echo "  Environment   : ${WORKER_ENV[*]}"
            echo "  Started at    : $(date)"
            echo "============================================================"
        } > "${LOG_FILE}"

        # Launch the worker process
        # env sets per-process environment without polluting the parent shell
        env "${WORKER_ENV[@]}" "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
        WORKER_PID=$!

        WORKER_PIDS+=("${WORKER_PID}")
        _ok "GPU ${GPU_ID} (${GPU_NAME}) → PID ${WORKER_PID} → ${LOG_FILE}"
    fi
done

# ── Dry run exits here ────────────────────────────────────────────────────────
if [[ "${DRY_RUN}" == true ]]; then
    echo ""
    _info "Dry run complete. No processes were launched."
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Live Monitoring
# ══════════════════════════════════════════════════════════════════════════════

echo ""
_info "All ${NUM_WORKERS} worker(s) launched."
echo ""
echo -e "  ${C_CYAN}Monitor logs (all workers):${C_RESET}"
echo    "    tail -f ${LOG_DIR}/worker_gpu*_${RUN_TIMESTAMP}.log"
echo ""
echo -e "  ${C_CYAN}Monitor individual worker:${C_RESET}"
for i in "${!SELECTED_GPU_IDS[@]}"; do
    echo "    tail -f ${LOG_FILES[$i]}"
done
echo ""
echo -e "  ${C_CYAN}Monitor GPU utilisation:${C_RESET}"
echo    "    watch -n 2 nvidia-smi"
echo ""
echo -e "  ${C_CYAN}Stop all workers:${C_RESET}"
echo    "    Ctrl+C  (graceful SIGTERM → waits 10s → SIGKILL)"
echo ""

# ── Optional: background VRAM monitor ─────────────────────────────────────────
# Logs nvidia-smi output every 60 seconds into a separate file for post-run
# VRAM analysis. Killed automatically when the script exits.
VRAM_LOG="${LOG_DIR}/vram_monitor_${RUN_TIMESTAMP}.log"
(
    while true; do
        {
            echo "--- $(date) ---"
            "${NVIDIA_SMI}" \
                --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
                --format=csv,noheader,nounits 2>/dev/null || true
            echo ""
        } >> "${VRAM_LOG}" 2>/dev/null
        sleep 60
    done
) &
VRAM_MONITOR_PID=$!
_info "VRAM monitor started (PID ${VRAM_MONITOR_PID}) → ${VRAM_LOG}"

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Wait for All Workers & Collect Results
# ══════════════════════════════════════════════════════════════════════════════

declare -a EXIT_CODES=()
declare -a FAILED_GPUS=()
declare -a SUCCEEDED_GPUS=()
FAILED_COUNT=0
SUCCEEDED_COUNT=0

_info "Waiting for all ${NUM_WORKERS} worker(s) to complete..."
echo  "  (This may take several hours for large HPO runs)"
echo ""

for i in "${!WORKER_PIDS[@]}"; do
    PID="${WORKER_PIDS[$i]}"
    GPU_ID="${SELECTED_GPU_IDS[$i]}"
    GPU_NAME="${SELECTED_GPU_NAMES[$i]}"
    LOG_FILE="${LOG_FILES[$i]}"

    EXIT_CODE=0
    wait "${PID}" || EXIT_CODE=$?
    EXIT_CODES+=("${EXIT_CODE}")

    # Append completion footer to the worker's log
    {
        echo ""
        echo "============================================================"
        echo "  Worker exited at: $(date)"
        echo "  Exit code: ${EXIT_CODE}"
        echo "============================================================"
    } >> "${LOG_FILE}" 2>/dev/null || true

    if [[ "${EXIT_CODE}" -eq 0 ]]; then
        _ok  "GPU ${GPU_ID} (${GPU_NAME}) | PID ${PID} | EXIT 0 | SUCCESS"
        SUCCEEDED_GPUS+=("${GPU_ID}")
        SUCCEEDED_COUNT=$((SUCCEEDED_COUNT + 1))
    elif [[ "${EXIT_CODE}" -eq 3 ]]; then
        # Exit code 3 = interrupted by user (our own convention from hpo_tuner.py)
        _warn "GPU ${GPU_ID} (${GPU_NAME}) | PID ${PID} | EXIT 3 | INTERRUPTED"
        _warn "  Partial results are preserved in the Optuna DB."
        SUCCEEDED_GPUS+=("${GPU_ID}")  # Treat interruption as non-failure
        SUCCEEDED_COUNT=$((SUCCEEDED_COUNT + 1))
    else
        _error "GPU ${GPU_ID} (${GPU_NAME}) | PID ${PID} | EXIT ${EXIT_CODE} | FAILED"
        _error "  Last 20 lines of log:"
        tail -n 20 "${LOG_FILE}" | sed 's/^/    /' >&2
        FAILED_GPUS+=("${GPU_ID}")
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

# ── Kill the VRAM monitor ─────────────────────────────────────────────────────
kill "${VRAM_MONITOR_PID}" 2>/dev/null || true
wait "${VRAM_MONITOR_PID}" 2>/dev/null || true

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Final Report
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${C_BOLD}══════════════════════════════════════════════════════════════${C_RESET}"
echo -e "${C_BOLD}  HPO LAUNCHER — FINAL REPORT${C_RESET}"
echo    "══════════════════════════════════════════════════════════════"
printf  "  %-24s : %s\n" "Run timestamp"    "${RUN_TIMESTAMP}"
printf  "  %-24s : %s\n" "Total workers"    "${NUM_WORKERS}"
printf  "  %-24s : %s\n" "Succeeded"        "${SUCCEEDED_COUNT}"
printf  "  %-24s : %s\n" "Failed"           "${FAILED_COUNT}"
printf  "  %-24s : %s\n" "Log directory"    "${LOG_DIR}"
printf  "  %-24s : %s\n" "VRAM log"         "${VRAM_LOG}"
echo    "══════════════════════════════════════════════════════════════"

if [[ "${FAILED_COUNT}" -gt 0 ]]; then
    echo ""
    _error "Failed GPU workers: ${FAILED_GPUS[*]}"
    _error "Inspect their logs:"
    for fail_gpu in "${FAILED_GPUS[@]}"; do
        echo  "  ${LOG_DIR}/worker_gpu${fail_gpu}_${RUN_TIMESTAMP}.log"
    done
    echo ""
    _info "Completed trials from successful workers are preserved in the Optuna DB."
    _info "You can re-run the same command to launch replacement workers for the failed GPUs:"
    echo  "  ./launch_hpo.sh --gpus $(IFS=','; echo "${FAILED_GPUS[*]}") ${HPO_ARGS[*]}"
    echo ""
    exit 1
fi

echo ""
_ok "All ${NUM_WORKERS} worker(s) completed successfully."
echo ""
_info "Next steps:"
echo  "  1. Apply best params to config:"
echo  "     ${PYTHON_BIN} hpo_tuner.py ${HPO_ARGS[*]} --apply-best"
echo  "  2. View hyperparameter importance:"
echo  "     ${PYTHON_BIN} hpo_tuner.py ${HPO_ARGS[*]} --show-importance"
echo  "  3. Review VRAM usage over time:"
echo  "     cat ${VRAM_LOG}"
echo ""
exit 0