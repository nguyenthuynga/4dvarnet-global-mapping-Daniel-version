#!/bin/bash
# =============================================================================
# Inference + data-challenge evaluation of the joint OSSE+OSE UNet2D
# (50/50 mix of GLORYS 2010-2017 and real data, config unet-joint-osse-ose)
# on the 5 test constellations of the table.
#
# For each cell: inferring.py (same arguments dvc.yaml builds, with the GLORYS
# normalisation used in training), then a copy of the evaluation notebook
# template with method_name/method_path patched, executed with nbconvert.
# params.yaml is never touched.
#
# RUN IT (tmux -- survives disconnection), on a GPU that is free:
#     tmux new -s joint
#     bash run_joint_cells.sh 2>&1 | tee run_joint_cells.log
#     # detach: Ctrl-b then d      reattach: tmux attach -t joint
# On another GPU of the node:
#     CUDA_VISIBLE_DEVICES=1 bash run_joint_cells.sh 2>&1 | tee run_joint_cells.log
# Only some cells: pass part of the name, e.g.   bash run_joint_cells.sh Test7nadirs
#
# WHERE IS IT? (from any terminal)
#     cat /Odyssey/private/n23nguye/4dvarnet-global-mapping-Daniel-version/run_joint_cells.status
#     -> one line per step, the scores at the end of each cell, and the command
#        that shows the progress bar of the running inference
#
# RESUMING: launch it again; finished cells are skipped and an existing map is
# reused. One UNet2D cell takes about 20 min (10 inference + 10 notebook).
# =============================================================================

set -o pipefail          # NOT -u: conda's MKL activate script uses unset vars
                         # NOT -e: one failed cell must not kill the batch

ONLY="${1:-}"            # optional filter on the job name

# ---------------------------------------------------------------- paths -----
REPO=/Odyssey/private/n23nguye/4dvarnet-global-mapping-Daniel-version
INFER=$REPO/inference
OUTDIR=$INFER/outputs
NB_DIR=/Odyssey/private/n23nguye/2023a_SSH_mapping_OSE/Notebooks
TEMPLATE=evaluation_2024_with_without_lwe_MaskMiost_TrainOSSEFix4Nadirs_Test4Nadirs_Unet.ipynb

TRAIN_ENV=4dvarnet-gm    # inference
EVAL_ENV=dc              # notebooks

# ------------------------------------------------------- inference knobs ----
PATCH_2D=15-720-1440
STRIDE=1-208-232
PERIOD_FROM=2023-12-01
PERIOD_TO=2025-01-31
ENS_SIZE=10              # same ensemble as the other UNet2D cells of the table
ENS_NOISE=0.2

# GLORYS stats used to normalise BOTH halves of the training data
# (base.yaml, datamodule.norm_stats.train) -> reused at inference
GLORYS_MEAN=-0.001930378327255099
GLORYS_STD=0.08604745118775728

MIN_FREE_GB=30           # don't start if the GPU has less free memory than this

STATUS_FILE=$REPO/run_joint_cells.status
LOGDIR=$REPO/logs_joint_cells

# ------------------------------------------------------------ checkpoint ----
RUN_JOINT="$REPO/outputs/2026-10-20/joint_osse_ose_unet2d_gappy-10-57-13"
CKPT_JOINT="$RUN_JOINT/unet-joint-osse-ose/checkpoints/val_mse_0p78957-epoch_593.ckpt"
CFG_JOINT="$RUN_JOINT/.hydra/config.yaml"

# ------------------------------------------------------- test constellations -
DATA=/SCRATCH/n23nguye/mapping_2023_2025
# IN_SWSW="$DATA/alongtrack_SWOTnSwon_real_2023-01-01_2025-07-31.nc"
# IN_2N="$DATA/alongtrack_only2nadirss3aNs6a_real_2023-01-01_2025-07-31.nc"
# IN_4N="$DATA/alongtrack_only4nadirsj3ns3as3bs6a_real_2023-01-01_2025-07-31.nc"
# IN_6N="$DATA/alongtrack_only6nadirs_real_2023-01-01_2025-07-31.nc"
IN_7NS="$DATA/alongtrack_S7nadirsNswotKarinWithLweRemoved_real_2023-01-01_2025-07-31.nc"

# ---------------------- the jobs: name|ckpt|cfg|input|patch|norm|ensemble ---
#   norm: glorys = +mean/+std above (as in training); self = stats of the input file
#   ensemble: empty = ENS_SIZE; 1 = a single inference without noise
JOBS=(
  "Unet2D_JointOSSEOSE_Test7nadirsNSWOT|$CKPT_JOINT|$CFG_JOINT|$IN_7NS|$PATCH_2D|glorys"
  # "Unet2D_JointOSSEOSE_TestSWOTnSwon|$CKPT_JOINT|$CFG_JOINT|$IN_SWSW|$PATCH_2D|glorys"
  # "Unet2D_JointOSSEOSE_Test2nadirs|$CKPT_JOINT|$CFG_JOINT|$IN_2N|$PATCH_2D|glorys"
  # "Unet2D_JointOSSEOSE_Test4nadirs|$CKPT_JOINT|$CFG_JOINT|$IN_4N|$PATCH_2D|glorys"
  # "Unet2D_JointOSSEOSE_Test6nadirs|$CKPT_JOINT|$CFG_JOINT|$IN_6N|$PATCH_2D|glorys"
)

# keep only the jobs matching the filter (if any)
SEL=()
for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME _ <<< "$job"
  [[ -z "$ONLY" || "$NAME" == *"$ONLY"* ]] && SEL+=("$job")
done
if [[ ${#SEL[@]} -eq 0 ]]; then
  echo "no job name contains '$ONLY' - nothing to do."; exit 1
fi

# =============================================================================
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p "$LOGDIR"

# one line per step, in STATUS_FILE and on screen
status() { echo "$(date '+%d/%m %H:%M')  $*" | tee -a "$STATUS_FILE"; }
hm()     { printf '%dh%02dm' $(($1 / 3600)) $(($1 % 3600 / 60)); }

echo "=========================================================="
echo " joint OSSE+OSE model: ${#SEL[@]} cells ${ONLY:+(filter: $ONLY) }  $(date)"
echo "=========================================================="

# ---- pre-flight: every file must exist before starting -----------------------
missing=0
for job in "${SEL[@]}"; do
  IFS='|' read -r NAME CKPT CFG INPUT PATCH NORM ENS <<< "$job"
  ENS=${ENS:-$ENS_SIZE}
  for f in "$CKPT" "$CFG" "$INPUT"; do
    [[ -f "$f" ]] || { echo "MISSING FILE ($NAME): $f"; missing=1; }
  done
  [[ "$PATCH" =~ ^[0-9]+-[0-9]+-[0-9]+$ ]] || { echo "BAD PATCH ($NAME): '$PATCH'"; missing=1; }
  [[ "$ENS" =~ ^[0-9]+$ && "$ENS" -ge 1 ]] || { echo "BAD ENSEMBLE ($NAME): '$ENS'"; missing=1; }
  [[ "$NORM" == "self" || "$NORM" == "glorys" ]] || { echo "BAD NORM ($NAME): '$NORM'"; missing=1; }
done
[[ -f "$NB_DIR/$TEMPLATE" ]] || { echo "MISSING TEMPLATE: $NB_DIR/$TEMPLATE"; missing=1; }
if [[ $missing -eq 1 ]]; then
  echo
  echo "Fix the problems listed above before running. Aborting."
  exit 1
fi
echo "checkpoint, config, inputs and template found."

# ---- GPU check: another inference (e.g. the UNet3D/4DVarNet batch) may be using it
if command -v nvidia-smi > /dev/null; then
  GPU_ID=${CUDA_VISIBLE_DEVICES:-0}; GPU_ID=${GPU_ID%%,*}
  read -r used total <<< "$(nvidia-smi --query-gpu=memory.used,memory.total \
                             --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ',')"
  free_gb=$(( (total - used) / 1024 ))
  echo "GPU $GPU_ID: $free_gb GB free"
  if (( free_gb < MIN_FREE_GB )); then
    echo "Only $free_gb GB free on GPU $GPU_ID; it is used by:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$GPU_ID"
    echo "Wait until it is free, or use another GPU:"
    echo "    CUDA_VISIBLE_DEVICES=1 bash run_joint_cells.sh 2>&1 | tee run_joint_cells.log"
    exit 1
  fi
fi
echo
status "=== batch started: ${#SEL[@]} cells${ONLY:+ (filter: $ONLY)}, logs in $LOGDIR"

ok=0; failed=0; reused=0; already=0; i=0; N=${#SEL[@]}

# =============================================================================
for job in "${SEL[@]}"; do
  i=$((i+1))
  IFS='|' read -r NAME CKPT CFG INPUT PATCH NORM ENS <<< "$job"
  ENS=${ENS:-$ENS_SIZE}
  if [[ "$ENS" -gt 1 ]]; then
    OUT_NC="$OUTDIR/${NAME}-n${ENS}-std${ENS_NOISE}.nc"
    ENS_ARGS=("+ensemble_noise=$ENS_NOISE" "+ensemble_size=$ENS")
  else
    OUT_NC="$OUTDIR/${NAME}.nc"            # one inference, no noise, no suffix
    ENS_ARGS=()
  fi
  NB="evaluation_2024_${NAME}.ipynb"
  DONE="$NB_DIR/.done_${NAME}"
  ILOG="$LOGDIR/${NAME}.infer.log"
  NLOG="$LOGDIR/${NAME}.notebook.log"
  TAG="[$i/$N] $NAME:"

  echo "=========================================================="
  echo ">>> $NAME   (cell $i of $N)"
  echo "    input : $(basename "$INPUT")"
  echo "    ckpt  : $(basename "$CKPT")"
  echo "    patch : $PATCH   norm: $NORM   ensemble: $ENS"
  echo "    start : $(date +%H:%M:%S)"
  echo "=========================================================="

  # ---------------- 0. already evaluated? ----------------
  if [[ -f "$OUT_NC" && -f "$DONE" && "$DONE" -nt "$OUT_NC" ]]; then
    echo "[skip] already inferred and evaluated: $NB_DIR/$NB"
    status "$TAG already done, skipped"
    already=$((already+1))
    echo
    continue
  fi

  # ---------------- 1. inference ----------------
  if [[ -f "$OUT_NC" ]]; then
    echo "[skip] inference - output already exists: $(basename "$OUT_NC")"
    status "$TAG inference output already there, reused"
    reused=$((reused+1))
  else
    EXTRA=()
    [[ "$NORM" == "glorys" ]] && EXTRA+=("+mean=$GLORYS_MEAN" "+std=$GLORYS_STD")
    cd "$INFER" || exit 1
    conda activate "$TRAIN_ENV"
    status "$TAG inference started ($(basename "$INPUT"), patch $PATCH, ensemble $ENS, norm $NORM)"
    status "      progress: tail -c 400 $ILOG | tr '\\r' '\\n' | grep -a Predicting | tail -1"
    t0=$SECONDS
    python inferring.py \
      +input="$INPUT" \
      +checkpoint_path="$CKPT" \
      +config_path="$CFG" \
      +output_path="./outputs/${NAME}.nc" \
      +patch=$PATCH +stride=$STRIDE \
      +period_from=$PERIOD_FROM \
      +period_to=$PERIOD_TO \
      +input_var=sla +output_var=sla \
      +output_dc_format=true \
      +batch_size=1 \
      "${ENS_ARGS[@]}" \
      "${EXTRA[@]}" \
      hydra.run.dir=. hydra.output_subdir=null \
      hydra/job_logging=disabled hydra/hydra_logging=disabled 2>&1 | tee "$ILOG"
    rc=${PIPESTATUS[0]}
    if [[ $rc -ne 0 ]]; then
      echo "!!! inference FAILED (exit $rc) for $NAME - skipping its evaluation"
      if [[ $rc -eq 137 || $rc -eq 255 ]]; then
        echo "    exit 137 / -9 = killed by the OOM killer (host RAM, not GPU)."
      fi
      status "$TAG inference FAILED (exit $rc) after $(hm $((SECONDS - t0))), see $ILOG"
      failed=$((failed+1))
      continue
    fi
    status "$TAG inference finished in $(hm $((SECONDS - t0)))"
  fi

  if [[ ! -f "$OUT_NC" ]]; then
    echo "!!! expected output not found: $OUT_NC"
    status "$TAG FAILED, output file not found: $OUT_NC"
    failed=$((failed+1))
    continue
  fi

  # ---------------- 2. notebook from template ----------------
  cd "$NB_DIR" || exit 1
  if [[ -f "$NB" ]]; then
    echo "[note] notebook exists, re-patching and re-running: $NB"
  else
    cp "$TEMPLATE" "$NB"
    echo "[new ] notebook created from template: $NB"
  fi

  # ---------------- 3. patch method_name / method_path ----------------
  python - "$NB_DIR/$NB" "$NAME" "$OUT_NC" << 'PY'
import sys, json, re
nb_path, name, out_nc = sys.argv[1:4]
nb = json.load(open(nb_path))
done = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "method_name" in src and "method_path" in src:
        src = re.sub(r"method_name\s*=\s*.*", f"method_name = {name!r}", src, count=1)
        # handles both   method_path = ( '...' )   and   method_path = '...'
        src = re.sub(r"method_path\s*=\s*\([\s\S]*?\)", f"method_path = {out_nc!r}", src, count=1)
        src = re.sub(r"method_path\s*=\s*(['\"]).*?\1", f"method_path = {out_nc!r}", src, count=1)
        cell["source"] = src.splitlines(keepends=True)
        done = True
        break
if not done:
    sys.exit("ERROR: no cell containing both method_name and method_path")
json.dump(nb, open(nb_path, "w"), indent=1)
print(f"[ok  ] method_name = {name}")
print(f"[ok  ] method_path = {out_nc}")
PY
  if [[ $? -ne 0 ]]; then
    echo "!!! notebook patch failed for $NAME"
    status "$TAG FAILED to prepare the notebook $NB"
    failed=$((failed+1))
    continue
  fi

  # ---------------- 4. execute the notebook ----------------
  conda activate "$EVAL_ENV"
  echo "[run ] evaluation notebook ..."
  status "$TAG evaluation notebook started"
  t0=$SECONDS
  jupyter nbconvert --to notebook --execute --inplace "$NB" 2>&1 | tee "$NLOG"
  rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "!!! notebook execution FAILED for $NAME"
    status "$TAG notebook FAILED after $(hm $((SECONDS - t0))), see $NLOG"
    failed=$((failed+1))
  else
    touch "$DONE"
    SCORES=$(python - "$NB_DIR/$NB" << 'PY'
import json, re, sys
nb = json.load(open(sys.argv[1]))
txt = ""
for cell in nb["cells"]:
    for o in cell.get("outputs", []):
        t = o.get("text") or o.get("data", {}).get("text/plain", "")
        txt += "".join(t) if isinstance(t, list) else str(t)
parts = []
for label, pat, fmt in [("RMSE", r"Total RMSE=\s*([-0-9.eE+]+)", "{:.4f}"),
                        ("score", r"Total RMSE score=\s*([-0-9.eE+]+)", "{:.4f}"),
                        ("eff. res.", r"[Ee]ffective resolution:\s*([-0-9.eE+]+)", "{:.1f} km")]:
    vals = re.findall(pat, txt)
    if vals:
        parts.append(label + " " + " / ".join(fmt.format(float(v)) for v in vals))
print(", ".join(parts) if parts else "scores not found in the notebook output")
PY
)
    status "$TAG done in $(hm $((SECONDS - t0))): $SCORES"
    echo "[done] scores are in $NB_DIR/$NB"
    ok=$((ok+1))
  fi
  echo
done

status "=== batch finished: $ok evaluated, $failed failed, $reused inferences reused, $already already done"
echo "=========================================================="
echo " ALL DONE   $(date)"
echo "   evaluated now             : $ok"
echo "   failed                    : $failed"
echo "   inferences reused from disk: $reused"
echo "   already done (skipped)    : $already"
echo
echo " result notebooks:"
for job in "${SEL[@]}"; do
  IFS='|' read -r NAME _ <<< "$job"
  nb="$NB_DIR/evaluation_2024_${NAME}.ipynb"
  [[ -f "$nb" ]] && echo "   $nb"
done
echo "=========================================================="