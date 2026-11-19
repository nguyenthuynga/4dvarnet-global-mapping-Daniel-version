#!/bin/bash
# =============================================================================
# Inference + evaluation of the 1/8 degree UNet3D (held-out target), test 7 nadirs + SWOT.
# Same steps as run_missing_cells.sh (inference with inferring.py, then a copy of the
# evaluation notebook patched and executed), for one job.
#
# SUBMIT (from the login node, after `ssh slurm`):
#     cd /Odyssey/private/n23nguye/4dvarnet-global-mapping-Daniel-version
#     sbatch run_infer_eval_8th.sh
# FOLLOW:  squeue -u $USER      tail -f infer_eval_8th_<jobid>.log      scancel <jobid>
#
# MEMORY: 15-512-512 patches on the 1/8 grid = 18,630 patches. The ORIGINAL inferring.py keeps
# them all in RAM (~300 GB) and gets killed; the patched inferring.py (merge while predicting,
# "_merge_pending") needs ~30-45 GB. The pre-flight check below stops if the patch is missing.
# =============================================================================
#SBATCH --job-name=infer_eval_8th
#SBATCH --partition=Odyssey_GPU
#SBATCH --account=odyssey
#SBATCH --qos=low
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --time=48:00:00
#SBATCH --output=infer_eval_8th_%j.log

set -o pipefail          # NOT -u: conda's MKL activate script uses unset vars
                         # NOT -e: one failed job must not kill the batch

# ---------------------------------------------------------------- paths -----
REPO=/Odyssey/private/n23nguye/4dvarnet-global-mapping-Daniel-version
INFER=$REPO/inference
OUTDIR=$INFER/outputs
NB_DIR=/Odyssey/private/n23nguye/2023a_SSH_mapping_OSE/Notebooks
TEMPLATE=evaluation_2024_with_without_lwe_MaskMiost_TrainOSSEFix4Nadirs_Test4Nadirs_Unet.ipynb

TRAIN_ENV=4dvarnet-gm    # inference
EVAL_ENV=dc              # notebooks

# ------------------------------------------------------- inference knobs ----
PATCH=15-512-512         # 1/8 degree (was 15-720-1440 at 1/4)
STRIDE=1-196-296
PERIOD_FROM=2023-12-01
PERIOD_TO=2025-01-31
ENS_SIZE=10
ENS_NOISE=0.2
USE_TRAIN_NORM=0         # 0: normalise with the input file's mean/std (as in your previous runs)
                         # 1: use datamodule.norm_stats.train of the training config (+mean/+std)

# ------------------------------------------------------------ checkpoint ----
RUN8="$REPO/outputs/2026-11-15/Resolution8th_UNet3DTransformer_size384_kmax7_real_random_7nadirNSwot_train20232025withoutAlg_precisionbf16mixed_HeldOutTarget-17-17-30"
CKPT_8TH="$RUN8/UNet3DTransformer-real-random_train20232025woAlg_8thresolution_HeldOutTarget/checkpoints/val_mse3p66171-epoch173.ckpt" #val_mse3p78371-epoch089.ckpt"
CFG_8TH="$RUN8/.hydra/config.yaml"

# ---------------------------------------------------------- test input --------
IN_7NS_8TH="/SCRATCH/n23nguye/mapping_2023_2025_8th/alongtrack_S7nadirsNswotKarinWithLweRemoved_real_2023-01-01_2025-07-31.nc"

# --------------------------------------------- the jobs: name|ckpt|cfg|input --
JOBS=(
  "Resolution8th_UNet3DTransformer_size384_kmax7_real_random_7nadirNSwot_train20232025withoutAlg_precisionbf16mixed_HeldOutTarget_Epoch173_Test7nadirsNSWOT|$CKPT_8TH|$CFG_8TH|$IN_7NS_8TH"
)

# =============================================================================
source /Odyssey/private/n23nguye/envs/nga/miniconda/etc/profile.d/conda.sh

echo "=========================================================="
echo " filling ${#JOBS[@]} table cells   $(date)"
echo "=========================================================="

# ---- pre-flight: every file must exist before starting a long batch --------
missing=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME CKPT CFG INPUT <<< "$job"
  for f in "$CKPT" "$CFG" "$INPUT"; do
    [[ -f "$f" ]] || { echo "MISSING FILE: $f"; missing=1; }
  done
done
[[ -f "$NB_DIR/$TEMPLATE" ]] || { echo "MISSING TEMPLATE: $NB_DIR/$TEMPLATE"; missing=1; }
if ! grep -q "_merge_pending" "$INFER/inferring.py"; then
  echo "inferring.py is the ORIGINAL version: at 1/8 it keeps ~300 GB of patches in RAM and"
  echo "gets killed. Put the patched inferring.py (with _merge_pending) in $INFER first."
  missing=1
fi
if [[ $missing -eq 1 ]]; then
  echo
  echo "Fix the paths listed above before running (duplicates are normal -"
  echo "the same file is checked once per job). Aborting."
  exit 1
fi
echo "all checkpoints, configs and inputs found."
echo

ok=0; failed=0; skipped=0

# =============================================================================
for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME CKPT CFG INPUT <<< "$job"
  OUT_NC="$OUTDIR/${NAME}-n${ENS_SIZE}-std${ENS_NOISE}.nc"
  NB="evaluation_2024_${NAME}.ipynb"

  echo "=========================================================="
  echo ">>> $NAME"
  echo "    input : $(basename "$INPUT")"
  echo "    ckpt  : $(basename "$CKPT")"
  echo "    start : $(date +%H:%M:%S)"
  echo "=========================================================="

  # ---------------- 1. inference ----------------
  if [[ -f "$OUT_NC" ]]; then
    echo "[skip] inference - output already exists: $(basename "$OUT_NC")"
    skipped=$((skipped+1))
  else
    cd "$INFER" || exit 1
    conda activate "$TRAIN_ENV"
    NORM_ARGS=()
    if [[ $USE_TRAIN_NORM -eq 1 ]]; then
      read -r NMEAN NSTD < <(python -c "
from omegaconf import OmegaConf; import sys
m, s = OmegaConf.load(sys.argv[1]).datamodule.norm_stats.train; print(m, s)" "$CFG")
      NORM_ARGS=(+mean=$NMEAN +std=$NSTD)
      echo "[norm] training normalisation: mean=$NMEAN std=$NSTD"
    fi
    echo "[run ] inference ..."
    time python inferring.py "${NORM_ARGS[@]}" \
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
      +ensemble_noise=$ENS_NOISE +ensemble_size=$ENS_SIZE \
      hydra.run.dir=. hydra.output_subdir=null \
      hydra/job_logging=disabled hydra/hydra_logging=disabled
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "!!! inference FAILED (exit $rc) for $NAME - skipping its evaluation"
      if [[ $rc -eq 137 || $rc -eq 255 ]]; then
        echo "    exit 137 / -9 = killed by the OOM killer (host RAM, not GPU)."
        echo "    -> check that the patched inferring.py is used, or raise --mem in the #SBATCH lines."
      fi
      failed=$((failed+1))
      continue
    fi
  fi

  if [[ ! -f "$OUT_NC" ]]; then
    echo "!!! expected output not found: $OUT_NC"
    echo "    (check the -n/-std suffix that inferring.py appends)"
    failed=$((failed+1))
    continue
  fi

  # ---------------- 2. notebook from template ----------------
  cd "$NB_DIR" || exit 1
  if [[ -f "$NB" ]]; then
    echo "[skip] notebook exists, re-patching and re-running: $NB"
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
    failed=$((failed+1))
    continue
  fi

  # ---------------- 4. execute the notebook ----------------
  conda activate "$EVAL_ENV"
  echo "[run ] evaluation notebook ..."
  time jupyter nbconvert --to notebook --execute --inplace "$NB"
  if [[ $? -ne 0 ]]; then
    echo "!!! notebook execution FAILED for $NAME"
    failed=$((failed+1))
  else
    echo "[done] scores are in $NB_DIR/$NB"
    ok=$((ok+1))
  fi
  echo
done

echo "=========================================================="
echo " ALL DONE   $(date)"
echo "   evaluated ok : $ok"
echo "   failed       : $failed"
echo "   inferences reused from disk : $skipped"
echo
echo " result notebooks:"
for job in "${JOBS[@]}"; do
  IFS='|' read -r NAME _ _ _ <<< "$job"
  nb="$NB_DIR/evaluation_2024_${NAME}.ipynb"
  [[ -f "$nb" ]] && echo "   $nb"
done
echo "=========================================================="