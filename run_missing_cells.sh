#!/bin/bash
# =============================================================================
# Fill the missing cells of the universal-vs-specialist table (14 runs).
#
#   GROUP A  new "@SWOT & swon" training row        -> 5 test constellations
#   GROUP B  corrected "@7 nadirs+SWOT" row         -> 5 test constellations
#   GROUP C  new "SWOT & swon" TEST column          -> 4 existing checkpoints
#
# For each job: run inference (inferring.py, same args dvc.yaml builds), then
# copy the evaluation notebook from a template, patch method_name/method_path,
# and execute it.  params.yaml is NEVER touched -- inferring.py is called
# directly, so your commented history stays intact and dvc caching can't skip
# a run.
#
# RUN IT (tmux -- survives disconnection):
#     tmux new -s infer
#     bash run_missing_cells.sh 2>&1 | tee run_missing_cells.log
#     # detach: Ctrl-b then d      reattach: tmux attach -t infer
#
# To run only part of it, comment out lines in the JOBS array below.
#
# MEMORY: patch 15-720-1440 with a 14-month period needs ~35-40 GB of *host*
# RAM in the merge step.  If a job dies with exit 137 / -9 it was the OOM
# killer: ask for more memory in your salloc (--mem=120Gb), or shorten the
# period, or drop ENS_SIZE.
# =============================================================================
#SBATCH --job-name=fill_table
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=fill_table_%j.log

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
PATCH=15-720-1440
STRIDE=1-208-232
PERIOD_FROM=2023-12-01
PERIOD_TO=2025-01-31
ENS_SIZE=10
ENS_NOISE=0.2

# ------------------------------------------------------------ checkpoints ---
### NEW: correct 7 nadirs + SWOT
CKPT_7NS="$REPO/outputs/2026-09-03/S7nadirsNSwot_unet-18-40-37/unet-masked/checkpoints/val_mse0p74788-epoch392.ckpt"
CFG_7NS="$REPO/outputs/2026-09-03/S7nadirsNSwot_unet-18-40-37/.hydra/config.yaml"

### NEW: SWOT and swon only
CKPT_SWSW="$REPO/outputs/2026-09-03/osse_SWOTnSwon_unet-14-37-52/unet-osse-mask-SWOT-n-Swon/checkpoints/val_mse1p48048-epoch470.ckpt"
CFG_SWSW="$REPO/outputs/2026-09-03/osse_SWOTnSwon_unet-14-37-52/.hydra/config.yaml"

### existing rows (for the new SWOT&swon test column)
CKPT_2N="$REPO/outputs/2026-06-18/Only2nadirs_s3aNs6a2024-21-17-16/unet-masked/checkpoints/val_mse_1p86143_epoch_419.ckpt"
CFG_2N="$REPO/outputs/2026-06-18/Only2nadirs_s3aNs6a2024-21-17-16/.hydra/config.yaml"

CKPT_4N="$REPO/outputs/2026-08-28/only4nadirsj3ns3as3bs6a_unet-17-50-28/unet-masked/checkpoints/val_mse1p10308-epoch599.ckpt"
CFG_4N="$REPO/outputs/2026-08-28/only4nadirsj3ns3as3bs6a_unet-17-50-28/.hydra/config.yaml"

CKPT_6N="$REPO/outputs/2026-06-12/Only6Nadirs2024-18-33-06/unet-masked/checkpoints/val_mse_096402_epoch_521.ckpt"
CFG_6N="$REPO/outputs/2026-06-12/Only6Nadirs2024-18-33-06/.hydra/config.yaml"

CKPT_UNI="$REPO/outputs/2026-08-14/osse_random_pretrain_UNet2D_GAPFREEtarget_bs16-12-28-15/unet-osse-random-UNet2D_GAPFREEtarget/checkpoints/val_mse1p07304-epoch578.ckpt"
CFG_UNI="$REPO/outputs/2026-08-14/osse_random_pretrain_UNet2D_GAPFREEtarget_bs16-12-28-15/.hydra/config.yaml"

# ------------------------------------------------------- test constellations -
DATA=/SCRATCH/n23nguye/mapping_2023_2025
IN_6N="$DATA/alongtrack_only6nadirs_real_2023-01-01_2025-07-31.nc"
IN_4N="$DATA/alongtrack_only4nadirsj3ns3as3bs6a_real_2023-01-01_2025-07-31.nc"
IN_2N="$DATA/alongtrack_only2nadirss3aNs6a_real_2023-01-01_2025-07-31.nc"
IN_7NS="$DATA/alongtrack_S7nadirsNswotKarinWithLweRemoved_real_2023-01-01_2025-07-31.nc"
IN_SWSW="$DATA/alongtrack_SWOTnSwon_real_2023-01-01_2025-07-31.nc"

# --------------------------------------------- the jobs: name|ckpt|cfg|input --
JOBS=(
  # ---- GROUP A : new "@SWOT & swon" training row ----
  "Unet2D_TrainSWOTnSwon_Test6nadirs|$CKPT_SWSW|$CFG_SWSW|$IN_6N"
  "Unet2D_TrainSWOTnSwon_Test4nadirs|$CKPT_SWSW|$CFG_SWSW|$IN_4N"
  "Unet2D_TrainSWOTnSwon_Test2nadirs|$CKPT_SWSW|$CFG_SWSW|$IN_2N"
  "Unet2D_TrainSWOTnSwon_TestSWOTnSwon|$CKPT_SWSW|$CFG_SWSW|$IN_SWSW"
  "Unet2D_TrainSWOTnSwon_Test7nadirsNSWOT|$CKPT_SWSW|$CFG_SWSW|$IN_7NS"

  # ---- GROUP B : corrected "@7 nadirs+SWOT" training row ----
  "Unet2D_Train7nadirsNSWOT_Test6nadirs|$CKPT_7NS|$CFG_7NS|$IN_6N"
  "Unet2D_Train7nadirsNSWOT_Test4nadirs|$CKPT_7NS|$CFG_7NS|$IN_4N"
  "Unet2D_Train7nadirsNSWOT_Test2nadirs|$CKPT_7NS|$CFG_7NS|$IN_2N"
  "Unet2D_Train7nadirsNSWOT_TestSWOTnSwon|$CKPT_7NS|$CFG_7NS|$IN_SWSW"
  "Unet2D_Train7nadirsNSWOT_Test7nadirsNSWOT|$CKPT_7NS|$CFG_7NS|$IN_7NS"

  # ---- GROUP C : new "SWOT & swon" TEST column for existing rows ----
  "Unet2D_Train2nadirs_TestSWOTnSwon|$CKPT_2N|$CFG_2N|$IN_SWSW"
  "Unet2D_Train4nadirs_TestSWOTnSwon|$CKPT_4N|$CFG_4N|$IN_SWSW"
  "Unet2D_Train6nadirs_TestSWOTnSwon|$CKPT_6N|$CFG_6N|$IN_SWSW"
  "Unet2D_TrainUniversal_TestSWOTnSwon|$CKPT_UNI|$CFG_UNI|$IN_SWSW"
)

# =============================================================================
source "$(conda info --base)/etc/profile.d/conda.sh"

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
    echo "[run ] inference ..."
    time python inferring.py \
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
        echo "    -> salloc with --mem=120Gb, or shorten PERIOD_FROM/TO, or lower ENS_SIZE."
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