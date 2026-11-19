#!/bin/bash
# =============================================================================
# Fill the last four rows of the universal-vs-specialist table (17 runs).
#
#   GROUP D  UNet2D OSE universal @7 nadirs+SWOT   -> 5 test constellations
#   GROUP E  UNet3D universal                      -> 4 test constellations
#   GROUP F  4DVarNet-56M (2D) universal           -> 4 test constellations
#   GROUP G  4DVarNet-3D-18M universal             -> 4 test constellations
#   (the "Test 7 nadirs+SWOT" cell of E, F, G already exists; the lines that
#    would recompute it are commented out at the end of each group)
#
# For each job: run inference (inferring.py, same args dvc.yaml builds), then
# copy the evaluation notebook from a template, patch method_name/method_path,
# and execute it. params.yaml is NEVER touched.
#
# Each job line is   name|checkpoint|config|input|patch|norm
#   patch : 15-720-1440 for the UNet2D models,
#           15-512-512  for UNet3D and the 4DVarNets (GPU memory), the same
#           patch you used for their existing "Test 7 nadirs+SWOT" cell
#   norm  : self  = mean/std of the input file (what every cell of the table used)
#           train = the stats the model was trained with; needs the inferring.py
#                   from finetune_v2.patch (checked before starting)
#
# RUN IT (tmux -- survives disconnection):
#     tmux new -s infer
#     bash run_missing_cells.sh 2>&1 | tee run_missing_cells.log
#     # detach: Ctrl-b then d      reattach: tmux attach -t infer
#
# Only some jobs: pass part of the job name. Two GPUs -> two tmux windows:
#     bash run_missing_cells.sh Unet     2>&1 | tee log_unet.log      # groups D + E
#     bash run_missing_cells.sh FDVarNet 2>&1 | tee log_4dvarnet.log  # groups F + G
#
# RESUMING: a job whose inference output exists is not re-inferred, and a job
# whose notebook already ran after that output was written is skipped entirely
# (marker file .done_<name> in NB_DIR). So if the batch stops (time limit, OOM),
# just launch it again. Delete the marker to force a re-evaluation.
#
# MEMORY: the merge step needs ~35-40 GB of *host* RAM for a 14-month period.
# Exit 137 / -9 = OOM killer: ask for more memory (--mem=120G), shorten the
# period, or drop ENS_SIZE.
# =============================================================================
# -----------------------------------------------------------------------------
# run_missing_cells_normtrain.sh: copy of run_missing_cells.sh for the norm=train test
# (UNet3D mixed OSSE+OSE and UNet3D OSE held-out, both with the training mean/std).
# SUBMIT:  sbatch run_missing_cells_normtrain.sh
# FOLLOW:  squeue -u $USER      tail -f normtrain_<jobid>.log
# -----------------------------------------------------------------------------
#SBATCH --job-name=normtrain
#SBATCH --partition=Odyssey_GPU
#SBATCH --account=odyssey
#SBATCH --qos=low
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=normtrain_%j.log

# The job runs on a compute node that does not get your Kerberos ticket, so /homes may be
# unreadable there ("Key has expired"). Keep conda/jupyter/python settings and caches out of /homes.
export HOME=/Odyssey/private/n23nguye/home_batch
mkdir -p "$HOME"

set -o pipefail          # NOT -u: conda's MKL activate script uses unset vars
                         # NOT -e: one failed job must not kill the batch

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
PATCH_2D=15-720-1440     # UNet2D
PATCH_BIG=15-512-512     # UNet3D, 4DVarNet-2D, 4DVarNet-3D
STRIDE=1-208-232

# PATCH_2D=15-1296-2880   # the whole 1/8° grid in one patch (15-720-1440 was the whole 1/4° grid)
# STRIDE=1-208-232     # unchanged: the spatial stride is ignored when the patch covers the whole grid

PERIOD_FROM=2023-12-01
PERIOD_TO=2025-01-31
ENS_SIZE=10
ENS_NOISE=0.2

# ------------------------------------------------------------ checkpoints ---
O=$REPO/outputs

# #UNet 2D 1/8th resolution kmax 8
# CKPT_2D="$O/2026-11-05/osse_random_pretrain_Resolution8th_UNet2D_GAPFREEtarget_k2to8-08-06-31/unet-osse-random-UNet2D_GAPFREEtarget_8th/checkpoints/val_mse0p76036-epoch809.ckpt"
# CFG_2D="$O/2026-11-05/osse_random_pretrain_Resolution8th_UNet2D_GAPFREEtarget_k2to8-08-06-31/.hydra/config.yaml"

#UNet 3D 1/4th resolution kmax 7, training data start from July 2023
CKPT_3D="$O/2026-11-15/UNet3DTransformer_bs4n4_mixed_OSSEgapfree_OSEheldout_7nadirNSwot_train20232025withoutAlg-18-30-25/UNet3DTransformer-real-random_train20232025woAlg_heldout_mixOSSE/checkpoints/val_mse3p26594-epoch389.ckpt" #val_msep3p23775-epoch734.ckpt"
CFG_3D="$O/2026-11-15/UNet3DTransformer_bs4n4_mixed_OSSEgapfree_OSEheldout_7nadirNSwot_train20232025withoutAlg-18-30-25/.hydra/config.yaml"


#UNet 3D 1/4th resolution, OSE only, held-out loss (check: same checkpoint as for its 190.9 row)
RUN_HO="$O/2026-11-13/UNet3DTransformer_bs8_real_random_7nadirNSwot_train20232025withoutAlg_heldoutTarget-09-31-21"
CKPT_HO="$RUN_HO/UNet3DTransformer-real-random_train20232025woAlg_LossOnlyOnHeldOutPixels/checkpoints/val_mse3p46611-epoch302.ckpt"
CFG_HO="$RUN_HO/.hydra/config.yaml"


# #UNet 3D 1/8th resolution kmax 7, training data start from July 2023
# CKPT_3D_8th="$O/2026-11-09/Resolution8th_UNet3DTransformer_size384_kmax7_real_random_7nadirNSwot_train20232025withoutAlg_FromJuly2023_precisionbf16mixed-15-02-01/UNet3DTransformer-real-random_train20232025woAlg_trainfromJuly2023_8thresolution.yaml/checkpoints/val_mse2p00651-epoch095.ckpt"
# CFG_3D_8th="$O/2026-11-09/Resolution8th_UNet3DTransformer_size384_kmax7_real_random_7nadirNSwot_train20232025withoutAlg_FromJuly2023_precisionbf16mixed-15-02-01/.hydra/config.yaml"


# ------------------------------------------------------- test constellations -
DATA=/SCRATCH/n23nguye/mapping_2023_2025
IN_7NS="$DATA/alongtrack_S7nadirsNswotKarinWithLweRemoved_real_2023-01-01_2025-07-31.nc"


# ------------------------------- the jobs: name|ckpt|cfg|input|patch|norm ---
# cheapest first, so the first cells arrive early
# UNet3DTransformer-real-random_train20232025woAlg_heldout_mixOSSE
JOBS=(
  # "Unet2D_Train6nadirs_TestSWOTnSwon|$CKPT_6N|$CFG_6N|$IN_SWSW|$PATCH_2D|self"
  # "Unet2D_Train6nadirs_Test2nadirs|$CKPT_6N|$CFG_6N|$IN_2N|$PATCH_2D|self"
  # "Unet2D_Train6nadirs_Test4nadirs|$CKPT_6N|$CFG_6N|$IN_4N|$PATCH_2D|self"
  # "Unet2D_Train6nadirs_Test6nadirs|$CKPT_6N|$CFG_6N|$IN_6N|$PATCH_2D|self"
  # "Resolution8th_UNet2D_osse_random_k2to8_Test7nadirsNSWOT|$CKPT_2D|$CFG_2D|$IN_7NS|$PATCH_2D|self"
  # "UNet3D_real_random_k2to7_heldout_mixOSSE_Test7nadirsNSWOT|$CKPT_3D|$CFG_3D|$IN_7NS|$PATCH_BIG|self"
  "UNet3D_real_random_k2to7_heldout_mixOSSE_Epoch389_normtrain_Test7nadirsNSWOT|$CKPT_3D|$CFG_3D|$IN_7NS|$PATCH_BIG|train"
  # "UNet3D_real_random_k2to7_heldout_normtrain_Test7nadirsNSWOT|$CKPT_HO|$CFG_HO|$IN_7NS|$PATCH_BIG|train"
  # "Resolution8th_UNet3D_real_random_k2to7_trainingdatafromJuly2023_Test7nadirsNSWOT|$CKPT_3D_8th|$CFG_3D_8th|$IN_7NS|$PATCH_BIG|self"
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
source /Odyssey/private/n23nguye/envs/nga/miniconda/etc/profile.d/conda.sh   # explicit path: works from sbatch

echo "=========================================================="
echo " filling ${#SEL[@]} table cells ${ONLY:+(filter: $ONLY) }  $(date)"
echo "=========================================================="

# ---- pre-flight: every file must exist before starting a long batch --------
missing=0
need_norm=0
for job in "${SEL[@]}"; do
  IFS='|' read -r NAME CKPT CFG INPUT PATCH NORM <<< "$job"
  for f in "$CKPT" "$CFG" "$INPUT"; do
    [[ -f "$f" ]] || { echo "MISSING FILE ($NAME): $f"; missing=1; }
  done
  [[ "$PATCH" =~ ^[0-9]+-[0-9]+-[0-9]+$ ]] || { echo "BAD PATCH ($NAME): '$PATCH'"; missing=1; }
  case "$NORM" in
    self) ;;
    train) need_norm=1 ;;
    *) echo "BAD NORM ($NAME): '$NORM' (self or train)"; missing=1 ;;
  esac
done
[[ -f "$NB_DIR/$TEMPLATE" ]] || { echo "MISSING TEMPLATE: $NB_DIR/$TEMPLATE"; missing=1; }
if [[ $missing -eq 1 ]]; then
  echo
  echo "Fix the problems listed above before running. Aborting."
  exit 1
fi
echo "all checkpoints, configs and inputs found."
echo

ok=0; failed=0; reused=0; already=0

# =============================================================================
for job in "${SEL[@]}"; do
  IFS='|' read -r NAME CKPT CFG INPUT PATCH NORM <<< "$job"
  OUT_NC="$OUTDIR/${NAME}-n${ENS_SIZE}-std${ENS_NOISE}.nc"
  NB="evaluation_2024_${NAME}.ipynb"
  DONE="$NB_DIR/.done_${NAME}"

  echo "=========================================================="
  echo ">>> $NAME"
  echo "    input : $(basename "$INPUT")"
  echo "    ckpt  : $(basename "$CKPT")"
  echo "    patch : $PATCH   norm: $NORM"
  echo "    start : $(date +%H:%M:%S)"
  echo "=========================================================="

  # ---------------- 0. already evaluated? ----------------
  if [[ -f "$OUT_NC" && -f "$DONE" && "$DONE" -nt "$OUT_NC" ]]; then
    echo "[skip] already inferred and evaluated: $NB_DIR/$NB"
    already=$((already+1))
    echo
    continue
  fi

  # ---------------- 1. inference ----------------
  if [[ -f "$OUT_NC" ]]; then
    echo "[skip] inference - output already exists: $(basename "$OUT_NC")"
    reused=$((reused+1))
  else
    EXTRA=()
    cd "$INFER" || exit 1
    conda activate "$TRAIN_ENV"
    if [[ "$NORM" == "train" ]]; then   # training mean/std from the config -> +mean/+std
      read -r NMEAN NSTD < <(python -c "
from omegaconf import OmegaConf; import sys
m, s = OmegaConf.load(sys.argv[1]).datamodule.norm_stats.train; print(m, s)" "$CFG")
      if [[ -z "$NSTD" ]]; then
        echo "!!! could not read datamodule.norm_stats.train from $CFG"
        failed=$((failed+1))
        continue
      fi
      EXTRA+=(+mean=$NMEAN +std=$NSTD)
      echo "[norm] training normalisation: mean=$NMEAN std=$NSTD"
    fi
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
      "${EXTRA[@]}" \
      hydra.run.dir=. hydra.output_subdir=null \
      hydra/job_logging=disabled hydra/hydra_logging=disabled
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "!!! inference FAILED (exit $rc) for $NAME - skipping its evaluation"
      if [[ $rc -eq 137 || $rc -eq 255 ]]; then
        echo "    exit 137 / -9 = killed by the OOM killer (host RAM, not GPU)."
        echo "    -> raise --mem in the #SBATCH lines, or shorten PERIOD_FROM/TO, or lower ENS_SIZE."
      fi
      echo "    CUDA out of memory? -> use a smaller patch for this model; with stride"
      echo "    $STRIDE the lat size must be one of 720/512/304 and lon one of 1440/976/512/280."
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
    failed=$((failed+1))
    continue
  fi

  # ---------------- 4. execute the notebook ----------------
  conda activate "$EVAL_ENV"
  echo "[run ] evaluation notebook ..."
  time jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=python3 "$NB"
  if [[ $? -ne 0 ]]; then
    echo "!!! notebook execution FAILED for $NAME"
    failed=$((failed+1))
  else
    touch "$DONE"
    echo "[done] scores are in $NB_DIR/$NB"
    ok=$((ok+1))
  fi
  echo
done

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