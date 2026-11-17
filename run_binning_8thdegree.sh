#!/bin/bash
# =============================================================================
# Bin the L3 along-track nadirs and SWOT KaRIn onto the 1/8 deg GLORYS grid.
# Each run writes, in the notebook's OUT_DIR:
#   alongtrack_<TAG>_real_2023-01-01_2025-07-31.nc   daily 2-D map (OSE input)
#   mask_<TAG>_2024_365d.nc                          2024 footprint (OSSE training mask)
#
# It executes the 1/8 deg copy of the ONLY_SWOT regrid notebook (that notebook
# handles nadirs only, KaRIn only, and both), whose cell 3 reads NADIRS /
# USE_KARIN / TAG from the environment. The notebook itself is not modified here.
#
# RUN (tmux, survives disconnection):
#     tmux new -s bin8
#     bash run_binning_8th.sh 2>&1 | tee run_binning_8th.log
# Only some runs (filter on the TAG):
#     bash run_binning_8th.sh s3a
# =============================================================================
#SBATCH --job-name=bin8th
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=bin8th_%j.log

set -o pipefail          # NOT -e: one failed run must not stop the others
ONLY="${1:-}"

REPO=/Odyssey/private/n23nguye/4dvarnet-global-mapping-Daniel-version
NB=Regrid_from_1D_to_2D_for_2024_SWOTnNadirs_2ConfigInOne_KarinWithLweRemoved_ONLY_SWOT_8th.ipynb
EXEC_DIR=$REPO/binning_8th_executed    # executed copy of the notebook for each run
ENV=dc                                 # the env you used for the 1/4 deg regrid notebooks
                                       # (needs jupyter, xarray, pyinterp, tqdm, matplotlib)

#   TAG | nadir codes, comma-separated ("" = none) | 1 = add SWOT KaRIn
RUNS=(
  # ---- single sources: OSSE masks (+ per-satellite OSE inputs) ----
  "alg|alg|0"
  "c2n|c2n|0"
  "h2b|h2b|0"
  "j3n|j3n|0"
  "s3a|s3a|0"
  "s3b|s3b|0"
  "s6alr|s6a-lr|0"
  "swon|swon|0"
  "SWOT||1"
  # ---- test constellations (OSE inference inputs) ----
  # "only2nadirss3aNs6a|s3a,s6a-lr|0"
  # "only4nadirsj3ns3as3bs6a|j3n,s3a,s3b,s6a-lr|0"
  # "only6nadirs|c2n,h2b,j3n,s3a,s3b,s6a-lr|0"
  "S7nadirsNswotKarinWithLweRemoved|c2n,h2b,j3n,s3a,s3b,s6a-lr,swon|1"
)

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
cd "$REPO" || exit 1
[[ -f "$NB" ]] || { echo "missing notebook: $REPO/$NB (copy and edit the ONLY_SWOT notebook first)"; exit 1; }
mkdir -p "$EXEC_DIR"

ok=0; failed=0
for r in "${RUNS[@]}"; do
  IFS='|' read -r TAG NADIRS USE_KARIN <<< "$r"
  [[ -n "$ONLY" && "$TAG" != *"$ONLY"* ]] && continue
  echo "=========================================================="
  echo ">>> $TAG   nadirs: ${NADIRS:-none}   KaRIn: $USE_KARIN   start $(date +%H:%M:%S)"
  if TAG="$TAG" NADIRS="$NADIRS" USE_KARIN="$USE_KARIN" \
     jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
       --output-dir "$EXEC_DIR" --output "binned8th_${TAG}.ipynb" "$NB"; then
    ok=$((ok+1)); echo "[done] $TAG   $(date +%H:%M:%S)"
  else
    failed=$((failed+1)); echo "!!! $TAG failed - see the traceback above"
  fi
done

echo "=========================================================="
echo " finished: $ok ok, $failed failed   $(date)"
echo " next: run Combine_SWOT_n_Swon.ipynb with D = the 1/8 deg OUT_DIR"
echo "       (writes the SWOT & swon test input)"
echo "=========================================================="