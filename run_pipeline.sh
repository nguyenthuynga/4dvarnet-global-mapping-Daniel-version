#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — one command for the whole train -> infer -> evaluate chain.
#
# Does exactly your manual steps, changing NONE of Daniel's code:
#   1. train         : python main.py xp=<XP>
#   2. pick ckpt     : find the best (lowest val_mse) checkpoint of THIS run
#   3. rename ckpt   : val_mse=0.97-epoch=353.ckpt -> val_mse0p97epoch353.ckpt
#   4. patch params  : uncomment/insert name+checkpoint+config+input_data in
#                      inference/params.yaml  (keeps everything else, comments incl.)
#   5. dvc repro     : run inference
#   6. evaluate      : nbconvert --execute the eval notebook (in the 'dc' env)
#
# You can also SKIP training and point at an existing run dir (e.g. re-evaluating
# an old checkpoint) with --run-dir.
#
# Usage:
#   sbatch run_pipeline.sh \
#       --xp        FDVarNet_3D-osse-random_GAPFREEtarget \
#       --input     /SCRATCH/n23nguye/mapping_2023_2025/alongtrack_S7nadirsNswotKarinWithLweRemoved_real_2023-01-01_2025-07-31.nc \
#       --name      FDVarNet_bs2_7nadirNSwot_osse_random_pretrain_GAPFREEtarget \
#       --notebook  evaluation_2024_without_lwe_7nadirsNswot_4DVarNet_OSSEpretrainedGAPFREEtarget.ipynb
#
# Optional:
#   --run-dir <path>   skip training, use this outputs/<...> dir instead
#   --patch  15-512-512   override dims.patch in params.yaml
#   --stride 1-208-232    override dims.stride in params.yaml
#   --ens-size 10  --ens-noise 0.2
#   --no-train / --no-eval   run only part of the chain
# =============================================================================
#SBATCH --job-name=sla_pipeline_FDVarNet_trainOSE
#SBATCH --partition=Odyssey_GPU
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=240:00:00
#SBATCH --output=pipeline_%j.log

set -euo pipefail

# ---- paths (edit once if your layout ever changes) --------------------------
REPO=/Odyssey/private/n23nguye/4dvarnet-global-mapping-Daniel-version
INFER_DIR=$REPO/inference
PARAMS=$INFER_DIR/params.yaml
NB_DIR=/Odyssey/private/n23nguye/2023a_SSH_mapping_OSE/Notebooks
TRAIN_ENV=4dvarnet-gm
EVAL_ENV=dc

# ---- defaults ---------------------------------------------------------------
XP=""; INPUT=""; NAME=""; NOTEBOOK=""; RUN_DIR=""
PATCH=""; STRIDE=""; ENS_SIZE=""; ENS_NOISE=""
DO_TRAIN=1; DO_EVAL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --xp)        XP="$2"; shift 2;;
    --input)     INPUT="$2"; shift 2;;
    --name)      NAME="$2"; shift 2;;
    --notebook)  NOTEBOOK="$2"; shift 2;;
    --run-dir)   RUN_DIR="$2"; DO_TRAIN=0; shift 2;;
    --patch)     PATCH="$2"; shift 2;;
    --stride)    STRIDE="$2"; shift 2;;
    --ens-size)  ENS_SIZE="$2"; shift 2;;
    --ens-noise) ENS_NOISE="$2"; shift 2;;
    --no-train)  DO_TRAIN=0; shift;;
    --no-eval)   DO_EVAL=0; shift;;
    --template)  TEMPLATE="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

[[ -z "$NAME" ]] && { echo "ERROR: --name is required (the inference method name)"; exit 1; }

# make conda usable inside a non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# =============================================================================
# 1. TRAIN
# =============================================================================
if [[ $DO_TRAIN -eq 1 ]]; then
  [[ -z "$XP" ]] && { echo "ERROR: --xp required unless --run-dir is given"; exit 1; }
  echo "======== [1/6] TRAIN  xp=$XP ========"
  cd "$REPO"
  export MKL_INTERFACE_LAYER="" #Chatgpt suggested this to solve this bug: ligne 1: MKL_INTERFACE_LAYER : variable sans liaison
  conda activate "$TRAIN_ENV"
  # snapshot existing run dirs so we can identify the NEW one afterwards
  before=$(mktemp)
  find "$REPO/outputs" -maxdepth 2 -type d 2>/dev/null | sort > "$before"
  time python main.py xp="$XP"
  after=$(mktemp)
  find "$REPO/outputs" -maxdepth 2 -type d 2>/dev/null | sort > "$after"
  # newest dir that didn't exist before = this run
  RUN_DIR=$(comm -13 "$before" "$after" | grep -v '/checkpoints$' | \
            xargs -r -I{} stat --format='%Y {}' {} | sort -nr | head -1 | cut -d' ' -f2-)
  rm -f "$before" "$after"
  [[ -z "$RUN_DIR" ]] && { echo "ERROR: could not detect the new run dir"; exit 1; }
fi

echo "Run dir: $RUN_DIR"
[[ ! -d "$RUN_DIR" ]] && { echo "ERROR: run dir does not exist: $RUN_DIR"; exit 1; }

# =============================================================================
# 2 + 3. PICK BEST CHECKPOINT + RENAME (strip '=' that dvc/hydra dislike)
# =============================================================================
echo "======== [2/6] pick best checkpoint ========"
# the training config is always here — used as params.yaml 'config:'
CONFIG_PATH="$RUN_DIR/.hydra/config.yaml"
[[ ! -f "$CONFIG_PATH" ]] && { echo "ERROR: no .hydra/config.yaml under $RUN_DIR"; exit 1; }

# lowest val_mse checkpoint (filenames look like val_mse=0.97517-epoch=353.ckpt)
CKPT_DIR=$(find "$RUN_DIR" -type d -name checkpoints | head -1)
[[ -z "$CKPT_DIR" ]] && { echo "ERROR: no checkpoints dir under $RUN_DIR"; exit 1; }

BEST_CKPT=$(python - "$CKPT_DIR" <<'PY'
import sys, re, glob, os
best, bestval = None, float("inf")
for f in glob.glob(os.path.join(sys.argv[1], "*.ckpt")):
    m = re.search(r"val_mse[=_]?([0-9]+[.p][0-9]+)", os.path.basename(f))
    if not m:                      # e.g. 'last.ckpt' — skip
        continue
    v = float(m.group(1).replace("p", "."))
    if v < bestval:
        best, bestval = f, v
print(best if best else "")
PY
)
[[ -z "$BEST_CKPT" ]] && { echo "ERROR: no val_mse checkpoint found in $CKPT_DIR"; exit 1; }
echo "best checkpoint: $BEST_CKPT  (val_mse minimum)"

echo "======== [3/6] rename checkpoint (remove '=') ========"
# val_mse=0.97517-epoch=353.ckpt  ->  val_mse0p97517epoch353.ckpt
base=$(basename "$BEST_CKPT")
newbase=$(echo "$base" | sed -e 's/=//g' -e 's/\./p/g' -e 's/pckpt$/.ckpt/')
CKPT_RENAMED="$CKPT_DIR/$newbase"
if [[ "$BEST_CKPT" != "$CKPT_RENAMED" ]]; then
  cp -n "$BEST_CKPT" "$CKPT_RENAMED"   # copy, keep original; -n = never clobber
fi
echo "renamed -> $CKPT_RENAMED"

# =============================================================================
# 4. PATCH params.yaml  (preserve every existing line, comments included)
# =============================================================================
echo "======== [4/6] patch params.yaml ========"
cp "$PARAMS" "$PARAMS.bak.$(date +%s)"      # timestamped backup, just in case

python - "$PARAMS" "$NAME" "$CKPT_RENAMED" "$CONFIG_PATH" "$INPUT" \
        "$PATCH" "$STRIDE" "$ENS_SIZE" "$ENS_NOISE" <<'PY'
import sys, re

params, name, ckpt, cfg, inp, patch, stride, ens_size, ens_noise = sys.argv[1:10]
lines = open(params).read().splitlines()

def set_active(lines, key, value, quote=True):
    """Comment out ALL existing (active OR commented) `key:` lines under method,
    then insert one active line. Nothing is deleted — old lines stay as comments."""
    v = f'"{value}"' if quote else value
    out, inserted = [], False
    pat = re.compile(rf'^\s*#?\s*{re.escape(key)}\s*:')
    for ln in lines:
        if pat.match(ln) and not ln.lstrip().startswith('#'):
            out.append('  # ' + ln.strip())          # demote the old active line
        else:
            out.append(ln)
    # insert the fresh active line right before the first 'input_var:' anchor
    final = []
    for ln in out:
        if not inserted and re.match(r'^\s*input_var\s*:', ln):
            final.append(f'  {key}: {v}')
            inserted = True
        final.append(ln)
    if not inserted:                                   # no anchor -> append at end
        final.append(f'  {key}: {v}')
    return final

lines = set_active(lines, 'name', name)
lines = set_active(lines, 'checkpoint', ckpt)
lines = set_active(lines, 'config', cfg)
if inp:
    lines = set_active(lines, 'input_data', inp)

# dims block lives at top-level (not under method:)
def set_dims(lines, key, value):
    if not value:
        return lines
    out, done = [], False
    pat = re.compile(rf'^\s*#?\s*{re.escape(key)}\s*:')
    for ln in lines:
        if pat.match(ln) and not done:
            out.append(f'  {key}: {value}')
            done = True
        else:
            out.append(ln)
    return out

lines = set_dims(lines, 'patch', patch)
lines = set_dims(lines, 'stride', stride)
if ens_size:
    lines = set_dims(lines, 'ensemble_size', ens_size)   # under method:, same helper works
if ens_noise:
    lines = set_dims(lines, 'ensemble_noise', ens_noise)

open(params, 'w').write('\n'.join(lines) + '\n')
print(f"  name       : {name}")
print(f"  checkpoint : {ckpt}")
print(f"  config     : {cfg}")
print(f"  input_data : {inp or '(unchanged)'}")
PY

echo "---- resulting active params (sanity) ----"
grep -nE '^\s+(name|checkpoint|config|input_data|patch|stride|ensemble_)' "$PARAMS" | grep -v '#'

# =============================================================================
# 5. INFERENCE
# =============================================================================
echo "======== [5/6] dvc repro ========"
cd "$INFER_DIR"
# still in the training env, which has dvc + torch
# already in the training env in training
# export MKL_INTERFACE_LAYER="" #Chatgpt suggested this to solve this bug: ligne 1: MKL_INTERFACE_LAYER : variable sans liaison
# conda activate "$TRAIN_ENV"
time dvc repro

# the output file dvc wrote (name + the -n{size}-std{noise} suffix inferring.py adds)
ES=${ENS_SIZE:-10}; EN=${ENS_NOISE:-0.2}
OUT_NC="$INFER_DIR/outputs/${NAME}-n${ES}-std${EN}.nc"
echo "inference output: $OUT_NC"
[[ ! -f "$OUT_NC" ]] && echo "WARN: expected $OUT_NC not found — check the -n/-std suffix"

# =============================================================================
# 5.5  CREATE the eval notebook from a template if it doesn't exist yet
# =============================================================================
if [[ $DO_EVAL -eq 1 && -n "$NOTEBOOK" ]]; then
  NB="$NB_DIR/$NOTEBOOK"
  if [[ ! -f "$NB" ]]; then
    echo "======== [5.5] notebook missing -> create from template ========"
    # default template = your existing UNet3D eval notebook
    TEMPLATE="${TEMPLATE:-evaluation_2024_without_lwe_7nadirsNswot_UNet3DTransformer_Time30.ipynb}"
    TPL="$NB_DIR/$TEMPLATE"
    [[ ! -f "$TPL" ]] && { echo "ERROR: template notebook not found: $TPL"; exit 1; }
    cp "$TPL" "$NB"
    echo "created $NB  (copied from $TEMPLATE)"
  else
    echo "notebook already exists, reusing: $NB"
  fi
fi

# =============================================================================
# 6. EVALUATE
# =============================================================================
if [[ $DO_EVAL -eq 1 && -n "$NOTEBOOK" ]]; then
  echo "======== [6/6] evaluate notebook ========"
  NB="$NB_DIR/$NOTEBOOK"

  # set method_name + method_path in the (possibly just-created) notebook
  python - "$NB" "$NAME" "$OUT_NC" <<'PY'
import sys, json, re
nb_path, name, out_nc = sys.argv[1:4]
nb = json.load(open(nb_path))
patched = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "method_name" in src and "method_path" in src:
        # replace the whole method_path assignment (handles the multi-line
        # `method_path = ( '...' )` form as well as a single-line one)
        src = re.sub(r"method_name\s*=\s*.*", f"method_name = {name!r}", src)
        src = re.sub(r"method_path\s*=\s*\([\s\S]*?\)",
                     f"method_path = {out_nc!r}", src, count=1)
        src = re.sub(r"method_path\s*=\s*(['\"]).*?\1",
                     f"method_path = {out_nc!r}", src, count=1)
        cell["source"] = src.splitlines(keepends=True)
        patched = True
        break
if not patched:
    sys.exit("ERROR: no cell with method_name/method_path found in the notebook")
json.dump(nb, open(nb_path, "w"), indent=1)
print(f"  patched {nb_path}")
print(f"    method_name = {name}")
print(f"    method_path = {out_nc}")
PY

  cd "$NB_DIR"
  conda activate "$EVAL_ENV"
  time jupyter nbconvert --to notebook --execute --inplace "$NOTEBOOK"
  echo "Done. Scores are in the executed notebook: $NB"
fi

echo "======== PIPELINE COMPLETE ========"