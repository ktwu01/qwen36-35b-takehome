# Shared Terminal-Bench 2.0 environment settings, sourced by scripts/run_tb2.sh.
# Exam-fixed resources: 32 CPU / 48 GB per task (memory enforced by harbor's watchdog;
# CPU is not enforced by the apptainer backend, see README "Terminal-Bench environment").
BASE=${BASE:-/var/tmp/$USER}
export APPTAINER_CACHEDIR=$BASE/apptainer-cache APPTAINER_TMPDIR=$BASE/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$BASE/sif-cache" "$BASE/tb2-jobs"
HARBOR=$BASE/venvs/harbor/bin/harbor
TB2_TASKS=${TB2_TASKS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/bench/tb2/terminal-bench}
TB2_MOUNTS="[{\"type\":\"bind\",\"source\":\"$BASE/harbor-rt\",\"target\":\"/opt/harbor-rt\",\"read_only\":true},{\"type\":\"bind\",\"source\":\"/etc/resolv.conf\",\"target\":\"/etc/resolv.conf\",\"read_only\":true},{\"type\":\"bind\",\"source\":\"$BASE/harbor-rt/hosts\",\"target\":\"/etc/hosts\",\"read_only\":true}]"
TB2_ENV_ARGS=(-e singularity --ek "singularity_image_cache_dir=$BASE/sif-cache" --mounts-json "$TB2_MOUNTS"
              --override-cpus 32 --override-memory-mb 49152)
