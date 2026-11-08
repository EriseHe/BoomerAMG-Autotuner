#!/bin/bash
# Proposed EC2 user-data only. Never run on the Mac or an existing experiment host.
set -euo pipefail
umask 077
[[ $(uname -s) == Linux && $EUID == 0 ]] || exit 64
# A fresh EC2 cloud-init runs as Linux root. From this point, even malformed
# settings, failed mkdir/log setup, or failed timer registration request stop.
stop_on_early_exit() {
    rc=$?
    # Some shells report zero for an expansion failure inside an EXIT trap.
    # Reaching this fallback always means initialization never completed.
    [[ $rc != 0 ]] || rc=64
    trap - EXIT
    set +e
    /sbin/shutdown -h now
    exit "$rc"
}
trap stop_on_early_exit EXIT
HARD_STOP_SECONDS=28800
if [[ ${RLTUNE_SERIAL_PILOT:-0} == 1 ]]; then
    HARD_STOP_SECONDS=${RLTUNE_PILOT_WALL_SECONDS:-28800}
    [[ $HARD_STOP_SECONDS =~ ^[0-9]+$ ]] && (( HARD_STOP_SECONDS > 0 && HARD_STOP_SECONDS <= 28800 )) || exit 64
elif [[ ${RLTUNE_TIMEOUT:-} =~ ^[0-9]+$ ]]; then
    HARD_STOP_SECONDS=$((RLTUNE_TIMEOUT + 7800))
fi
# The service manager owns this timer, independently of cloud-init/SSH. A
# timeout marker is best effort and bounded; no sync/upload gates shutdown.
systemd-run --unit=rltune-hard-stop --on-active="${HARD_STOP_SECONDS}s" /bin/sh -c \
    '/usr/bin/timeout -k 1s 2s /bin/sh -c "mkdir -p /opt/rltune-singleton/job; date -u > /opt/rltune-singleton/job/HARD_TIMEOUT.txt" || :; exec /sbin/shutdown -h now'
: "${RLTUNE_BUNDLE_URI:?}" "${RLTUNE_BUNDLE_SHA256:?}" "${RLTUNE_OUTPUT_URI:?}"
: "${RLTUNE_JOB:?}" "${RLTUNE_MODE:?}" "${RLTUNE_TIMEOUT:?}" "${AWS_DEFAULT_REGION:?}"
[[ $RLTUNE_TIMEOUT =~ ^[0-9]+$ ]] || exit 64
WORK=/opt/rltune-singleton
export RLTUNE_BOOT_STARTED_UNIX="$(date +%s)"
mkdir "$WORK"  # Fail closed on an already-used instance; no resume or overwrite.
exec > >(tee -a "$WORK/bootstrap.log") 2>&1
finish() {
    rc=$?
    trap - EXIT
    set +e
    mkdir -p "$WORK/job"
    if [[ -d "$WORK/build-evidence" ]]; then cp -R "$WORK/build-evidence" "$WORK/job/build-evidence"; fi
    cp "$WORK/bootstrap.log" "$WORK/job/bootstrap.log"
    printf '{"bootstrap_exit_code":%d,"upload_confirmed":false}\n' "$rc" > "$WORK/job/bootstrap_status.json"
    if command -v aws >/dev/null; then
        # Do not follow run/source symlinks or package the environment/history.
        python3 - "$WORK" <<'PY'
import pathlib, sys, tarfile
root = pathlib.Path(sys.argv[1])
with tarfile.open(root/'results.tgz', 'w:gz', dereference=False) as tar:
    for path in sorted((root/'job').rglob('*')):
        rel = path.relative_to(root/'job')
        if path.is_file() and not path.is_symlink() and 'source' not in rel.parts:
            tar.add(path, arcname=str(rel), recursive=False)
PY
        if [[ -f "$WORK/results.tgz" ]]; then
            (cd "$WORK" && sha256sum results.tgz > results.tgz.sha256)
            for attempt in 1 2 3; do
                if timeout 600 aws s3 cp "$WORK/results.tgz" "$RLTUNE_OUTPUT_URI/results.tgz" --only-show-errors &&
                   timeout 600 aws s3 cp "$WORK/results.tgz.sha256" "$RLTUNE_OUTPUT_URI/results.tgz.sha256" --only-show-errors &&
                   timeout 600 aws s3 cp "$RLTUNE_OUTPUT_URI/results.tgz" "$WORK/returned.tgz" --only-show-errors &&
                   cmp -s "$WORK/results.tgz" "$WORK/returned.tgz"; then
                    printf '{"bootstrap_exit_code":%d,"upload_confirmed":true}\n' "$rc" > "$WORK/upload_receipt.json"
                    timeout 60 aws s3 cp "$WORK/upload_receipt.json" "$RLTUNE_OUTPUT_URI/upload_receipt.json" --only-show-errors
                    break
                fi
            done
        fi
    fi
    # Stop, never terminate: preserve EBS even if install or upload failed.
    # The EC2 request explicitly sets instance-initiated shutdown behavior=stop.
    /sbin/shutdown -h now
    exit "$rc"
}
trap finish EXIT
export DEBIAN_FRONTEND=noninteractive
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
timeout 1800 bash -c 'apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential cmake libopenmpi-dev openmpi-bin awscli time ca-certificates'
timeout 600 aws s3 cp "$RLTUNE_BUNDLE_URI" "$WORK/source.tgz" --only-show-errors
printf '%s  %s\n' "$RLTUNE_BUNDLE_SHA256" "$WORK/source.tgz" | sha256sum -c -
mkdir "$WORK/source"
python3 - "$WORK" <<'PY'
import pathlib, sys, tarfile
root = pathlib.Path(sys.argv[1])
with tarfile.open(root/'source.tgz', 'r:gz') as tar:
    for member in tar.getmembers():
        p = pathlib.PurePosixPath(member.name)
        if not member.isfile() or p.is_absolute() or '..' in p.parts:
            raise ValueError('Unsafe source archive member')
    tar.extractall(root/'source')
PY
cd "$WORK/source"
python3 deployment/aws/campaign.py verify-source
if [[ ${RLTUNE_SERIAL_PILOT:-0} == 1 ]]; then
    printf '%s' "$RLTUNE_PILOT_BUDGET_B64" | base64 -d > "$WORK/pilot-budget.json"
    python3 deployment/aws/serial_pilot.py validate-budget --budget-file "$WORK/pilot-budget.json"
fi
python3 -m venv "$WORK/venv"
export PATH="$WORK/venv/bin:$PATH"
timeout 1200 python -m pip install 'setuptools>=68' wheel
timeout 1800 env MPICC=mpicc python -m pip install --no-binary=mpi4py -e .
timeout 1800 make -C hypre JOBS=2
mkdir "$WORK/build-evidence"
dpkg-query -W python3 cmake libopenmpi-dev openmpi-bin awscli time > "$WORK/build-evidence/packages.txt"
mpicc --showme > "$WORK/build-evidence/mpicc.txt"
cp hypre/build/CMakeCache.txt "$WORK/build-evidence/CMakeCache.txt"
# Root only bootstraps. Run MPI/experiments as an ordinary dedicated user.
useradd --system --create-home --shell /bin/bash rltune-worker
chown -R rltune-worker:rltune-worker "$WORK"
if [[ ${RLTUNE_SERIAL_PILOT:-0} == 1 ]]; then
    runuser -u rltune-worker -- env PATH="$PATH" RLTUNE_BOOT_STARTED_UNIX="$RLTUNE_BOOT_STARTED_UNIX" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        python deployment/aws/serial_pilot.py run --budget-file "$WORK/pilot-budget.json" --output "$WORK/job"
else
    runuser -u rltune-worker -- env PATH="$PATH" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    python deployment/aws/campaign.py run --job "$RLTUNE_JOB" --mode "$RLTUNE_MODE" \
    --timeout-seconds "$RLTUNE_TIMEOUT" --output "$WORK/job"
fi
