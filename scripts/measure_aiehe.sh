#!/usr/bin/env bash
# Measures AIE-HE with its Encode connected to its Encrypt
# (src/ckks_encode_encrypt_aiehe): aiehe_d (planA, decomposed NTT) and aiehe_u
# (planB, undecomposed). Their time already includes the Encode. Run on the AIE machine.
# All hosts are Release builds. L is the number of data primes.
#
# usage: bash scripts/measure_aiehe.sh [logNs] [Ls]
#        defaults: "13 14 15 16 17" and "5 10 15 20 25 30 35 40 45 50"
set -euo pipefail
LOGNS=${1:-"13 14 15 16 17"}
LS=${2:-"5 10 15 20 25 30 35 40 45 50"}
# AIE runs as `make run` (100 measured after 10 warm-up).
ITERS=${ITERS:-100}
WARMUP=${WARMUP:-10}
SCALE_BITS=${SCALE_BITS:-58}

root=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$root/src/ckks_encrypt_SEAL/thirdparty/SEAL/CMakeLists.txt" ] \
  || { echo "SEAL (79ad62fb) missing: put it at src/ckks_encrypt_SEAL/thirdparty/SEAL" >&2; exit 1; }
mkdir -p "$root/results"
out=$(mktemp -d "$root/results/aiehe-$(date +%Y%m%d-%H%M%S)-XXXXXX")
echo "results: $out"
trap 'echo "failed; logs: $out" >&2' ERR

dir_of() { case $1 in aiehe_d) echo "$root/src/ckks_encrypt_planA" ;; aiehe_u) echo "$root/src/ckks_encrypt_planB" ;; esac; }

# The connected AIE-HE builds the host source with src/ckks_encode_encrypt_aiehe
# (header, CMake, SEAL). Both hosts are built only this way.
joined=$root/src/ckks_encode_encrypt_aiehe
for v in aiehe_d aiehe_u; do
  src=$(dir_of $v)
  ( cmake -S "$joined" -B "$src/_build_joined" -DAIEHE_SRC="$src/test.cpp" -DCMAKE_BUILD_TYPE=Release &&
    cmake --build "$src/_build_joined" -j 4 ) > "$out/host_$v.log" 2>&1 || { tail -n 30 "$out/host_$v.log"; exit 1; }
done

lmax=$(echo $LS | tr ' ' '\n' | sort -n | tail -1)
echo "variant,N,L,encrypt_us" > "$out/aie.csv"
# logN outermost: the xclbins are rebuilt only when logN changes.
for n in $LOGNS; do
  for v in aiehe_d aiehe_u; do
    d=$(dir_of $v)
    case $v in
      aiehe_d) targets="build_divided_ntt/final.xclbin build_fused/final.xclbin build_mac/final.xclbin" ;;
      aiehe_u) targets="build_mac/final.xclbin build_tuboi_ntt/final.xclbin build_divided_ntt/final.xclbin" ;;
    esac
    ( cd "$d" && make clean && make $targets logN=$n rns_L=5 tile_vol_pow=9 ) > "$out/build_${v}_$n.log" 2>&1 \
      || { echo "build $v logN=$n failed"; tail -n 30 "$out/build_${v}_$n.log"; exit 1; }
    run() {
      local l=$1; shift
      (cd "$d" && ./_build_joined/kernel -x build_mac/final.xclbin -i build_mac/insts.bin -k MLIR_AIE \
         --n_log $n --rns_L $l --tile_vol_pow 9 --verbosity 0 --scale_bits $SCALE_BITS "$@")
    }
    # Do not time an unverified build: check the smallest and the largest L.
    for l in 5 $lmax; do
      run $l --iters 1 --warmup 0 --benchmark 0 > "$out/verify_${v}_${n}_$l.log" 2>&1 || true
      grep -q 'PASS!' "$out/verify_${v}_${n}_$l.log" \
        || { echo "verify $v logN=$n L=$l failed"; tail -n 20 "$out/verify_${v}_${n}_$l.log"; exit 1; }
    done
    for l in $LS; do
      t=$(run $l --iters $ITERS --warmup $WARMUP --benchmark 1 | sed -n -E 's/.*Avg NPU time: ([0-9.]+).*/\1/p')
      [ -n "$t" ] || { echo "no timing for $v logN=$n L=$l"; exit 1; }
      echo "$v,$((1 << n)),$l,$t" | tee -a "$out/aie.csv"
    done
  done
done

python3 - "$out" <<'PY'
import csv, sys
from pathlib import Path
out = Path(sys.argv[1])
aie = {}
for r in csv.DictReader(open(out / 'aie.csv')):
    aie[(r['variant'], int(r['N']), int(r['L']))] = float(r['encrypt_us'])
variants = ['aiehe_d', 'aiehe_u']  # the time already includes the encode
rows = []
for (n, l) in sorted({(n, l) for (_, n, l) in aie}):
    t = [aie[(v, n, l)] for v in variants]
    rows.append([n, l] + t + [min(t)])
head = ['N', 'L'] + [f'{v}_total_us' for v in variants] + ['min_total_us']
with open(out / 'summary.csv', 'w', newline='') as f:
    csv.writer(f).writerows([head] + rows)
print(f"{'N':>7} {'L':>3} " + ' '.join(f'{v:>9}' for v in variants) + '   min  (total ms)')
for r in rows:
    print(f"{r[0]:>7} {r[1]:>3} " + ' '.join(f'{x/1000:9.2f}' for x in r[2:4]) + f'  {r[-1]/1000:7.2f}')
PY
echo "results: $out"
