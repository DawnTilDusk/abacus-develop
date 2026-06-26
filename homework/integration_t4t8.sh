#!/bin/bash
# ============================================================================
# integration_t4t8.sh — 任务4&8 集成测试 (轻量版)
#
# 问题: 原始 Fe_nspin2 示例 (ecutwfc=60, kpt 4x4x4) 计算量过大，
#       OMP task 版本 4 MPI 进程 × (4 workers+1) = 20 外层线程/进程，
#       与内层 ABACUS OMP for 叠加后超订严重。
#
# 修复:
#   1. 自定义极简 INPUT (ecutwfc=20, kpt 1×1×1, scf_nmax=5)
#      → 每次运行 ~2-5 秒
#   2. 每方案仅 2 个 MPI 进程
#   3. OMP task 版本: OMP_NUM_THREADS=5 (4 workers + 1 主线程)
#
# 用法:
#   chmod +x integration_t4t8.sh
#   ./integration_t4t8.sh
#
# 前提: build/ 和 build_omp/ 均已编译
# ============================================================================

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
N="${ROOT}"
EXE_THREAD="${N}/build/abacus_basic_para"
EXE_OMP="${N}/build_omp/abacus_basic_para"

MPI_PROCS=2; REPEAT=2
RESULTS="${N}/results_t4t8"
rm -rf "${RESULTS}"; mkdir -p "${RESULTS}"/{logs,out_thread,out_omp}
L="${RESULTS}/logs"; T="${RESULTS}/out_thread"; O="${RESULTS}/out_omp"

echo "=========================================="
echo " Task 4&8 Integration Test (lightweight)"
echo " MPI=${MPI_PROCS}  Repeat=${REPEAT}"
echo " ecutwfc=20 kpt=1x1x1 scf_nmax=5"
echo "=========================================="

for e in "${EXE_THREAD}" "${EXE_OMP}"; do
    [ ! -f "${e}" ] && echo "ERROR: ${e} not found" && exit 1
done

BASE_CASE="${N}/examples/08_charge_density/03_pw_Fe_nspin2"

# 生成极简 INPUT/KPT（保留 STRU、赝势路径不变）
make_workdir() {
    local w=$1
    rm -rf "${w}" && cp -r "${BASE_CASE}" "${w}"
    cat > "${w}/INPUT" << 'EOF'
INPUT_PARAMETERS
pseudo_dir           ../../../tests/PP_ORB
symmetry             1
basis_type           pw
calculation          scf
nspin                2
ecutwfc              20
scf_thr              1.0e-4
scf_nmax             5
out_chg              1
ks_solver            cg
smearing_method      gaussian
smearing_sigma       0.015
mixing_type          broyden
mixing_beta          0.5
EOF
    cat > "${w}/KPT" << 'EOF'
K_POINTS
0
Gamma
1 1 1  0 0 0
EOF
}

run_one() {
    local exe=$1 tag=$2 rid=$3 env=$4
    local w="/tmp/abacus_${tag}_${rid}"
    make_workdir "${w}"
    cd "${w}"
    echo "  [${tag} run ${rid}] starting..."
    bash -c "${env} mpirun --oversubscribe -np ${MPI_PROCS} ${exe}" \
        > "${L}/${tag}_r${rid}.log" 2>&1 || true
    echo "  [${tag} run ${rid}] done (exit=$?)"
    local od="${RESULTS}/out_${tag}/${tag}_r${rid}"
    mkdir -p "${od}"
    [ -d "OUT.ABACUS" ] && cp OUT.ABACUS/* "${od}/" 2>/dev/null || true
    cd "${N}" && rm -rf "${w}"
}

echo ""; echo "--- Phase 1: Run ---"
for r in $(seq 1 ${REPEAT}); do
    run_one "${EXE_THREAD}" "thread" "${r}" ""
done
for r in $(seq 1 ${REPEAT}); do
    run_one "${EXE_OMP}" "omp" "${r}" "OMP_NUM_THREADS=5 OMP_MAX_ACTIVE_LEVELS=2"
done

echo ""; echo "--- Phase 2: Correctness (bit-identical) ---"
ps=0; fl=0
for r1 in $(seq 1 ${REPEAT}); do for r2 in $(seq 1 ${REPEAT}); do
    for s in chgs1 chgs2; do
        f1="${T}/thread_r${r1}/${s}.cube"
        f2="${O}/omp_r${r2}/${s}.cube"
        if [ -f "${f1}" ] && [ -f "${f2}" ]; then
            diff <(xxd "${f1}") <(xxd "${f2}") >/dev/null 2>&1 \
                && ps=$((ps+1)) \
                || { echo "  MISMATCH: ${s}.cube r${r1}-r${r2}"; fl=$((fl+1)); }
        else
            echo "  MISSING: ${s}.cube r${r1}-r${r2}"; fl=$((fl+1))
        fi
    done
done; done
echo "  PASS=${ps} FAIL=${fl}"

# 文件大小
echo ""; echo "  Size check:"
for s in chgs1 chgs2; do
    f1="${T}/thread_r1/${s}.cube"; f2="${O}/omp_r1/${s}.cube"
    if [ -f "${f1}" ] && [ -f "${f2}" ]; then
        sz1=$(wc -c < "${f1}"); sz2=$(wc -c < "${f2}")
        m="OK"; [ "${sz1}" != "${sz2}" ] && m="MISMATCH"
        echo "    ${s}.cube  thread=${sz1} omp=${sz2} [${m}]"
    fi
done

echo ""; echo "--- Phase 3: Timer ---"
parse_tm(){ grep '\[TIMER\]' "$1" 2>/dev/null|grep "$2"|grep -oP '\d+\.?\d*\s*s'|head -1|grep -oP '\d+\.?\d*'||echo "N/A"; }
printf "  %-30s %12s %12s %10s\n" "Timer" "std::thread" "OMP task" "Speedup"
for tm in "init_rho_async_binary" "init_rho_recip2real" "write_vdata_palgrid"; do
    st=0;ct=0;so=0;co=0
    for r in $(seq 1 ${REPEAT}); do
        vt=$(parse_tm "${L}/thread_r${r}.log" "${tm}")
        vo=$(parse_tm "${L}/omp_r${r}.log" "${tm}")
        [ "${vt}" != "N/A" ]&&{ st=$(echo "${st}+${vt}"|bc -l);ct=$((ct+1));}
        [ "${vo}" != "N/A" ]&&{ so=$(echo "${so}+${vo}"|bc -l);co=$((co+1));}
    done
    if [ ${ct} -gt 0 ]&&[ ${co} -gt 0 ]; then
        at=$(echo "scale=4;${st}/${ct}"|bc -l)
        ao=$(echo "scale=4;${so}/${co}"|bc -l)
        sp=$(echo "scale=2;${at}/${ao}"|bc -l)
        printf "  %-30s %12s %12s %10s\n" "${tm}" "${at}s" "${ao}s" "${sp}x"
    fi
done

echo ""; echo "Done. Logs: ${L}/"
