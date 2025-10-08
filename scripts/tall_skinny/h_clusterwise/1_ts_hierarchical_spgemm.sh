#!/bin/bash

#OpenMP settings:
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

ts_short_datasets=(
  "webbase-1M" "patents_main" "AS365" "com-LiveJournal" "europe_osm"
  "GAP-road" "kkt_power" "M6" "NLR" "wikipedia-20070206"
)

# Common path prefixes
BIN_PATH="${CLUSTERWISE_ROOT}/bin/tsHierarchicalClusterSpGEMM_hw"

# Function that takes dataset names as arguments
run_on_datasets() {
  local datasets=("$@")
  for dataset in "${datasets[@]}"; do
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: ${dataset}>~~~~~~~~~~~~~~~~~~~~~~~~~"
    srun $BIN_PATH text \
      "${DATA_PATH}/${dataset}/${dataset}.mtx" \
      "${TS_DATA_PATH}" \
      "${CLOSE_PAIR_DATA_PATH}/${dataset}.mtx"
    echo ""
  done
}

run_on_datasets "${ts_short_datasets[@]}"
