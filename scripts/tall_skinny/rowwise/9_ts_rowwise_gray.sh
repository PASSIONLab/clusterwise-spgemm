#!/bin/bash

#OpenMP settings:
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Dataset lists as strings
ts_short_datasets=(
  "webbase-1M" "patents_main" "AS365" "com-LiveJournal" "europe_osm"
  "GAP-road" "kkt_power" "M6" "NLR" "wikipedia-20070206"
)

# Common path prefixes
BIN_PATH="${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw"

# Function that takes dataset names as arguments
run_on_datasets() {
  local datasets=("$@")
  for dataset in "${datasets[@]}"; do
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: ${dataset}-GrayOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
    srun $BIN_PATH text \
      "${DATA_PATH}/${dataset}/${dataset}.mtx" \
      "${GRAY_DATA_PATH}/${dataset}.grayorder" \
      "${TS_DATA_PATH}"
    echo ""
  done
}

# Execute the benchmark:
run_on_datasets "${ts_short_datasets[@]}"
