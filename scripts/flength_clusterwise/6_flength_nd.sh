#!/bin/bash

#OpenMP settings:
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# Dataset lists as strings
initial_datasets=(
  "2cubes_sphere" "cage12" "cop20k_A" "m133-b3" "mac_econ_fwd500" "mario002"
  "mc2depi" "poisson3Da" "pwtk" "webbase-1M" "cage15" "cant" "conf5_4-8x8-05"
  "consph" "delaunay_n24" "filter3D" "hood" "majorbasis" "mono_500Hz" "offshore"
  "patents_main" "pdb1HYS" "rma10" "scircuit" "shipsec1" "wb-edu"
)

ispass_datasets=(
  "333SP" "adaptive" "af_shell10" "AS365" "as-Skitter" "channel-500x100x100-b050"
  "cit-Patents" "com-LiveJournal" "CurlCurl_4" "europe_osm" "G3_circuit" "GAP-road"
  "hugebubbles-00020" "hugetrace-00020" "hugetric-00010" "HV15R" "kkt_power" "M6"
  "nlpkkt240" "NLR" "packing-500x100x100-b050" "patents" "Queen_4147" "rajat31"
  "rgg_n_2_24_s0" "soc-LiveJournal1" "stokes" "sx-stackoverflow" "venturiLevel3"
  "wikipedia-20070206" "wiki-Talk" "wiki-topcats"
)

new_datasets=(
  "mip1" "mouse_gene" "BenElechi1" "atmosmodl" "pkustk14" "great-britain_osm"
  "hugebubbles-00000" "hugebubbles-00010" "hugetrace-00000" "hugetrace-00010"
  "hugetrace-00010" "hugetric-00000" "hugetric-00020" "italy_osm"
  "kron_g500-logn17" "kron_g500-logn18" "kron_g500-logn19" "kron_g500-logn20"
  "rgg_n_2_20_s0" "rgg_n_2_21_s0" "rgg_n_2_22_s0" "rgg_n_2_23_s0" "road_central"
  "road_usa" "asia_osm" "coPapersCiteseer" "coPapersDBLP" "delaunay_n21"
  "delaunay_n22" "delaunay_n23" "germany_osm" "halfb" "dielFilterV3real"
  "kmer_V1r" "bmw3_2" "audikw_1" "Hardesty3" "JP" "msdoor" "relat9" "F1"
  "uk-2002" "fem_hifreq_circuit" "bundle_adj" "mycielskian18" "nd24k" "torso1"
  "bone010" "Ga41As41H72" "gearbox" "thermal2" "higgs-twitter"
  "TSOPF_RS_b2383_c1"
)

# Common path prefixes
BIN_PATH="${CLUSTERWISE_ROOT}/bin/ReorderedFlengthClusterSpGEMM_hw"

# Function that takes dataset names as arguments
run_on_datasets() {
  local datasets=("$@")
  for dataset in "${datasets[@]}"; do
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: ${dataset}-NDOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
    srun $BIN_PATH text \
      "${DATA_PATH}/${dataset}/${dataset}.mtx" \
      "${ND_DATA_PATH}/${dataset}.ndorder"
    echo ""
  done
}

# Execute the benchmark:
run_on_datasets "${initial_datasets[@]}"
run_on_datasets "${ispass_datasets[@]}"
run_on_datasets "${new_datasets[@]}"
