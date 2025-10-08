#!/bin/bash

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: webbase-1M>~~~~~~~~~~~~~~~~~~~~~~~~~"
OMP_NUM_THREADS=8 srun -t 30 -N 4 -n 64 -c 16 --cpu-bind=cores -q interactive -C cpu  "${COMBBLAS_ROOT}/_build/Applications/betwcent" "${DATA_PATH}/webbase-1M/webbase-1M.mtx" 12 4096 "${TS_DATA_PATH}/webbase-1M"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: patents_main>~~~~~~~~~~~~~~~~~~~~~~~~~"
OMP_NUM_THREADS=8 srun -t 30 -N 4 -n 64 -c 16 --cpu-bind=cores -q interactive -C cpu  "${COMBBLAS_ROOT}/_build/Applications/betwcent" "${DATA_PATH}/patents_main/patents_main.mtx" 12 4096 "${TS_DATA_PATH}/patents_main"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: AS365>~~~~~~~~~~~~~~~~~~~~~~~~~"
OMP_NUM_THREADS=8 srun -t 30 -N 4 -n 64 -c 16 --cpu-bind=cores -q interactive -C cpu  "${COMBBLAS_ROOT}/_build/Applications/betwcent" "${DATA_PATH}/AS365/AS365.mtx" 12 4096 "${TS_DATA_PATH}/AS365"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: com-LiveJournal>~~~~~~~~~~~~~~~~~~~~~~~~~"
OMP_NUM_THREADS=8 srun -t 30 -N 4 -n 64 -c 16 --cpu-bind=cores -q interactive -C cpu  "${COMBBLAS_ROOT}/_build/Applications/betwcent" "${DATA_PATH}/com-LiveJournal/com-LiveJournal.mtx" 12 4096 "${TS_DATA_PATH}/com-LiveJournal"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: europe_osm>~~~~~~~~~~~~~~~~~~~~~~~~~"
OMP_NUM_THREADS=8 srun -t 30 -N 4 -n 64 -c 16 --cpu-bind=cores -q interactive -C cpu  "${COMBBLAS_ROOT}/_build/Applications/betwcent" "${DATA_PATH}/europe_osm/europe_osm.mtx" 12 4096 "${TS_DATA_PATH}/europe_osm"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: GAP-road>~~~~~~~~~~~~~~~~~~~~~~~~~"
OMP_NUM_THREADS=8 srun -t 30 -N 4 -n 64 -c 16 --cpu-bind=cores -q interactive -C cpu  "${COMBBLAS_ROOT}/_build/Applications/betwcent" "${DATA_PATH}/GAP-road/GAP-road.mtx" 12 4096 "${TS_DATA_PATH}/GAP-road"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: kkt_power>~~~~~~~~~~~~~~~~~~~~~~~~~"
OMP_NUM_THREADS=8 srun -t 30 -N 4 -n 64 -c 16 --cpu-bind=cores -q interactive -C cpu  "${COMBBLAS_ROOT}/_build/Applications/betwcent" "${DATA_PATH}/kkt_power/kkt_power.mtx" 12 4096 "${TS_DATA_PATH}/kkt_power"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: M6>~~~~~~~~~~~~~~~~~~~~~~~~~"
OMP_NUM_THREADS=8 srun -t 30 -N 4 -n 64 -c 16 --cpu-bind=cores -q interactive -C cpu  "${COMBBLAS_ROOT}/_build/Applications/betwcent" "${DATA_PATH}/M6/M6.mtx" 12 4096 "${TS_DATA_PATH}/M6"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: NLR>~~~~~~~~~~~~~~~~~~~~~~~~~"
OMP_NUM_THREADS=8 srun -t 30 -N 4 -n 64 -c 16 --cpu-bind=cores -q interactive -C cpu  "${COMBBLAS_ROOT}/_build/Applications/betwcent" "${DATA_PATH}/NLR/NLR.mtx" 12 4096 "${TS_DATA_PATH}/NLR"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: wikipedia-20070206>~~~~~~~~~~~~~~~~~~~~~~~~~"
OMP_NUM_THREADS=8 srun -t 30 -N 4 -n 64 -c 16 --cpu-bind=cores -q interactive -C cpu  "${COMBBLAS_ROOT}/_build/Applications/betwcent" "${DATA_PATH}/wikipedia-20070206/wikipedia-20070206.mtx" 12 4096 "${TS_DATA_PATH}/wikipedia-20070206"
