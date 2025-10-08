#!/bin/bash

#OpenMP settings:
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

## Short list of datasets
echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: webbase-1M-NDOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/webbase-1M/webbase-1M.mtx" "${ND_DATA_PATH}/webbase-1M.ndorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: patents_main-NDOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/patents_main/patents_main.mtx" "${ND_DATA_PATH}/patents_main.ndorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: AS365-NDOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/AS365/AS365.mtx" "${ND_DATA_PATH}/AS365.ndorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: com-LiveJournal-NDOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/com-LiveJournal/com-LiveJournal.mtx" "${ND_DATA_PATH}/com-LiveJournal.ndorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: europe_osm-NDOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/europe_osm/europe_osm.mtx" "${ND_DATA_PATH}/europe_osm.ndorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: GAP-road-NDOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/GAP-road/GAP-road.mtx" "${ND_DATA_PATH}/GAP-road.ndorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: kkt_power-NDOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/kkt_power/kkt_power.mtx" "${ND_DATA_PATH}/kkt_power.ndorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: M6-NDOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/M6/M6.mtx" "${ND_DATA_PATH}/M6.ndorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: NLR-NDOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/NLR/NLR.mtx" "${ND_DATA_PATH}/NLR.ndorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: wikipedia-20070206-NDOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/wikipedia-20070206/wikipedia-20070206.mtx" "${ND_DATA_PATH}/wikipedia-20070206.ndorder" "${TS_DATA_PATH}"
