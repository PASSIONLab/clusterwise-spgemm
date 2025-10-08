#!/bin/bash

#OpenMP settings:
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

## Short list of datasets
echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: webbase-1M-RCMOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/webbase-1M/webbase-1M.mtx" "${RCM_DATA_PATH}/webbase-1M.rcmorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: patents_main-RCMOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/patents_main/patents_main.mtx" "${RCM_DATA_PATH}/patents_main.rcmorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: AS365-RCMOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/AS365/AS365.mtx" "${RCM_DATA_PATH}/AS365.rcmorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: com-LiveJournal-RCMOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/com-LiveJournal/com-LiveJournal.mtx" "${RCM_DATA_PATH}/com-LiveJournal.rcmorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: europe_osm-RCMOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/europe_osm/europe_osm.mtx" "${RCM_DATA_PATH}/europe_osm.rcmorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: GAP-road-RCMOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/GAP-road/GAP-road.mtx" "${RCM_DATA_PATH}/GAP-road.rcmorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: kkt_power-RCMOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/kkt_power/kkt_power.mtx" "${RCM_DATA_PATH}/kkt_power.rcmorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: M6-RCMOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/M6/M6.mtx" "${RCM_DATA_PATH}/M6.rcmorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: NLR-RCMOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/NLR/NLR.mtx" "${RCM_DATA_PATH}/NLR.rcmorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: wikipedia-20070206-RCMOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/wikipedia-20070206/wikipedia-20070206.mtx" "${RCM_DATA_PATH}/wikipedia-20070206.rcmorder" "${TS_DATA_PATH}"
