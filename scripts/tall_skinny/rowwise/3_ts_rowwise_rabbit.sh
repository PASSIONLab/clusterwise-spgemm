#!/bin/bash

#OpenMP settings:
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

## Short list of datasets
echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: webbase-1M-RabbitOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/webbase-1M/webbase-1M.mtx" "${RABBIT_DATA_PATH}/webbase-1M.rabbitorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: patents_main-RabbitOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/patents_main/patents_main.mtx" "${RABBIT_DATA_PATH}/patents_main.rabbitorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: AS365-RabbitOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/AS365/AS365.mtx" "${RABBIT_DATA_PATH}/AS365.rabbitorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: com-LiveJournal-RabbitOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/com-LiveJournal/com-LiveJournal.mtx" "${RABBIT_DATA_PATH}/com-LiveJournal.rabbitorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: europe_osm-RabbitOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/europe_osm/europe_osm.mtx" "${RABBIT_DATA_PATH}/europe_osm.rabbitorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: GAP-road-RabbitOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/GAP-road/GAP-road.mtx" "${RABBIT_DATA_PATH}/GAP-road.rabbitorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: kkt_power-RabbitOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/kkt_power/kkt_power.mtx" "${RABBIT_DATA_PATH}/kkt_power.rabbitorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: M6-RabbitOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/M6/M6.mtx" "${RABBIT_DATA_PATH}/M6.rabbitorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: NLR-RabbitOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/NLR/NLR.mtx" "${RABBIT_DATA_PATH}/NLR.rabbitorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: wikipedia-20070206-RabbitOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/wikipedia-20070206/wikipedia-20070206.mtx" "${RABBIT_DATA_PATH}/wikipedia-20070206.rabbitorder" "${TS_DATA_PATH}"
