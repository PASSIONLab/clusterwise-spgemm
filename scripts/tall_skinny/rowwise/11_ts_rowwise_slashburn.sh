#!/bin/bash

#OpenMP settings:
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

## Short list of datasets
echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: webbase-1M-SlushburnOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/webbase-1M/webbase-1M.mtx" "${SLASHBURN_DATA_PATH}/webbase-1M.slashburnorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: patents_main-SlushburnOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/patents_main/patents_main.mtx" "${SLASHBURN_DATA_PATH}/patents_main.slashburnorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: AS365-SlushburnOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/AS365/AS365.mtx" "${SLASHBURN_DATA_PATH}/AS365.slashburnorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: com-LiveJournal-SlushburnOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/com-LiveJournal/com-LiveJournal.mtx" "${SLASHBURN_DATA_PATH}/com-LiveJournal.slashburnorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: europe_osm-SlushburnOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/europe_osm/europe_osm.mtx" "${SLASHBURN_DATA_PATH}/europe_osm.slashburnorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: GAP-road-SlushburnOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/GAP-road/GAP-road.mtx" "${SLASHBURN_DATA_PATH}/GAP-road.slashburnorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: kkt_power-SlushburnOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/kkt_power/kkt_power.mtx" "${SLASHBURN_DATA_PATH}/kkt_power.slashburnorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: M6-SlushburnOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/M6/M6.mtx" "${SLASHBURN_DATA_PATH}/M6.slashburnorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: NLR-SlushburnOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/NLR/NLR.mtx" "${SLASHBURN_DATA_PATH}/NLR.slashburnorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: wikipedia-20070206-SlushburnOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/wikipedia-20070206/wikipedia-20070206.mtx" "${SLASHBURN_DATA_PATH}/wikipedia-20070206.slashburnorder" "${TS_DATA_PATH}"
