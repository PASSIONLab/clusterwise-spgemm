#!/bin/bash

#OpenMP settings:
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

## Short list of datasets
echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: webbase-1M-DegreeOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/webbase-1M/webbase-1M.mtx" "${DEGREE_DATA_PATH}/webbase-1M.degreeorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: patents_main-DegreeOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/patents_main/patents_main.mtx" "${DEGREE_DATA_PATH}/patents_main.degreeorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: AS365-DegreeOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/AS365/AS365.mtx" "${DEGREE_DATA_PATH}/AS365.degreeorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: com-LiveJournal-DegreeOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/com-LiveJournal/com-LiveJournal.mtx" "${DEGREE_DATA_PATH}/com-LiveJournal.degreeorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: europe_osm-DegreeOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/europe_osm/europe_osm.mtx" "${DEGREE_DATA_PATH}/europe_osm.degreeorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: GAP-road-DegreeOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/GAP-road/GAP-road.mtx" "${DEGREE_DATA_PATH}/GAP-road.degreeorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: kkt_power-DegreeOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/kkt_power/kkt_power.mtx" "${DEGREE_DATA_PATH}/kkt_power.degreeorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: M6-DegreeOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/M6/M6.mtx" "${DEGREE_DATA_PATH}/M6.degreeorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: NLR-DegreeOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/NLR/NLR.mtx" "${DEGREE_DATA_PATH}/NLR.degreeorder" "${TS_DATA_PATH}"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~<DATASET: wikipedia-20070206-DegreeOrder>~~~~~~~~~~~~~~~~~~~~~~~~~"
srun "${CLUSTERWISE_ROOT}/bin/tsReorderedRowSpGEMM_hw" text "${DATA_PATH}/wikipedia-20070206/wikipedia-20070206.mtx" "${DEGREE_DATA_PATH}/wikipedia-20070206.degreeorder" "${TS_DATA_PATH}"
