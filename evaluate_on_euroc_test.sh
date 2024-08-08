#!/bin/bash

MODE='stereo'       # 'mono', 'stereo', 'rgbd'
EXPNAME=$1

OUT_DIR='./output'
DATAROOT='./data'

SENSOR='t265'

scenes='MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult'
scenes='MH_01_easy'
scenes='esslingen/hse_dach/2023-07-20/euroc'

echo "Start evaluating on EuRoC dataset..."

for sc in ${scenes};
do
  echo Running on $sc ...
#  python run.py configs/EuRoC/euroc_mono.yaml \
   python run.py configs/EuRoC/euroc_${SENSOR}.yaml \
                --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME \
                --only_tracking \
                --input_folder ${DATAROOT}/${sc}
  echo $sc done!
done

echo Results for all scenes are:

for sc in ${scenes}
do
  echo
  echo For ${sc}:
  cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_traj.txt
  echo
  cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_mesh.txt
done

echo All Done!
