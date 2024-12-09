#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1

#MODE=$1
EXPNAME=$1

#OUT_DIR=$3

DATAROOT=./data
OUT_DIR=./output

scenes='esslingen/hse_hinterhof/2024-05-24_2/d435i esslingen/hse_hinterhof/2024-05-24_2/pi_cam_02'
scenes='kwald/drosselweg/flaeche1/2024-01-13/d435i'

echo "Start evaluating on TUM dataset..."

for sc in ${scenes};
do
  echo Running on $sc ...
  sensor=$(basename $sc)
  if [[ $sensor == "d435i" ]]
  then
    mode="rgbd"
  fi
  if [[ $sensor == "pi_cam_02" ]]
  then
    mode="mono"
  fi

  echo Using sensor $sensor
  python run.py configs/TUM_RGBD/tum_${sensor}.yaml \
                --mode $mode --output ${OUT_DIR}/${sc}/$EXPNAME \
                --only_tracking \
                --input_folder ${DATAROOT}/${sc}
  # if [[ $MODE == "mono" ]]
  # then
  #   python run.py configs/TUM_RGBD/scene${sc}_mono.yaml --device cuda:0 --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME --only_tracking
  # else
  #   python run.py configs/TUM_RGBD/scene${sc}.yaml --device cuda:0 --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME --only_tracking
  # fi
  echo $sc done!
done

echo Results for all scenes are:

for sc in ${scenes}
do
  echo
  echo For ${sc}:
  cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_traj.txt
  echo
  # cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_mesh.txt
done

echo All Done!
