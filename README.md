# 3D-RS-PointCloudCertifying
Research done as VSRP in King Abullah's University of Science and Technology by Gabriel Pérez Santamaría
executing training
python3 Train.py \
--experiment_name defaultTry1 \
--dataset modelnet10 \
--model pointnet \
--aug_method nominal

python3 Train.py --experiment_name defaultTry1 --dataset modelnet10 --model pointnet --aug_method nominal --batch_sz 50 --sampled_points 4096 --epochs 20