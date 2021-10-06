#!/bin/bash
valid=true
count=1
sigma=0.05
while [ $valid ]
do
echo $count
echo $sigma

echo pointnet2Shearing0$sigma
CUDA_VISIBLE_DEVICES=1 python3 Certify.py --model pointnet2 --base_classifier_path Pointent2andDGCNN/output/train/pointnetBaseline/FinalModel.pth.tar --sigma $sigma --certify_method shearing --experiment_name pointnet2Shearing0$sigma
echo pointnet2Tapering0$sigma
CUDA_VISIBLE_DEVICES=1 python3 Certify.py --model pointnet2 --base_classifier_path Pointent2andDGCNN/output/train/pointnetBaseline/FinalModel.pth.tar --sigma $sigma --certify_method tapering --experiment_name pointnet2Tapering0$sigma

echo dgcnnShearing0$sigma
CUDA_VISIBLE_DEVICES=1 python3 Certify.py --model dgcnn --base_classifier_path Pointent2andDGCNN/output/train/dgcnnBaseline/FinalModel.pth.tar --sigma $sigma --certify_method shearing --experiment_name dgcnnShearing0$sigma
echo dgcnnTapering0$sigma
CUDA_VISIBLE_DEVICES=1 python3 Certify.py --model dgcnn --base_classifier_path Pointent2andDGCNN/output/train/dgcnnBaseline/FinalModel.pth.tar --sigma $sigma --certify_method tapering --experiment_name dgcnnTapering0$sigma


if [ $count -eq 20 ];
then
break
fi

sigma=`echo "scale=4; $sigma+0.05" | bc`
((count++))
done