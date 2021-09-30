#!/bin/bash
valid=true
count=1
sigma=0.05
while [ $valid ]
do
echo $count
echo $sigma

echo pointnet2Translation0$sigma
CUDA_VISIBLE_DEVICES=1 python3 Certify.py --model pointnet2 --base_classifier_path Pointent2andDGCNN/output/train/pointnetBaseline/FinalModel.pth.tar --sigma $sigma --certify_method translation --experiment_name pointnet2Translation0$sigma
echo pointnet2Rotation0$sigma
CUDA_VISIBLE_DEVICES=1 python3 Certify.py --model pointnet2 --base_classifier_path Pointent2andDGCNN/output/train/pointnetBaseline/FinalModel.pth.tar --sigma $sigma --certify_method rotation --experiment_name pointnet2Rotation0$sigma

echo dgcnnTranslation0$sigma
CUDA_VISIBLE_DEVICES=1 python3 Certify.py --model dgcnn --base_classifier_path Pointent2andDGCNN/output/train/dgcnnBaseline/FinalModel.pth.tar --sigma $sigma --certify_method translation --experiment_name dgcnnTranslation0$sigma
echo dgcnnRotation0$sigma
CUDA_VISIBLE_DEVICES=1 python3 Certify.py --model dgcnn --base_classifier_path Pointent2andDGCNN/output/train/dgcnnBaseline/FinalModel.pth.tar --sigma $sigma --certify_method rotation --experiment_name dgcnnRotation0$sigma


if [ $count -eq 20 ];
then
break
fi

sigma=`echo "scale=4; $sigma+0.05" | bc`
((count++))
done