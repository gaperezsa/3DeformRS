#!/bin/bash
valid=true
count=1
sigma=.20
while [ $valid ]
do
echo $count
echo $sigma

echo dgcnnTranslation0$sigma
python3 Certify.py --model dgcnn --dataset modelnet10 --base_classifier_path Pointent2andDGCNN/output/train/modelnet10dgcnnbaseline/FinalModel.pth.tar --sigma $sigma --certify_method translation --experiment_name dgcnnTranslationModelnet10_0$sigma
echo dgcnnShearing$sigma
python3 Certify.py --model dgcnn --dataset modelnet10 --base_classifier_path Pointent2andDGCNN/output/train/modelnet10dgcnnbaseline/FinalModel.pth.tar --sigma $sigma --certify_method shearing --experiment_name dgcnnShearingModelnet10_0$sigma
echo dgcnnTapering0$sigma
python3 Certify.py --model dgcnn --dataset modelnet10 --base_classifier_path Pointent2andDGCNN/output/train/modelnet10dgcnnbaseline/FinalModel.pth.tar --sigma $sigma --certify_method tapering --experiment_name dgcnnTaperingModelnet10_0$sigma

if [ $count -eq 3 ];
then
break
fi

sigma=`echo "scale=4; $sigma+0.05" | bc`
((count++))
done