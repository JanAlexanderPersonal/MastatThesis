
#!/bin/bash
# run the data preprocessor

read -p "prepare xVertSeg dataset? (y/n) " REPLY
echo   # (optional) move to a new line
if [ $REPLY = "y" ]
then
    python3 /root/space/U_Net_Lessmann/data/prepare_xVertSeg.py --source /root/space/data/ --output /root/space/temp/dataset/
fi

read -p "run preprocessing? (y/n) " REPLY
echo   # (optional) move to a new line
if [ $REPLY = "y" ]
then
    python3 /root/space/U_Net_Lessmann/data/preprocessing.py --dataset /root/space/temp/dataset --output_isotropic /root/space/temp/iso/ --output_crop /root/space/temp/crop/
    echo "Pre-processing done. Start training"
fi

read -p "start training? (y/n) " REPLY
echo   # (optional) move to a new line
if [ $REPLY = "y" ]
then
    tree
    python3 /root/space/U_Net_Lessmann/train.py --dataset /root/space/temp/iso/
fi