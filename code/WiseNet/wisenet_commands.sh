
#!/bin/bash
# run the data preprocessor

read -p "prepare xVertSeg dataset? (y/n) " REPLY
echo   # (optional) move to a new line
if [ $REPLY = "y" ]
then
    python3 /root/space/DataPreparation/prepare_xVertSeg.py \
        --source /root/space/data/ \
        --output /root/space/temp/dataset/
fi

read -p "start training? (y/n) " REPLY
echo   # (optional) move to a new line
if [ $REPLY = "y" ]
then
    tree
    python3 /root/space/WiseNet/src/main.py -e wisenet_pascal -sb /root/space/WiseNet/results -d /root/space/data -r 1
fi