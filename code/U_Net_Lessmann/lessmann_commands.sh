
#!/bin/bash
# run the data preprocessor

read -p "prepare xVertSeg dataset? (y/n) " REPLY
echo   # (optional) move to a new line
if [ $REPLY = "y" ]
then
    python3 /root/space/U_Net_Lessmann/data/prepare_xVertSeg.py \
        --source /root/space/data/ \
        --output /root/space/temp/dataset/
fi

read -p "run preprocessing? (y/n) " REPLY
echo   # (optional) move to a new line
if [ $REPLY = "y" ]
then
    python3 /root/space/U_Net_Lessmann/data/preprocessing.py \
        --dataset /root/space/temp/dataset \
        --output_isotropic /root/space/temp/iso/ \
        --output_crop /root/space/temp/crop/
    echo "Pre-processing done."
fi

read -p "run visualisation? (y/n) " REPLY
echo   # (optional) move to a new line
if [ $REPLY = "y" ]
then
    python3 /root/space/U_Net_Lessmann/data/visualized_preprocessed_images.py \
        --dataset /root/space/temp/iso
    echo "Visualisation of pre-processed dataset done (iso)"
    python3 /root/space/U_Net_Lessmann/data/visualized_preprocessed_images.py \
        --dataset /root/space/temp/crop
    echo "Visualisation of pre-processed dataset done (crop)"
fi

read -p "start training? (y/n) " REPLY
echo   # (optional) move to a new line
if [ $REPLY = "y" ]
then
    # rm -rf /root/space/U_Net_Lessmann/tensorboard
    tree
    python3 /root/space/U_Net_Lessmann/train.py \
        --dataset /root/space/temp/iso/ \
        --weight /root/space/temp/weight/ \
        --checkpoints /root/space/temp/checkpoints \
        --tensorboard /root/space/U_Net_Lessmann/tensorboard \
        --lr 1e-4 \
        --iterations 100000 \
        --log_interval 1000 \
        --eval_iters 250 \
        --resume True
fi