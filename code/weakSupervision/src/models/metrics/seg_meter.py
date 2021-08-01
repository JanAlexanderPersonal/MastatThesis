import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from typing import Dict

from pprint import pformat

import logging
logger = logging.getLogger(__name__)

CLASS_NAMES = ['Background', 'L1', 'L2', 'L3', 'L4', 'L5']
SINGLE_CLASS_NAMES = ['Background', 'Vertebra']

class SegMeter:
    """Calulate segmentation metrics

    function 
        val_on_batch(model, batch)
            will predict the batch on the model and calculate the confusion matrix on the result

        get_avg_score()
            Returns the evaluation metrics on the confusion matrix
    """

    def __init__(self, split):
        self.cf = None
        self.struct_list = []
        self.split = split

    def val_on_batch(self, model, batch):
        """Construction of the confusion matrix for this batch

        Args:
            model (PyTorch model): PyTorch model
            batch (Torch tensor): Tensor to evaluate

        Remark: 
            In the original Laradji code, this was only calculated for the non-background points (ind = mask_org != 255).
            In this version, I want the confusion matrix of the complete result.
            I also tried to simplify (and correct) the calculation of the confusion matrix.
        """

        logging.debug('start validation on batch')

        # Get groud truth and prediction (as np.ndarrays)
        gt = batch["masks"].data.cpu().numpy()
        pred = model.predict_on_batch(batch)

        logger.debug(f'mask shape : {gt.shape} with type {type(gt)} and unique values {np.unique(gt)}')
        logger.debug(f'predicted mask shape : {pred.shape} with type {type(pred)} and unique values {np.unique(pred)}')



        n_classes = model.n_classes

        cf = confusion_matrix(gt.reshape(-1), pred.reshape(-1), labels=[i for i in range(n_classes)], normalize=None)

        if self.cf is None:
            self.cf = cf
        else:
            self.cf += cf

    def val_on_volume(self, gt : np.ndarray, pred : np.ndarray, n_classes : int):
        cf = confusion_matrix(gt.reshape(-1), pred.reshape(-1), labels=[i for i in range(n_classes)], normalize=None)
        if self.cf is None:
            self.cf = cf
        else:
            self.cf += cf
        logger.debug(f'New confusion matrix :\n{self.cf}')

    def calc_accuracy(self) -> np.ndarray:
        #               k - 1       
        #                ===        
        #                \    a     
        #                /     ii   
        #                ===        
        #               i = 0       
        # accuracy = ---------------
        #            k - 1          
        #            =====          
        #            \     k - 1    
        #             \     ===     
        #              \    \    a  
        #              /    /     ij
        #             /     ===     
        #            /     j = 0    
        #            =====          
        #            i = 0          

        
        return np.nan_to_num( np.diag(self.cf).sum() / self.cf.sum() )

    def calc_precision(self) -> np.ndarray:
        #                 a     
        #                  ii   
        # precision  = ---------
        #          i   k - 1    
        #               ===     
        #               \    a  
        #               /     ij
        #               ===     
        #              j = 0    

        return np.nan_to_num( np.diag(self.cf) / self.cf.sum(axis = 1) )

    def calc_recall(self) -> np.ndarray:
        #              a     
        #               ii   
        # recall  = ---------
        #       i   k - 1    
        #            ===     
        #            \    a  
        #            /     ij
        #            ===     
        #           i = 0    

        return np.nan_to_num( np.diag(self.cf) / self.cf.sum(axis = 0) )

    def calc_dice(self) -> np.ndarray:
        #        2.precision.recall
        # dice = ------------------
        #        precision + recall

        return np.nan_to_num( 2 * self.calc_precision() * self.calc_recall() / (self.calc_precision() + self.calc_recall()) )

    def calc_iou(self) -> np.ndarray:
        #                    a              
        #                     ii            
        # IoU  = ---------------------------
        #    i   k - 1       k - 1          
        #         ===         ===           
        #         \    a   +  \    a   - a  
        #         /     ij    /     ji    ii
        #         ===         ===           
        #        j = 0       j = 0          

        return np.nan_to_num(np.diag(self.cf) / ( self.cf.sum(axis = 0) + self.cf.sum(axis = 1) - np.diag(self.cf)))

    def metrics_df(self) -> pd.DataFrame:
        if self.cf.shape[0] == 6:
            metrics_df = pd.concat([
                pd.DataFrame(metric, columns=[metric_name], index = CLASS_NAMES ) for metric, metric_name in zip([self.calc_accuracy(), self.calc_precision(), self.calc_recall(), self.calc_dice(), self.calc_iou()], ['accuracy', 'precision', 'recall', 'dice', 'iou'])
            ], axis=1)
        elif self.cf.shape[0] == 2:
            metrics_df = pd.concat([
                pd.DataFrame(metric, columns=[metric_name], index = SINGLE_CLASS_NAMES ) for metric, metric_name in zip([self.calc_accuracy(), self.calc_precision(), self.calc_recall(), self.calc_dice(), self.calc_iou()], ['accuracy', 'precision', 'recall', 'dice', 'iou'])
            ], axis=1)
        else:
            raise AssertionError
        return metrics_df


    def get_avg_score(self) -> Dict[str , float]:
        """Calculate the average metric scores given the confusion matrix this SegMeter was build with.

        Returns:
            Dict[str , float]: Dictionary with 8 different metric scores:
                            <split>_dice
                            <split>_iou
                            <split>_prec (precision)
                            <split>_recall (sensitivity)
                            <split>_fscore (F1 - metric)
                            <split>_score (Dice score)
                            <split>_struct (structured metric - objectness score)
        """

        # All metrics are computed based on the confusion matrix (self.cf : np.ndarray)


        logger.debug(f'Get average score from confusion matrix :\n{pformat(self.cf)} \n resulting in weights : {np.power(np.sum(self.cf,axis = 1).astype(float), -1)}')

        val_dict = dict()
        val_dict['%s_dice' % self.split] = np.mean(self.calc_dice())
        val_dict['%s_weighted_dice' % self.split] = np.average(self.calc_dice(), weights = np.power(np.sum(self.cf,axis = 1).astype(float), -1))
        val_dict['%s_iou' % self.split] = np.mean(self.calc_iou())

        val_dict['%s_prec' % self.split] = np.mean(self.calc_precision())
        val_dict['%s_recall' % self.split] = np.mean(self.calc_recall())

        val_dict['%s_score' % self.split] = np.average(self.calc_dice(), weights = np.power(np.sum(self.cf,axis = 1).astype(float), -1))
        #val_dict['%s_score' % self.split] = np.mean(self.calc_dice())

        logger.debug(f'Return validation metric average score : {pformat(val_dict)}')
        return val_dict

