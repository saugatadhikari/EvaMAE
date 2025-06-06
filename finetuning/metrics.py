import statistics as stats
import numpy as np
from sklearn.metrics import confusion_matrix



class Evaluator():
    """
    The Evaluator currently can do the following metrics:
        - Precision
        - Recall
        - Fscore
    """

    def __init__(self, use_evaloss=False, landslide=False):

        # Declare Metrics
        self.DRY_ACC = 0
        self.FLOOD_ACC = 0
        
        self.DRY_PRECISION = 0
        self.FLOOD_PRECISION = 0
        
        self.DRY_RECALL = 0
        self.FLOOD_RECALL = 0
        
        self.DRY_FSCORE = 0
        self.FLOOD_FSCORE = 0

        self.use_evaloss = use_evaloss
        self.landslide = landslide
    
    def run_eval(self, pred_unpadded, gt_labels):
        
        # cm = confusion_matrix(gt_labels.flatten(), pred_unpadded.flatten(), labels = [0, 1, 2])
        # TP_0 = cm[0][0]
        # FP_0 = cm[1][0]
        # FN_0 = cm[0][1]
        # TN_0 = cm[1][1]

        # TP_1 = cm[1][1]
        # FP_1 = cm[0][1]
        # FN_1 = cm[1][0]
        # TN_1 = cm[0][0]

        if self.landslide:
            cm = confusion_matrix(gt_labels, pred_unpadded, labels = [0, 1])

            TP_0 = cm[0][0]
            FP_0 = cm[1][0]
            FN_0 = cm[0][1]
            TN_0 = cm[1][1]

            TP_1 = cm[1][1]
            FP_1 = cm[0][1]
            FN_1 = cm[1][0]
            TN_1 = cm[0][0]
        else:
            if not self.use_evaloss:
                cm = confusion_matrix(gt_labels, pred_unpadded, labels = [0, 1, 2])

                TP_0 = cm[0][0]
                FP_0 = cm[2][0]
                FN_0 = cm[0][2]
                TN_0 = cm[2][2]

                TP_1 = cm[2][2]
                FP_1 = cm[0][2]
                FN_1 = cm[2][0]
                TN_1 = cm[0][0]
            else:
                cm = confusion_matrix(gt_labels, pred_unpadded, labels = [-1, 1, 0])

                TP_0 = cm[0][0]
                FP_0 = cm[1][0]
                FN_0 = cm[0][1]
                TN_0 = cm[1][1]

                TP_1 = cm[1][1]
                FP_1 = cm[0][1]
                FN_1 = cm[1][0]
                TN_1 = cm[0][0]

        # print(cm)

        
        
        
        ####DRY
        # self.DRY_ACC = ((TP_0+TN_0)/(TP_0+TN_0+FP_0+FN_0))*100
        # print("Dry Accuracy: ", self.DRY_ACC)
        # self.DRY_PRECISION = ((TP_0)/(TP_0+FP_0))*100
        # print("Dry Precision: ", self.DRY_PRECISION)
        # self.DRY_RECALL = ((TP_0)/(TP_0+FN_0))*100
        # print("Dry Recall: ", self.DRY_RECALL)
        # self.DRY_FSCORE = ((2*self.DRY_PRECISION*self.DRY_RECALL)/(self.DRY_PRECISION+self.DRY_RECALL))
        # print("Dry F-score: ", self.DRY_FSCORE)

        if (TP_0+FP_0+FN_0) > 0:
            self.DRY_IOU = ((TP_0)/(TP_0+FP_0+FN_0))*100
        else:
            self.DRY_IOU = None

        # print("Dry IoU: ", self.DRY_IOU)
        # print("ERROR!!!")
        # print(cm)
        # print("-----------")
        
        # print("\n")
        
        ####FLOOD
        # self.FLOOD_ACC = ((TP_1+TN_1)/(TP_1+TN_1+FP_1+FN_1))*100
        # print("Flood Accuracy: ", self.FLOOD_ACC)
        # self.FLOOD_PRECISION = ((TP_1)/(TP_1+FP_1))*100
        # print("Flood Precision: ", self.FLOOD_PRECISION)
        # self.FLOOD_RECALL = ((TP_1)/(TP_1+FN_1))*100
        # print("Flood Recall: ", self.FLOOD_RECALL)
        # self.FLOOD_FSCORE = ((2*self.FLOOD_PRECISION*self.FLOOD_RECALL)/(self.FLOOD_PRECISION+self.FLOOD_RECALL))
        # print("Flood F-score: ", self.FLOOD_FSCORE)
        if (TP_1+FP_1+FN_1) > 0:
            self.FLOOD_IOU = ((TP_1)/(TP_1+FP_1+FN_1))*100
        else:
            self.FLOOD_IOU = None
        # print("Flood IoU: ", self.FLOOD_IOU)
        
        # print("\nMean IoU: ", (self.DRY_IOU+self.FLOOD_IOU)/2)

        valid_ious = [iou for iou in [self.DRY_IOU, self.FLOOD_IOU] if iou is not None]
        if valid_ious:
            self.MEAN_IOU = sum(valid_ious) / len(valid_ious)
        else:
            self.MEAN_IOU = None  # Or None

        # self.MEAN_IOU = (self.DRY_IOU+self.FLOOD_IOU)/2
        
        return self.FLOOD_IOU, self.DRY_IOU, self.MEAN_IOU
        

        
    
    
    @property
    def f_accuracy(self):        
        if self.FLOOD_ACC > 0:
            return self.FLOOD_ACC
        else:
            return 0.0

    @property
    def f_precision(self):        
        if self.FLOOD_PRECISION > 0:
            return self.FLOOD_PRECISION
        else:
            return 0.0

 
    @property
    def f_recall(self):
        if self.FLOOD_RECALL > 0:
            return self.FLOOD_RECALL
        else:
            return 0.0
        
        
    @property
    def f_fscore(self):
        if self.FLOOD_FSCORE > 0:
            return self.FLOOD_FSCORE
        else:
            return 0.0
    
    @property
    def f_iou(self):
        if self.FLOOD_IOU > 0:
            return self.FLOOD_IOU
        else:
            return 0.0
    
    
    
    
    @property
    def d_accuracy(self):        
        if self.DRY_ACC > 0:
            return self.DRY_ACC
        else:
            return 0.0
    
    @property
    def d_precision(self):        
        if self.DRY_PRECISION > 0:
            return self.DRY_PRECISION
        else:
            return 0.0

 
    @property
    def d_recall(self):
        if self.DRY_RECALL > 0:
            return self.DRY_RECALL
        else:
            return 0.0
        
        
    @property
    def d_fscore(self):
        if self.DRY_FSCORE > 0:
            return self.DRY_FSCORE
        else:
            return 0.0
    
    
    @property
    def d_iou(self):
        if self.DRY_IOU > 0:
            return self.DRY_IOU
        else:
            return 0.0
            
            
            
