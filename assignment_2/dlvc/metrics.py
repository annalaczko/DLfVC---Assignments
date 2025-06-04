from abc import ABCMeta, abstractmethod
import torch
import numpy as np

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(self.classes)

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''

        self.num_classes = len(self.classes)
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)



    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

        if prediction.ndim != 4 or target.ndim != 3:
            raise ValueError("Dimension is not compatible")

        b, c, h, w = prediction.shape
        if target.shape != (b, h, w):
            raise ValueError("Shape mismatch or class mismatch between prediction and target.")

        if c != len(self.classes):
            raise ValueError(f"Class dimension is not compatible between prediction and target {c} : {self.classes}")

        pred_labels = prediction.argmax(dim=1) #getting the prediction by choosing argmax
        mask = (target != 255)  # ignoring pixels, boolean tensor

        for i in range(b):
            true = target[i][mask[i]].view(-1)
            pred = pred_labels[i][mask[i]].view(-1)

            for t, p in zip(true, pred):
                if 0 <= t.item() < self.num_classes and 0 <= p.item() < self.num_classes:
                    self.confusion_matrix[t.item(), p.item()] += 1
   

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"Mean IoU: {self.mIoU():.2f}"
          

    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        TP = torch.diag(self.confusion_matrix)
        TP_FN = self.confusion_matrix.sum(dim=1) 
        TP_FP = self.confusion_matrix.sum(dim=0)
        denom = TP_FP + TP_FN - TP

        ious = []
        for i in range(self.num_classes):
            if denom[i] == 0:
                ious.append(0.0)
            else:
                ious.append(TP[i].item() / denom[i].item())

        return float(np.mean(ious)) if ious else 0.0





