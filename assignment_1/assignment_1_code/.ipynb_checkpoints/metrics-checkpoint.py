from abc import ABCMeta, abstractmethod
import torch


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """

        if not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError("Prediction and target are not tensors")

        if len(prediction.shape) != 2 or len(target.shape) != 1:
            raise ValueError("Invalid input tensor shapes.")

        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Target and Prediction length is not the same")

        pred_labels = prediction.argmax(dim=1)

        for pred, true in zip(pred_labels, target):
            pred = pred.item()
            true = true.item()
            class_name = self.classes[true]
            self.total_pred[class_name] += 1
            if pred == true:
                self.correct_pred[class_name] += 1
                self.n_matching += 1
            self.n_total += 1

    def __str__(self):
        """
        Return a string representation of the performance, accuracy and per class accuracy.
        """

        return f"Accuracy: {self.accuracy():.4f},\nPer-class Accuracy: {self.per_class_accuracy():.4f}"

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """

        if (self.n_total==0):
            return 0
        return self.n_matching/self.n_total
        
    def per_class_accuracy(self) -> float:
        """
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """

        sum_ = 0.0
        class_count = 0
        for classname in self.classes:
            if self.total_pred[classname]> 0:
                sum_+= self.correct_pred[classname] / self.total_pred[classname]
                class_count+= 1
        return sum_ / class_count if class_count > 0 else 0.0
        
