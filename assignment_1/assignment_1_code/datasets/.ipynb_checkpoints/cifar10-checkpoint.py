import pickle
from typing import Tuple
import numpy as np


from assignment_1_code.datasets.dataset import Subset, ClassificationDataset


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Initializes the CIFAR-10 dataset.
        """
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        self.fdir = fdir
        self.subset = subset
        self.transform = transform

        self.images, self.labels = self.load_cifar()

    def load_cifar(self) -> Tuple:
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Depending on which subset is selected, the corresponding images and labels are returned.

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        """

        # TODO implement
        # See the CIFAR-10 website on how to load the data files

        if not self.fdir.endswith("/"):
                self.fdir += "/"
        
        if self.subset == Subset.TRAINING:
            files = [f"data_batch_{i}" for i in range(1, 5)]
        elif self.subset == Subset.VALIDATION:
            files = ["data_batch_5"]
        elif self.subset == Subset.TEST:
            files = ["test_batch"]
        else:
            raise ValueError("Invalid subset specified.")
    
        all_images = []
        all_labels = []
    
        for filename in files:
            filepath = self.fdir + filename
            try:
                with open(filepath, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    images_flat = batch[b'data']
                    labels = batch[b'labels']
    
                    images = images_flat.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                    all_images.append(images)
                    all_labels.extend(labels)
            except FileNotFoundError:
                raise ValueError(f"Missing file or invalid directory: {filepath}")
            except Exception as e:
                raise ValueError(f"Error loading file {filepath}: {e}")
    
        images_array = np.concatenate(all_images, axis=0).astype(np.uint8)
        labels_array = np.array(all_labels, dtype=np.int64)
        return images_array, labels_array

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """

        if idx < 0 or idx >= self.__len__():
            raise IndexError("Index out of range")

        
        image = self.images[idx]
        label = self.labels[idx]
    
        if self.transform:
            image = self.transform(image)
    
        return image, label

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        return len(np.unique(self.labels))
