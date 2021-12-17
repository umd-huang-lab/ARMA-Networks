"""Evaluation metric implementations for audio synthesis tasks"""
import glob
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Type

import numpy as np
import torch
from scipy.io import wavfile
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class WavDataSet(Dataset):
    """Torch dataset for wavfile directories"""

    def __init__(
        self,
        samples: str,
        labels: Optional[Iterable[Any]] = None,
        transform: Optional[Callable[[np.ndarray], Any]] = None,
        train: bool = True,
        dtype: Type = torch.FloatTensor,
    ):
        """
        Args:
            samples: Path to directory containing audio samples (wav files are supported)
            transform: Optionally provide a preprocessing function to transform the audio data before predicting the
                       class with the classifier
            dtype: Datatype to cast the loaded numpy array of audio data to (done before passing to transform function)
        """
        self.files = glob.glob(os.path.join(Path(samples), "*.wav"))
        self.labels = np.array(labels) if labels is not None else None
        self.transform = transform if transform is not None else lambda audio: audio
        self.train = train
        self.dtype = dtype

        if not train:
            if labels is None:
                raise ValueError("Cannot create test dataloader without labels")
            if labels.shape[0] != len(self.files):
                raise ValueError(
                    f"The number of labels provided does not match the number of samples, got {labels.shape[0]} labels"
                    f" and {len(self.files)} samples"
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]
        _, audio = wavfile.read(file)
        audio = self.transform(torch.from_numpy(audio).to(dtype=self.dtype))
        return audio if self.train else audio, self.labels[item]


def _check_cuda(cuda: bool):
    # Make sure cuda is available if using cuda
    if cuda and not torch.cuda.is_available():
        raise EnvironmentError("CUDA set to true but no CUDA enabled device available")
    # Warn if cuda is available and not using
    if not cuda and torch.cuda.is_available():
        warnings.warn("A CUDA enabled device is available, but cuda is not set to True")


def audio_inception_score(
    classifier: Callable[..., np.ndarray],
    samples: Union[str, Path],
    transform: Optional[Callable[[np.ndarray], Any]] = None,
    batch_size: int = 4,
    splits: int = 10,
    n_classes: int = 10,
    shuffle: bool = True,
    cuda: bool = True,
) -> np.ndarray:
    """Inception score implementation adapted for audio synthesis performance evaluation

    Based on https://github.com/openai/improved-gan/blob/master/inception_score/model.py
    From Improved Techniques for Training GANs (Goodfellow, 2016) https://arxiv.org/pdf/1606.03498.pdf

    Args:
        classifier: Classification model (in evaluation mode) which classifies an audio sample into <n_classes> by
                    computing confidence scores for each class, for each sample
        samples: Path to directory containing audio samples (wav files are supported)
        transform: Optionally provide a preprocessing function to transform the audio data before predicting the class
                   with the classifier
        batch_size: Integer representing the number of samples to predict on in each iteration
        splits: Integer representing the number of splits to chunk the predictions into, producing an inception score
                for each chunk
        n_classes: The number of classes predicted by the classification model
        shuffle: Boolean flag, whether or not to shuffle the dataset
        cuda: Boolean flag, whether or not to use a CUDA device for the classification model

    Returns:
        <splits> x 1 np.ndarray containing the computed inception score for each split
    """
    _check_cuda(cuda)
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # CUDA type if on cuda
    dataloader = DataLoader(
        WavDataSet(samples, None, transform, False, dtype), batch_size=batch_size, shuffle=shuffle, num_workers=0
    )

    # Must have >= 1 sample per split
    n = len(dataloader.dataset)
    if splits > n:
        raise ValueError(f"Cannot compute inception score for {splits} splits from only {n} samples")

    # Process classification predictions in batches
    predictions = np.empty((n, n_classes), dtype=np.float64)
    for i, batch in enumerate(dataloader):
        preds = classifier(batch)
        preds = F.softmax(preds).data.cpu().numpy()
        predictions[i * batch_size : (i + 1) * batch_size] = preds

    # Compute inception scores
    scores = np.empty(splits, dtype=np.float64)
    split_size = n // splits
    for i in range(splits):
        preds_split = predictions[i * split_size : (i + 1) * split_size]
        kl = preds_split * (np.log(preds_split) - np.log(np.expand_dims(np.mean(preds_split, axis=0), axis=0)))
        kl = np.exp(np.mean(np.sum(kl, axis=1)))
        scores[i] = kl
    return scores


def pitch_accuracy_entropy(
    classifier: Callable[..., np.ndarray],
    samples: str,
    labels: np.ndarray,
    transform: Optional[Callable[[np.ndarray], Any]] = None,
    batch_size: int = 4,
    shuffle: bool = True,
    cuda: bool = True,
):
    """Implementation of pitch accuracy and pitch entropy as described in GANSynth: Adversarial Neural Audio Synthesis
    (Engel, 2019) https://arxiv.org/abs/1902.08710

    Args:
        classifier: Classification model (in evaluation mode) which classifies an audio sample into <n_classes> by
                    computing confidence scores for each class, for each sample
        samples: Path to directory containing audio samples (wav files are supported)
        labels: Numpy array of integers representing the true label for each corresponding sample (index of label)
        transform: Optionally provide a preprocessing function to transform the audio data before predicting the class
                   with the classifier
        batch_size: Integer representing the number of samples to predict on in each iteration
        shuffle: Boolean flag, whether or not to shuffle the dataset
        cuda: Boolean flag, whether or not to use a CUDA device for the classification model

    Returns:
        <splits> x 1 np.ndarray containing the computed inception score for each split
    """
    _check_cuda(cuda)
    dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # CUDA type if on cuda
    dataloader = DataLoader(
        WavDataSet(samples, labels, transform, True, dtype), batch_size=batch_size, shuffle=shuffle, num_workers=0
    )

    predictions = np.empty(len(dataloader.dataset), dtype=np.int32)
    for i, batch in enumerate(dataloader):
        preds = classifier(batch)
        preds = F.argmax(preds).data.cpu().numpy()
        predictions[i * batch_size : (i + 1) * batch_size] = preds
    probs = np.array([(predictions == i).mean() for i in range(labels.min(), labels.max() + 1)], dtype=np.float64)
    return (labels == predictions).mean(), -(probs @ np.log(probs))  # Compute accuracy and entropy of predictions
