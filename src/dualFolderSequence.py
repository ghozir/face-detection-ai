from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class DualFolderSequence(Sequence):
    def __init__(self,
                 path1,
                 path2,
                 target_size=(64, 64),
                 color_mode='grayscale',
                 batch_size=64,
                 shuffle=True,
                 seed=42,
                 augment_params=None):

        self.batch_size = batch_size
        self.batch_size_half = batch_size // 2
        self.shuffle = shuffle
        self.seed = seed

        self.datagen1 = ImageDataGenerator(rescale=1./255, **(augment_params or {}))
        self.datagen2 = ImageDataGenerator(rescale=1./255, **(augment_params or {}))

        self.gen1 = self.datagen1.flow_from_directory(
            path1,
            target_size=target_size,
            color_mode=color_mode,
            batch_size=self.batch_size_half,
            class_mode='categorical',
            shuffle=shuffle,
            seed=seed
        )

        self.gen2 = self.datagen2.flow_from_directory(
            path2,
            target_size=target_size,
            color_mode=color_mode,
            batch_size=self.batch_size_half,
            class_mode='categorical',
            shuffle=shuffle,
            seed=seed
        )

        # Pastikan class_indices identik
        assert self.gen1.class_indices == self.gen2.class_indices, "Class indices mismatch!"

        self.class_indices = self.gen1.class_indices
        self.classes = np.concatenate([self.gen1.classes, self.gen2.classes])
        self.samples = self.gen1.samples + self.gen2.samples

    def __len__(self):
        return self.samples // self.batch_size

    def __getitem__(self, index):
        x1, y1 = next(self.gen1)
        x2, y2 = next(self.gen2)
        return np.concatenate([x1, x2]), np.concatenate([y1, y2])

    def on_epoch_end(self):
        if self.shuffle:
            self.gen1.on_epoch_end()
            self.gen2.on_epoch_end()
