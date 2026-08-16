"""
NIH ChestX-ray14 loading, preprocessing and the Keras Sequence generator
used by all federated clients.

UPDATED VERSION:
- Memory-efficient image preprocessing
- Explicit float32 usage
- Resize before expensive normalization
- Preserves CLAHE preprocessing
- Preserves federated client data splitting
"""

import itertools
import os
import gc

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config


# ======================================================================
# LABEL HANDLING
# ======================================================================

def load_and_prepare_dataframe():
    """
    Load Data_Entry_2017.csv.

    Performs:
    1. Metadata loading
    2. Multi-label one-hot encoding
    3. NIH official train/test split
    4. Train/validation split
    """

    df_all = pd.read_csv(config.CSV_PATH)

    # --------------------------------------------------------------
    # OLD CODE
    # --------------------------------------------------------------
    # df_all = df_all.rename(columns={
    #     'OriginalImagePixelSpacing[x': 'PixelSpacingX',
    #     'y]': 'PixelSpacingY',
    #     'OriginalImage[Width': 'Width',
    #     'Height]': 'Height',
    # })
    #
    # Kept as comments for reference.
    # --------------------------------------------------------------

    # Rename malformed NIH metadata column names
    df_all = df_all.rename(columns={
        'OriginalImagePixelSpacing[x': 'PixelSpacingX',
        'y]': 'PixelSpacingY',
        'OriginalImage[Width': 'Width',
        'Height]': 'Height',
    })

    # NIH official train/validation and test image lists
    train_val_images = pd.read_csv(
        config.TRAIN_VAL_LIST,
        header=None,
        names=['Image Index']
    )

    test_images = pd.read_csv(
        config.TEST_LIST,
        header=None,
        names=['Image Index']
    )

    # --------------------------------------------------------------
    # Find all disease labels
    # --------------------------------------------------------------

    unique_labels_all = sorted(
        list(
            set(
                itertools.chain.from_iterable(
                    df_all['Finding Labels']
                    .apply(lambda x: x.split('|'))
                    .values
                )
            )
        )
    )

    # --------------------------------------------------------------
    # OLD ONE-HOT IMPLEMENTATION
    # --------------------------------------------------------------
    # one_hot = pd.DataFrame(
    #     0.0,
    #     index=np.arange(len(df_all)),
    #     columns=unique_labels_all
    # )
    #
    # for index, row in df_all.iterrows():
    #     for label in row['Finding Labels'].split('|'):
    #         one_hot.loc[index, label] = 1.0
    #
    # df_all = pd.concat([df_all, one_hot], axis=1)
    #
    # This code is retained conceptually but can be expensive because
    # it modifies the DataFrame row-by-row.
    # --------------------------------------------------------------

    # Memory-efficient one-hot encoding
    one_hot = pd.DataFrame(
        0.0,
        index=df_all.index,
        columns=unique_labels_all,
        dtype=np.float32
    )

    for idx, labels in df_all['Finding Labels'].items():
        for label in labels.split('|'):
            one_hot.at[idx, label] = 1.0

    df_all = pd.concat([df_all, one_hot], axis=1)

    # "No Finding" is not considered a disease class
    unique_labels = [
        label for label in unique_labels_all
        if label != 'No Finding'
    ]

    # --------------------------------------------------------------
    # Official NIH train/test split
    # --------------------------------------------------------------

    train_val_df = df_all[
        df_all['Image Index'].isin(
            train_val_images['Image Index']
        )
    ]

    # Split NIH training/validation data
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1,
        random_state=config.RANDOM_SEED
    )

    # Official NIH test set
    test_df = df_all[
        df_all['Image Index'].isin(
            test_images['Image Index']
        )
    ]

    return train_df, val_df, test_df, unique_labels


# ======================================================================
# FEDERATED CLIENT DATA SPLITTING
# ======================================================================

def split_dataframe(
    df,
    num_clients,
    seed=config.RANDOM_SEED
):
    """
    Split training dataframe into roughly equal Pandas DataFrames.

    Each shard represents one independent healthcare institution /
    federated client.
    """

    # Shuffle the complete training dataset
    shuffled = df.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)

    # --------------------------------------------------------------
    # OLD CODE
    # --------------------------------------------------------------
    # return np.array_split(shuffled, num_clients)
    #
    # Depending on NumPy/Pandas behavior this could return objects
    # that are interpreted incorrectly.
    # --------------------------------------------------------------

    # Explicitly create Pandas DataFrame shards
    splits = np.array_split(
        np.arange(len(shuffled)),
        num_clients
    )

    return [
        shuffled.iloc[indices].reset_index(drop=True)
        for indices in splits
    ]


# ======================================================================
# DATA AUGMENTATION
# ======================================================================

data_augmentation = tf.keras.Sequential(
    name='aug',
    layers=[
        tf.keras.layers.RandomFlip(
            'horizontal',
            name='hflip'
        )
    ]
)


# ======================================================================
# NIH CHEST X-RAY SEQUENCE
# ======================================================================

class CXR14Sequence(tf.keras.utils.Sequence):
    """
    Keras Sequence for NIH ChestX-ray14.

    Pipeline:

        Original X-ray
              |
              v
        Grayscale conversion
              |
              v
        Gaussian blur
              |
              v
        CLAHE enhancement
              |
              v
        Resize to 224x224
              |
              v
        float32 normalization
              |
              v
        RGB conversion
              |
              v
        Data augmentation
              |
              v
        CNN model
    """

    def __init__(
        self,
        df,
        images_dir,
        unique_labels,
        batch_size=32,
        shuffle=True,
        preprocessing=True,
        augmentation=False
    ):

        super().__init__()

        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.unique_labels = unique_labels
        self.batch_size = batch_size
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.shuffle = shuffle

        # CLAHE configuration
        self.transform = cv.createCLAHE(
            clipLimit=3,
            tileGridSize=(10, 10)
        )

        # Image path cache
        self._image_path_cache = None

    # ==================================================================
    # EPOCH END
    # ==================================================================

    def on_epoch_end(self):

        if self.shuffle:

            self.df = self.df.sample(
                frac=1,
                random_state=None
            ).reset_index(drop=True)

    # ==================================================================
    # NUMBER OF BATCHES
    # ==================================================================

    def __len__(self):

        return int(
            np.ceil(
                len(self.df) / self.batch_size
            )
        )

    # ==================================================================
    # IMAGE PATH RESOLUTION
    # ==================================================================

    def _resolve_image_path(self, image_index):
        """
        NIH dataset stores images in multiple folders.

        The mapping is created only once and then reused.
        """

        if self._image_path_cache is None:

            print(
                "Building NIH image path cache. "
                "This happens only once per client."
            )

            self._image_path_cache = {}

            for root, _, files in os.walk(
                self.images_dir
            ):

                for filename in files:

                    if filename.lower().endswith('.png'):

                        self._image_path_cache[
                            filename
                        ] = os.path.join(
                            root,
                            filename
                        )

        return self._image_path_cache.get(
            image_index
        )

    # ==================================================================
    # MEMORY-EFFICIENT PREPROCESSING
    # ==================================================================

    def preprocess(self, image):

        """
        Memory-efficient preprocessing.

        IMPORTANT:
        The original implementation performed normalization while the
        image was still 1024x1024 and could become float64.

        That caused:

            MemoryError:
            Unable to allocate 24.0 MiB for
            (1024, 1024, 3) float64

        The new pipeline resizes the image before converting to
        float32 RGB representation.
        """

        # --------------------------------------------------------------
        # OLD CODE
        # --------------------------------------------------------------
        #
        # image = np.array(image)
        #
        # if image.ndim == 3:
        #     image = cv.cvtColor(
        #         image,
        #         cv.COLOR_RGB2GRAY
        #     )
        #
        # gb_image = cv.GaussianBlur(
        #     image,
        #     (9, 9),
        #     0
        # )
        #
        # clahe_img = self.transform.apply(
        #     gb_image
        # )
        #
        # clahe_img = cv.cvtColor(
        #     clahe_img,
        #     cv.COLOR_GRAY2RGB
        # )
        #
        # clahe_img = self._min_max_normalize(
        #     clahe_img
        # )
        #
        # return cv.resize(
        #     clahe_img,
        #     config.IMAGE_SIZE
        # )
        #
        # PROBLEM:
        # Normalization happened before resize.
        #
        # For a 1024x1024 image, converting to float64 could require
        # approximately 24 MB for just one RGB temporary array.
        #
        # --------------------------------------------------------------

        # Make sure input is uint8
        if image.dtype != np.uint8:

            image = image.astype(
                np.uint8,
                copy=False
            )

        # --------------------------------------------------------------
        # Convert to grayscale
        # --------------------------------------------------------------

        if image.ndim == 3:

            # cv.imread() returns BGR
            gray = cv.cvtColor(
                image,
                cv.COLOR_BGR2GRAY
            )

        else:

            gray = image

        # --------------------------------------------------------------
        # Gaussian blur
        # --------------------------------------------------------------

        blurred = cv.GaussianBlur(
            gray,
            (9, 9),
            0
        )

        # --------------------------------------------------------------
        # CLAHE enhancement
        # --------------------------------------------------------------

        clahe_img = self.transform.apply(
            blurred
        )

        # --------------------------------------------------------------
        # IMPORTANT MEMORY OPTIMIZATION
        #
        # Resize while still uint8.
        # --------------------------------------------------------------

        resized = cv.resize(
            clahe_img,
            config.IMAGE_SIZE,
            interpolation=cv.INTER_AREA
        )

        # --------------------------------------------------------------
        # Convert only the small 224x224 image to float32
        # --------------------------------------------------------------

        resized = resized.astype(
            np.float32,
            copy=False
        )

        # --------------------------------------------------------------
        # Normalize to [0, 1]
        # --------------------------------------------------------------

        resized /= 255.0

        # --------------------------------------------------------------
        # Convert grayscale -> RGB
        #
        # Required because the CNN expects:
        #
        # (224, 224, 3)
        # --------------------------------------------------------------

        rgb_image = cv.cvtColor(
            resized,
            cv.COLOR_GRAY2RGB
        )

        # Ensure float32
        rgb_image = rgb_image.astype(
            np.float32,
            copy=False
        )

        return rgb_image

    # ==================================================================
    # OLD NORMALIZATION METHOD
    # ==================================================================

    # --------------------------------------------------------------
    # OLD CODE
    # --------------------------------------------------------------
    #
    # @staticmethod
    # def _min_max_normalize(image):
    #     return (
    #         image - image.min()
    #     ) / (
    #         image.max() -
    #         image.min() +
    #         1e-8
    #     )
    #
    # PROBLEM:
    # NumPy could promote uint8 values to float64.
    #
    # --------------------------------------------------------------

    # ==================================================================
    # SAFE FLOAT32 NORMALIZATION
    # ==================================================================

    @staticmethod
    def _min_max_normalize(image):

        """
        Safe float32 normalization.

        This function is retained for compatibility with any code that
        may call it directly.

        The main preprocessing path now uses /255.0 after resizing.
        """

        image = image.astype(
            np.float32,
            copy=False
        )

        min_val = image.min()
        max_val = image.max()

        if max_val - min_val < 1e-8:

            return np.zeros_like(
                image,
                dtype=np.float32
            )

        return (
            image - min_val
        ) / (
            max_val - min_val
        )

    # ==================================================================
    # BATCH LOADING
    # ==================================================================

    def __getitem__(self, index):

        batch_df = self.df.iloc[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        images = []
        labels = []

        # --------------------------------------------------------------
        # Load individual images
        # --------------------------------------------------------------

        for _, row in batch_df.iterrows():

            path = self._resolve_image_path(
                row['Image Index']
            )

            if path is None:

                continue

            img = cv.imread(
                path,
                cv.IMREAD_COLOR
            )

            if img is None:

                continue

            # ----------------------------------------------------------
            # Preprocessing
            # ----------------------------------------------------------

            if self.preprocessing:

                img = self.preprocess(img)

            else:

                # ------------------------------------------------------
                # OLD CODE
                # ------------------------------------------------------
                #
                # img = cv.resize(
                #     img,
                #     config.IMAGE_SIZE
                # ) / 255.0
                #
                # This could also produce float64.
                #
                # ------------------------------------------------------

                img = cv.resize(
                    img,
                    config.IMAGE_SIZE,
                    interpolation=cv.INTER_AREA
                )

                img = img.astype(
                    np.float32,
                    copy=False
                )

                img /= 255.0

            images.append(img)

            labels.append(
                row[
                    self.unique_labels
                ].values.astype(
                    np.float32
                )
            )

        # ==================================================================
        # CREATE BATCH ARRAYS
        # ==================================================================

        if len(images) == 0:

            # Return correctly shaped empty arrays
            x = np.empty(
                (
                    0,
                    config.IMAGE_SIZE[0],
                    config.IMAGE_SIZE[1],
                    3
                ),
                dtype=np.float32
            )

            y = np.empty(
                (
                    0,
                    len(self.unique_labels)
                ),
                dtype=np.float32
            )

        else:

            x = np.asarray(
                images,
                dtype=np.float32
            )

            y = np.asarray(
                labels,
                dtype=np.float32
            )

        # ==================================================================
        # DATA AUGMENTATION
        # ==================================================================

        if (
            self.augmentation
            and len(x) > 0
        ):

            x = data_augmentation(
                x,
                training=True
            ).numpy()

            # Explicitly ensure float32
            x = x.astype(
                np.float32,
                copy=False
            )

        # ==================================================================
        # MEMORY CLEANUP
        # ==================================================================

        # Delete Python lists after conversion
        del images
        del labels

        # Occasionally help Python release temporary objects.
        # This is intentionally lightweight and only affects CPU memory.
        gc.collect()

        return x, y