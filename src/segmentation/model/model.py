from typing import Tuple

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    BatchNormalization,
    concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class SegmentationModel:
    _model: Model

    def __init__(self, input_shape):
        self._model = self._get_model(input_shape)

    def compile(self):
        self._model.compile(
                optimizer=Adam(
                        learning_rate=5e-5,
                        beta_1=0.9,
                        beta_2=0.99,
                ),
                loss=self.bce_dice_loss,
                metrics=[self.dice],
        )

    def fit(self, X, Y, batch_size, epochs, validation_data, callbacks, verbose=0):
        return self._model.fit(
                x=X,
                y=Y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data=validation_data,
                callbacks=callbacks,
        )

    def load_weights(self, file_path):
        self._model.load_weights(filepath=file_path)

    def predict(self, X):
        return self._model.predict(X)

    @staticmethod
    def _get_model(input_shape: Tuple[int, int, int]) -> Model:
        inp = Input(input_shape)

        x, norm_a = SegmentationModel.pool_norm(inp, 32)
        x, norm_b = SegmentationModel.pool_norm(x, 64)
        x, _ = SegmentationModel.pool_norm(x, 128, pool_size=(1, 1))
        x, _ = SegmentationModel.pool_norm(x, 256, pool_size=(1, 1))
        x = SegmentationModel.conv(x, 256)

        x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = SegmentationModel.conv(x, 128)

        x = Conv2DTranspose(64, (2, 2), padding='same')(x)
        x = concatenate([x, norm_b])
        x = SegmentationModel.conv(x, 64)

        x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
        x = concatenate([x, norm_a], axis=3)
        x = SegmentationModel.conv(x, 32)

        out = Conv2D(1, (1, 1), activation='sigmoid', name='segmentation')(x)

        return Model(inputs=inp, outputs=out)

    @staticmethod
    def dice(y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])

        return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    @staticmethod
    def dice_loss(y_true, y_pred):
        return 1.0 - SegmentationModel.dice(y_true, y_pred)

    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        return 0.5 * binary_crossentropy(y_true, y_pred) + 0.5 * SegmentationModel.dice_loss(y_true, y_pred)

    @staticmethod
    def pool_norm(inp, filters, pool_size=(2, 2)):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(inp)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
        norm = BatchNormalization()(x)
        x = MaxPooling2D(pool_size)(norm)
        x = Dropout(0.2)(x)

        return x, norm

    @staticmethod
    def conv(inp, filters):
        x = BatchNormalization()(inp)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer="he_normal")(x)

        return x
