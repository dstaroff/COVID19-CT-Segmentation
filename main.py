import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)

import argparse

import numpy as np
import tqdm

from src.segmentation.image import ImageLoader
from src.segmentation.model import SegmentationModel
from src.utils import (Const, Estimator)


def get_args():
    parser = argparse.ArgumentParser(
            description='Инструмент для оценки поражения легких COVID 19 с помощью обработки КТ-снимков.',
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
            '-i',
            '--imgpath',
            action='store',
            help='Путь к изображению КТ-снимка',
    )
    group.add_argument(
            '-f',
            '--folderpath',
            action='store',
            help='Путь к папке с изображениями КТ-снимков',
    )

    return validate_args(parser.parse_args())


def validate_args(args):
    if args.imgpath is not None:
        if not os.path.exists(args.imgpath) or not os.path.isfile(args.imgpath):
            raise FileNotFoundError(f'Объект по пути "{args.imgpath}" не существует или не является файлом')
    else:
        if not os.path.exists(args.folderpath) or not os.path.isdir(args.folderpath):
            raise FileNotFoundError(f'Объект по пути "{args.folderpath}" не существует или не является папкой')

        file_paths = map(
                lambda path: os.path.join(args.folderpath, path),
                os.listdir(args.folderpath)
        )

        inappropriate_files = []
        for file_path in file_paths:
            if not os.path.exists(file_path) or \
                    not os.path.isfile(file_path) or \
                    os.path.basename(file_path).split('.')[-1] != 'png':
                inappropriate_files.append(file_path)

        if len(inappropriate_files) > 0:
            error = [
                f'Следующие объекты не существуют или не являются файлами или не являются PNG изображениями:',
                '\n'.join(inappropriate_files)
            ]
            raise FileNotFoundError('\n'.join(error))

    return args


def load_model(weights_path):
    model = SegmentationModel((Const.IMG_SIZE, Const.IMG_SIZE, 1))
    model.load_weights(weights_path)

    return model


# noinspection PyUnboundLocalVariable,PyBroadException
def main():
    try:
        args = get_args()
    except FileNotFoundError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(e)
        exit(1)

    try:
        lungs_model = load_model(Const.LUNGS_MODEL_WEIGHTS_PATH)
        covid_model = load_model(Const.COVID_MODEL_WEIGHTS_PATH)
    except Exception:
        print('Не удалось загрузить модель сегментации')
        exit(1)

    if args.imgpath is not None:
        original_image = np.array([ImageLoader.load_sample(args.imgpath)])

        lungs_image = lungs_model.predict(original_image)[0]
        covid_image = covid_model.predict(original_image)[0]

        affection = Estimator.content(covid_image, lungs_image)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(figsize=(9, 9))
        fig.canvas.manager.set_window_title(f'Поражение легких COVID 19 = {affection:.2%}')
        fig.patch.set_facecolor('black')
        axes.set_xticks([])
        axes.set_yticks([])
        axes.imshow(original_image[0], cmap='bone')
        axes.contourf(
                lungs_image.reshape((Const.IMG_SIZE, Const.IMG_SIZE)),
                levels=np.linspace(0.5, 1.0, 2),
                cmap='Greens',
                alpha=0.3,
        )
        axes.contourf(
                covid_image.reshape((Const.IMG_SIZE, Const.IMG_SIZE)),
                levels=np.linspace(0.5, 1.0, 2),
                cmap='Reds',
                alpha=0.5,
        )
        plt.show()

        print(f'Поражение легких COVID 19 = {affection:.2%}')
    else:
        original_images = ImageLoader.load_samples(args.folderpath)
        lungs_images = lungs_model.predict(original_images)
        covid_images = covid_model.predict(original_images)

        affection = 0.0
        for i in tqdm.trange(len(original_images)):
            lungs_image = lungs_images[i]
            covid_image = covid_images[i]

            affection += Estimator.content(covid_image, lungs_image)

        affection /= len(original_images)

        print(f'Общее поражение легких COVID 19 = {affection:.2%}')


if __name__ == "__main__":
    main()
