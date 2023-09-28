"""
1.实现图片数据增强：包括多种方法，如：图片旋转，比例调整，高斯噪声，饱和度变换等；
2.若图片为标注数据集，包含boundingbox的坐标，可同样对其进行变换，并保证圈选内容的不变性;
3.实现多种方式random组合调用
"""
import cv2
import numpy as np
import copy
import random
# from PIL import Image
import os
import random
from typing import List,Dict


class ImageAugment(object):

    @classmethod
    def image_read(cls, image_path: str):
        try:
            # return cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
            return cv2.imread(image_path)
        except Exception as e:
            print(e)
            return None

    @classmethod
    def image_rotate_90_clockwise(cls, image, mode: str = "img", notes: List[Dict] = None):
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_height, rotated_width = rotated_image.shape[:2]

        if mode == 'notes' and notes is None:
            raise ValueError("When mode is set to 'notes', you must provide notes data.")
        elif mode == 'notes':
            points = [i.get('points') for i in notes]
            rotated_points = []

            for box in points:
                rotated_box = []
                for point in box:
                    x = rotated_width - point[1]  # 原始坐标中的x对应旋转后的宽度减去y
                    y = point[0]  # 原始坐标中的y对应旋转后的x
                    rotated_box.append([x, y])
                rotated_points.append(rotated_box)

            rotated_notes = copy.deepcopy(notes)
            for i in range(len(rotated_notes)):
                rotated_notes[i]['points'] = rotated_points[i]
            return rotated_image, rotated_notes
        elif mode == 'img':
            return rotated_image

    @classmethod
    def image_rotate_180(cls, image, mode: str = "img", notes: List[Dict] = None):
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        rotated_height, rotated_width = rotated_image.shape[:2]
        if mode == 'notes' and notes is None:
            raise ValueError("When mode is set to 'notes', you must provide notes data.")
        elif mode == 'notes':
            points = [i.get('points') for i in notes]
            rotated_points = []
            for box in points:
                rotated_box = []
                for point in box:
                    x = rotated_width - point[0]
                    y = rotated_height - point[1]
                    rotated_box.append([x, y])
                rotated_points.append(rotated_box)

            rotated_notes = copy.deepcopy(notes)
            for i in range(len(rotated_notes)):
                rotated_notes[i]['points'] = rotated_points[i]
            return rotated_image, rotated_notes
        elif mode == 'img':
            return rotated_image

    @classmethod
    def image_rotate_90_counterclockwise(cls, image, mode: str = "img", notes: List[Dict] = None):
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_height, rotated_width = rotated_image.shape[:2]
        if mode == 'notes' and notes is None:
            raise ValueError("When mode is set to 'notes', you must provide notes data.")
        elif mode == 'notes':
            points = [i.get('points') for i in notes]
            rotated_points = []
            for box in points:
                rotated_box = []
                for point in box:
                    x = point[1]
                    y = rotated_height - point[0]
                    rotated_box.append([x, y])
                rotated_points.append(rotated_box)
            rotated_notes = copy.deepcopy(notes)
            for i in range(len(rotated_notes)):
                rotated_notes[i]['points'] = rotated_points[i]
            return rotated_image, rotated_notes
        elif mode == 'img':
            return rotated_image

    @classmethod
    def resize_image(cls, image, mode: str = "img", notes: List[Dict] = None, scale_percent: int = None):
        if scale_percent is None:
            scale_percent = random.randrange(30, 80)
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        resized_image = cv2.resize(image, (width, height))
        if mode == 'notes' and notes is None:
            raise ValueError("When mode is set to 'notes', you must provide notes data.")
        elif mode == 'notes':
            points = [i.get('points') for i in notes]
            resized_points = []
            for box in points:
                resized_box = []
                for point in box:
                    x = int(point[0] * scale_percent / 100)
                    y = int(point[1] * scale_percent / 100)
                    resized_box.append([x, y])
                resized_points.append(resized_box)

            resized_notes = copy.deepcopy(notes)
            for i in range(len(resized_notes)):
                resized_notes[i]['points'] = resized_points[i]

            return resized_image, resized_notes
        elif mode == 'img':
            return resized_image

    @classmethod
    def add_gaussian_noise(cls, image, mode: str = "img", notes: List[Dict] = None, mean=70, std=20):
        noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        if mode == 'notes' and notes is None:
            raise ValueError("When mode is set to 'notes', you must provide notes data.")
        elif mode == 'notes':
            gaussian_notes = copy.deepcopy(notes)
            return noisy_image, gaussian_notes
        elif mode == 'img':
            return noisy_image

    @classmethod
    def add_salt_and_pepper_noise(cls, image, mode: str = "img", notes: List[Dict] = None, salt_ratio=0.01,
                                  pepper_ratio=0.01):
        noisy_image = np.copy(image)
        height, width = image.shape[:2]
        total_pixels = height * width
        num_salt = int(total_pixels * salt_ratio)
        salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
        noisy_image[salt_coords[0], salt_coords[1], :] = 255

        num_pepper = int(total_pixels * pepper_ratio)
        pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
        noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
        if mode == 'notes' and notes is None:
            raise ValueError("When mode is set to 'notes', you must provide notes data.")
        elif mode == 'notes':
            salt_pepper_notes = copy.deepcopy(notes)
            return noisy_image, salt_pepper_notes
        elif mode == 'img':
            return noisy_image

    # @classmethod
    # def add_pepper_and_salt_noise(cls, image, mode: str = "img", notes: List[Dict] = None, pepper_ratio=0.01,
    #                               salt_ratio=0.01):
    #     noisy_image = np.copy(image)
    #     height, width = image.shape[:2]
    #     total_pixels = height * width
    #
    #     num_pepper = int(total_pixels * pepper_ratio)
    #     pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    #     noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    #
    #     num_salt = int(total_pixels * salt_ratio)
    #     salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    #     noisy_image[salt_coords[0], salt_coords[1], :] = 255
    #
    #     if mode == "notes":
    #         salt_pepper_notes = copy.deepcopy(notes)
    #         return noisy_image, salt_pepper_notes
    #     elif mode == 'img':
    #         return noisy_image

    # @classmethod
    # def random_bright(cls, image, mode: str = "img", notes: List[Dict] = None, u=32):
    #     img_np = np.array(image) / 255.0  # 转换为NumPy数组并进行归一化
    #     if np.random.random() > 0.5:
    #         alpha = np.random.uniform(-u, u) / 255
    #         img_np += alpha
    #         img_np = np.clip(img_np, 0.0, 1.0)  # 对像素值进行裁剪，确保在合理范围内
    #     img_np = (img_np * 255).astype(np.uint8)  # 将数组转换为8位整型
    #
    #     if mode == 'notes' and notes is None:
    #         raise ValueError("When mode is set to 'notes', you must provide notes data.")
    #     elif mode == 'notes':
    #         random_bright_notes = copy.deepcopy(notes)
    #         return Image.fromarray(img_np), random_bright_notes
    #     elif mode == 'img':
    #         return Image.fromarray(img_np)

    @classmethod
    def random_bright(cls, image, mode: str = "img", notes: List[Dict] = None, lower=0.5, upper=1.5):
        factor = np.random.uniform(lower, upper)
        bright_img = np.clip(image * factor, 0, 255).astype(np.uint8)
        if mode == "notes":
            random_bright_notes = copy.deepcopy(notes)
            return bright_img, random_bright_notes
        elif mode == 'img':
            return bright_img

    @classmethod
    def random_saturation(cls, image, mode: str = "img", notes: List[Dict] = None, lower=0.5, upper=1.5):
        if np.random.random() > 0.5:
            alpha = np.random.uniform(lower, upper)
            image[:, :, 1] = image[:, :, 1] * alpha
            image[:, :, 1] = np.clip(image[:, :, 1], 0, 255).astype(np.uint8)  # 对像素值进行裁剪，确保在合理范围内
        if mode == 'notes' and notes is None:
            raise ValueError("When mode is set to 'notes', you must provide notes data.")
        elif mode == 'notes':
            random_saturation_notes = copy.deepcopy(notes)
            return image, random_saturation_notes
        elif mode == 'img':
            return image

    @classmethod
    def convert_to_grayscale(cls, image, mode: str = "img", notes: List[Dict] = None):
        image_np = np.array(image)
        grayscale_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        if mode == 'notes' and notes is None:
            raise ValueError("When mode is set to 'notes', you must provide notes data.")
        elif mode == 'notes':
            grayscale_notes = copy.deepcopy(notes)
            return grayscale_image, grayscale_notes
        elif mode == 'img':
            return grayscale_image

    @classmethod
    def random_combination_augment(cls, image_path: str, mode: str = "img", notes: List[Dict] = None):
        aug_image_name = augment_image_filename(image_path=image_path)
        image = cls.image_read(image_path)
        # cv2.imwrite('0.jpg',image)
        augmentations_candidates_func1 = [
            cls.image_rotate_90_counterclockwise, cls.image_rotate_180,
            cls.image_rotate_90_clockwise, cls.resize_image, cls.add_gaussian_noise,
            cls.add_salt_and_pepper_noise
        ]
        augmentations_candidates_func2 = [cls.random_bright, cls.random_saturation]
        augmentations_candidates_func3 = [cls.convert_to_grayscale]

        selected_augmentations1 = random.sample(augmentations_candidates_func1,
                                                k=random.randint(1, len(augmentations_candidates_func1)))
        selected_augmentations2 = random.sample(augmentations_candidates_func2,
                                                k=random.randint(0, len(augmentations_candidates_func2)))
        selected_augmentations3 = random.sample(augmentations_candidates_func3,
                                                k=random.randint(0, len(augmentations_candidates_func3)))

        augmented_image, augmented_notes = image, notes

        if mode == 'notes' and notes is None:
            raise ValueError("When mode is set to 'notes', you must provide notes data.")
        elif mode == 'notes':
            for augmentation in selected_augmentations1 + selected_augmentations2 + selected_augmentations3:
                augmented_image, augmented_notes = augmentation(image=augmented_image, mode=mode,
                                                                notes=augmented_notes)
            # if isinstance(augmented_image, Image.Image):
            #     augmented_image = np.array(augmented_image)

            cv2.imwrite(aug_image_name, augmented_image)
            return aug_image_name, augmented_notes
        elif mode == 'img':
            for augmentation in selected_augmentations1 + selected_augmentations2 + selected_augmentations3:
                print(augmentation)
                augmented_image = augmentation(image=augmented_image, mode=mode)

            # if isinstance(augmented_image, Image.Image):
            #     augmented_image = np.array(augmented_image)

            cv2.imwrite(aug_image_name, augmented_image)
            return aug_image_name


# 获取文件名称
def get_filename_from_path(file_path):
    filename = os.path.basename(file_path)
    return filename


# 获取文件路径
def get_directory_from_path(file_path):
    directory = os.path.dirname(file_path)
    return directory


# 为增强数据命名
def augment_image_filename(image_path: str):
    directory = get_directory_from_path(image_path)
    filename, extension = os.path.splitext(get_filename_from_path(image_path))
    suffix = random.randint(1000, 9999)
    augmented_image_name = f"{filename}_{suffix}{extension}"
    augmented_image_path = os.path.join(directory, augmented_image_name)
    return augmented_image_path


if __name__ == '__main__':

    ImageAugment.random_combination_augment("./1.jpg")
    cv2.imwrite('./5.jpg',ImageAugment.resize_image(ImageAugment.image_read('./1.jpg')))
    cv2.imwrite('./3.jpg',ImageAugment.random_bright(ImageAugment.image_read('./1.jpg')))
    cv2.imwrite('./4.jpg',ImageAugment.random_saturation(ImageAugment.image_read('./1.jpg')))