# from __future__ import print_function, division
from dataloader import image_preprocess as prep
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]], dtype=np.uint8)) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)


class ImageListWithIndex(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader, need_make_dataset=True):
        """

        :param image_list:
        :param labels:
        :param transform:
        :param target_transform:
        :param loader:
        """
        if need_make_dataset:
            imgs = make_dataset(image_list, labels)
        else:
            imgs = image_list
        if len(imgs) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + image_list)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)


def load_images(images_file_path, batch_size, resize_size=256, is_train=True, crop_size=224, test_sample_ratio=1.0):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if not is_train:
        start_center = (resize_size - crop_size - 1) / 2
        transformer = transforms.Compose([
            prep.ResizeImage(resize_size),
            prep.PlaceCrop(crop_size, start_center, start_center),
            transforms.ToTensor(),
            normalize])
        image_lines = open(images_file_path).readlines()
        if test_sample_ratio < 1.0:
            sample_line = int(len(image_lines) * test_sample_ratio)
            print("sample ratio: %.3f, ori: %.3f, sample: %.3f" % (test_sample_ratio, len(image_lines), sample_line))
            image_lines = np.random.choice(image_lines, int(len(image_lines) * test_sample_ratio), replace=False)
        else:
            print("no sample ori: %.3f" % (len(image_lines)))
        images = ImageListWithIndex(image_lines, transform=transformer)
        images_loader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        transformer = transforms.Compose([prep.ResizeImage(resize_size),
                                          transforms.RandomResizedCrop(crop_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
        image_lines = open(images_file_path).readlines()
        if test_sample_ratio < 1.0:
            assert "we shouldn't sample train set!"
            # sample_line = int(len(image_lines) * test_sample_ratio)
            # image_lines = np.random.choice(image_lines, sample_line, replace=False)
        images = ImageListWithIndex(image_lines, transform=transformer)
        images_loader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=4)
    return images_loader


def inverted_imgs(imgs):
    total_class_num = len(imgs[0][1])
    inverts = [[] for _ in range(total_class_num)]
    for item in imgs:
        for i, c in enumerate(item[1]):
            if c == 1:
                inverts[i].append(item)
    return inverts


def load_balance_images(images_file_path, batch_size, resize_size=256, is_train=True, crop_size=224):
    if not is_train:
        return load_images(images_file_path, batch_size, resize_size=resize_size, is_train=is_train, crop_size=crop_size)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformer = transforms.Compose([prep.ResizeImage(resize_size),
                                          transforms.RandomResizedCrop(crop_size),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])
        # images = ImageBalanceList(open(images_file_path).readlines(), transform=transformer)
        # images = ImageList(open(images_file_path).readlines(), transform=transformer)
        # images_loader = torch.utils.data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=4,
        #                                             drop_last=True)

        imgs = make_dataset(open(images_file_path).readlines(), None)
        inverts = inverted_imgs(imgs)
        image_loaders = []
        half_batch = batch_size // 2
        for invert in inverts:
            # print("class num: %d" % len(invert))
            invert_len = len(invert)
            i_class_images = ImageListWithIndex(invert, transform=transformer, need_make_dataset=False)
            loader = torch.utils.data.DataLoader(i_class_images, batch_size=min(invert_len, half_batch), shuffle=True, num_workers=4,
                                                 drop_last=True)
            image_loaders.append(loader)
    return image_loaders

