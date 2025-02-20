from torch.utils.data import Dataset


__all__ = ['cifar10_mean', 'cifar10_std', 'cifar100_mean', 'cifar100_std', 'normal_mean', 'normal_std',
           'DatasetWithIndices', 'TensorDatasetWithIndices', 'Unnormalizer']
cifar10_mean = (0.491400, 0.482158, 0.4465231)
cifar10_std = (0.247032, 0.243485, 0.2615877)
cifar100_mean = (0.507075, 0.486549, 0.440918)
cifar100_std = (0.267334, 0.256438, 0.276151)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

class DatasetWithIndices(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return (data, target), index

    def __len__(self):
        return len(self.dataset)

class TensorDatasetWithIndices(Dataset):
    def __init__(self, tensor_dataset):
        self.data_tensor = tensor_dataset.tensors[0]
        self.target_tensor = tensor_dataset.tensors[1]

    def __getitem__(self, index):
        data = self.data_tensor[index]
        target = self.target_tensor[index]
        return (data, target), index

    def __len__(self):
        return self.data_tensor.size(0)
    
       
class Unnormalizer:
    def __init__(self, mean=(0.5,), std=(0.5,)):
        """
        Initializes the Unnormalizer class with mean and standard deviation.
        
        Args:
            mean (tuple): The mean used in the normalization. Default is (0.5,).
            std (tuple): The standard deviation used in the normalization. Default is (0.5,).
        """
        self.mean = mean
        self.std = std

    def unnormalize(self, tensor):
        """
        Unnormalizes a tensor using the instance's mean and standard deviation.
        
        Args:
            tensor (Tensor): Normalized image tensor.
        
        Returns:
            Tensor: Unnormalized image tensor.
        """
        # Unnormalize the tensor: (tensor * std) + mean
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

    def set_mean_std(self, mean, std):
        """
        Sets new mean and standard deviation for unnormalizing tensors.
        
        Args:
            mean (tuple): New mean to set.
            std (tuple): New standard deviation to set.
        """
        self.mean = mean
        self.std = std
        
        
        

# from PIL import Image
# import os
# import os.path
# import numpy as np
# import pickle
# from typing import Any, Callable, Optional, Tuple

# from torchvision.datasets import VisionDataset
# from torchvision.datasets.utils import check_integrity, download_and_extract_archive

# ''' 
# This class is slightly revised to fit the multiple inheritance scheme in `NoiseDataset`.
# The only difference from the official torchvision is that `super()...` is removed.
# '''

# class CIFAR10(VisionDataset):
#     """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
#     Args:
#         root (string): Root directory of dataset where directory
#             ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
#         train (bool, optional): If True, creates dataset from training set, otherwise
#             creates from test set.
#         transform (callable, optional): A function/transform that takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#     """
#     base_folder = 'cifar-10-batches-py'
#     url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
#     filename = "cifar-10-python.tar.gz"
#     tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
#     train_list = [
#         ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
#         ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
#         ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
#         ['data_batch_4', '634d18415352ddfa80567beed471001a'],
#         ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
#     ]

#     test_list = [
#         ['test_batch', '40351d587109b95175f43aff81a1287e'],
#     ]
#     meta = {
#         'filename': 'batches.meta',
#         'key': 'label_names',
#         'md5': '5ff9c542aee3614f3951f8cda6e48888',
#     }

#     def __init__(
#             self,
#             root: str,
#             train: bool = True,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             download: bool = False,
#     ) -> None:
#         # import ipdb; ipdb.set_trace()
#         self.train = train  # training set or test set
#         # self.root = root
#         if download:
#             self.download()

#         if not self._check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You can use download=True to download it')

#         if self.train:
#             downloaded_list = self.train_list
#         else:
#             downloaded_list = self.test_list

#         self.data: Any = []
#         self.targets = []

#         # now load the picked numpy arrays
#         for file_name, checksum in downloaded_list:
#             file_path = os.path.join(self.root, self.base_folder, file_name)
#             with open(file_path, 'rb') as f:
#                 entry = pickle.load(f, encoding='latin1')
#                 self.data.append(entry['data'])
#                 if 'labels' in entry:
#                     self.targets.extend(entry['labels'])
#                 else:
#                     self.targets.extend(entry['fine_labels'])

#         self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
#         self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

#         self._load_meta()

#     def _load_meta(self) -> None:
#         path = os.path.join(self.root, self.base_folder, self.meta['filename'])
#         if not check_integrity(path, self.meta['md5']):
#             raise RuntimeError('Dataset metadata file not found or corrupted.' +
#                                ' You can use download=True to download it')
#         with open(path, 'rb') as infile:
#             data = pickle.load(infile, encoding='latin1')
#             self.classes = data[self.meta['key']]
#         self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is index of the target class.
#         """
#         img, target = self.data[index], self.targets[index]

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self) -> int:
#         return len(self.data)

#     def _check_integrity(self) -> bool:
#         for fentry in (self.train_list + self.test_list):
#             filename, md5 = fentry[0], fentry[1]
#             fpath = os.path.join(self.root, self.base_folder, filename)
#             if not check_integrity(fpath, md5):
#                 return False
#         return True

#     def download(self) -> None:
#         if self._check_integrity():
#             print('Files already downloaded and verified')
#             return
#         download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

#     def extra_repr(self) -> str:
#         return "Split: {}".format("Train" if self.train is True else "Test")


# class CIFAR100(CIFAR10):
#     """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
#     This is a subclass of the `CIFAR10` Dataset.
#     """
#     base_folder = 'cifar-100-python'
#     url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
#     filename = "cifar-100-python.tar.gz"
#     tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
#     train_list = [
#         ['train', '16019d7e3df5f24257cddd939b257f8d'],
#     ]

#     test_list = [
#         ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
#     ]
#     meta = {
#         'filename': 'meta',
#         'key': 'fine_label_names',
#         'md5': '7973b15100ade9c7d40fb424638fde48',
#     }


# class NoiseCIFAR10(CIFAR10):

#     def __init__(
#         self,
#         root: str = "/home/murong/data",
#         train: bool = True,
#         transform=None,
#         download=True,
#         gaussian_std: float = 0.0,
#         gaussian_perturb_rate: float = 0.0,
#     ) -> None:

#         self.root = root
#         self.transform = transform
#         self.target_transform = None
#         self.gaussian_std = gaussian_std

#         CIFAR10.__init__(self,
#                          root=root,
#                          train=train,
#                          transform=transform,
#                          download=download)
        
#         self.hard_indices = np.random.choice(
#             len(self.data), int(gaussian_perturb_rate * len(self.data)), replace=False)
        
        
#     # def add_gaussian_noise(self, image: torch.Tensor) -> torch.Tensor:
#     #     """Applies Gaussian noise to the image."""
#     #     if self.gaussian_std > 0:  # Only add noise if std is greater than 0
#     #         noise = torch.randn(image.size()) * (self.gaussian_std * torch.max(image))
#     #         noisy_image = image + noise
#     #         # noisy_image = torch.clamp(noisy_image, 0, 1)  # Ensure pixel values stay in valid range
#     #         return noisy_image
#     #     return image  # No noise added
    
#     def add_gaussian_noise_to_raw_image(self, image: np.ndarray) -> np.ndarray:
#         """
#         Applies Gaussian noise to the original image before transformation.
#         The image is expected to be in the range [0, 255].
#         """
#         if self.gaussian_std > 0:  # Only add noise if std is greater than 0
#             noise = np.random.normal(0, self.gaussian_std, image.shape)  # Gaussian noise
#             noise = noise * np.max(image)  # Scale noise to the range of the image
#             noisy_image = image + noise
#             noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values stay in [0, 255] range
#             return noisy_image.astype(np.uint8)  # Convert back to unsigned 8-bit integer (uint8)
#         return image  # No noise added
    
    
#     def __getitem__(self, index):
#         image, target = self.data[index], self.targets[index]
#         image = Image.fromarray(image)
        
#         if index in self.hard_indices:
#             image = self.add_gaussian_noise_to_raw_image(np.array(image))
#             image = Image.fromarray(image)

#         if self.transform is not None:
#             image = self.transform(image)
            
#         if self.target_transform is not None:
#             target = self.target_transform(target)
            
#         return image, target
    