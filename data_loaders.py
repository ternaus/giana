from torch.utils.data import Dataset
import cv2


class GianaDataset(Dataset):
    def __init__(self, file_names: list, transform=None):
        self.file_paths = file_names
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_file_path = self.file_paths[idx]
        image = load_image(image_file_path)

        file_name = image_file_path.name

        mask_file_path = image_file_path.parent.parent / 'mask' / file_name

        mask = load_mask(mask_file_path)

        data = {"image": image, "mask": mask}

        augmented = self.transform(**data)

        image, mask = augmented["image"], augmented["mask"]

        return image, mask


def load_image(path):
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask = cv2.imread(str(path), 0)
    return mask
