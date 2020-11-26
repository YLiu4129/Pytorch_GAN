import os
from PIL import Image
from torch.utils.data import Dataset

class dataset(Dataset):

    def __init__(self, root, transform=None):
        # absolute path
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transform


    # return specific image
    def __getitem__(self, index):

        img_path = self.imgs[index]
        pil_img = Image.open(img_path)

        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.array(pil_img)
            # create a tensor from array
            data = torch.from_numpy(pil_img)

        return data

    def __len__(self):

        return len(self.imgs)

