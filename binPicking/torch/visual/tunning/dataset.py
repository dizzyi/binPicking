import os 
import numpy as np
import torch 
import torch.utils.data
from PIL import Image

class binPickingDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms = None):
        self.root = root
        self.trasforms = transforms
        self.imgs = list(
            sorted(
                os.listdir(
                    os.path.join(
                        root,
                        "PNGImages"
                    )
                )
            )
        )
        self.masks = list(
            sorted(
                os.listdir(
                    os.path.join(
                        root,
                        "Masks"
                    )
                )
            )
        )

    def __getitem__(self, index):
        # define the path to image and mask, and load them
        img_path = os.path.join(self.root,"PNGImages", self.imgs[index])
        mask_path = os.path.join(self.root,"Masks", self.masks[index])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)

        boxes = []

        for i in range(num_objs):
            pos = np.where(mask[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin,ymin,xmax,ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,),dtype=torch.int64)
        masks = torch.as_tensor(mask, dtype = torch.uint8)

        image_id = torch.tensor([index])

        area = (boxes[:,3]-boxes[:.1])*(boxes[:,2]-boxes[:,0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target['boxes']    = boxes
        target['lables']   = labels
        target['masks']    = masks
        target['image_id'] = image_id
        target['area']     = area
        target['iscrowd']  = iscrowd

        if self.trasforms is not None:
            img, target  = self.trasforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.imgs)