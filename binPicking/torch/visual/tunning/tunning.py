import utils
import torch
import torch.utils
import transforms as T

from .engine import train_one_epoch, evaluate

from .dataset import binPickingDataset
from .models import get_MaskRCNN

DATA_ROOT = ''
NUM_CLASS = 2
TOTAL_EPOCH = 10

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    ###############################################################
    # Datast and DataLoader
    ###############################################################
    dataset      = binPickingDataset(DATA_ROOT, get_transform(train=True))
    dataset_test = binPickingDataset(DATA_ROOT, get_transform(train=False))

    torch.manual_seed(1)

    indices = torch.randperm(len(dataset)).tolist
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        collate_fn=utils.collate_fn
    )
    

    ###############################################################
    # Model
    ###############################################################

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = NUM_CLASS

    # get the model using our helper function
    model = get_MaskRCNN(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    
    for epoch in range(TOTAL_EPOCH):
        train_one_epoch(
            model,
            optimizer,
            data_loader,
            device,
            epoch,
            print_freq=10
        )
        lr_scheduler.step()
        evaluate(
            model,
            data_loader_test,
            device = device
        )
    torch.save(model.state_dict, 'tuned_model')

if __name__=="__main__":
    main()