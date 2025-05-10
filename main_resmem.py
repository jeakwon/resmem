from models.pretrained import ResNet18CIFAR100, ResNet50CIFAR100
from models.retrained import ResNet18CIFAR99, ResNet50CIFAR99
from dataset import cifar100_dataloaders
from utils import NormalizeByChannelMeanStd


model = ResNet18CIFAR99()
model.normalize = NormalizeByChannelMeanStd(mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762])
train_full_loader, val_loader, _ = cifar100_dataloaders(batch_size=256, data_dir='test_dir', num_workers=4)
marked_loader, _, test_loader = cifar100_dataloaders(batch_size=256, data_dir='test_dir', num_workers=4,
    class_to_replace=[0], num_indexes_to_replace=None, indexes_to_replace=None, seed=2,
    only_mark=True, shuffle=True,no_aug=True)

print(model)
print(train_full_loader, val_loader, marked_loader, test_loader)

# from evaluation.linear_probing import get_class_wise_lp_acc
# class_wise_lp_acc = get_class_wise_lp_acc(model, train_loader, test_loader, criterion, device, num_classes, num_epochs=10, lr=0.001)