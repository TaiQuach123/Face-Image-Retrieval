from dataset import *
from model import SiameseModel
from loss import SupConLoss
from PIL import Image
from tqdm import tqdm
import numpy as np

import argparse


def parse_arg():
    parser = argparse.ArgumentParser('Siamese Network Traning For Face Embedding')
    parser.add_argument('--partition_path', default='./list_eval_partition.txt',
                      help='Partition file used for create train-val-test')
    parser.add_argument('--identity_path', default='./identity_CelebA.txt',
                      help = 'file contains label for each image')
    parser.add_argument('--dataset_dir', default='./img_align_celeba/')
    parser.add_argument('--img_size', default=(160,160),
                        help='img size usage')
    parser.add_argument('--batch_size', default=64,
                        help='batch size usage')
    parser.add_argument('--lr', default=1e-4,
                        help='learning rate for optimizer')
    args = parser.parse_args()
    return args


def training(model, dataloader, val_dataloader, optimizer, criteria, epochs, device):
  model.to(device)
  best_val_loss = np.inf
  for epoch in range(1, epochs+1):
    model.train()
    total_train_loss = 0
    for img_paths, labels in tqdm(dataloader):
      augmented_imgs = []
      for img in img_paths:
        img = img.split('\t')
        augmented_imgs.append(train_transforms(Image.open(img[0])))
        augmented_imgs.append(train_transforms(Image.open(img[0])))
        augmented_imgs.append(train_transforms(Image.open(img[1])))
      augmented_imgs = torch.stack(augmented_imgs, dim = 0)
      labels = labels.repeat_interleave(3)

      imgs = augmented_imgs.to(device)
      labels = labels.to(device)

      features = model(imgs).unsqueeze(1)
      optimizer.zero_grad()
      loss = criteria(features, labels)
      loss.backward()
      optimizer.step()
      total_train_loss += loss.item()
    print('Epoch {}: Loss = {}'.format(epoch, total_train_loss/len(dataloader)))
    #loss_train.append(total_train_loss/len(dataloader))


    model.eval()
    total_val_loss = 0
    for img_paths, labels in val_dataloader:
      with torch.no_grad():
        augmented_imgs = []
        for img in img_paths:
          img = img.split('\t')
          augmented_imgs.append(train_transforms(Image.open(img[0])))
          augmented_imgs.append(train_transforms(Image.open(img[0])))
          augmented_imgs.append(train_transforms(Image.open(img[1])))
        augmented_imgs = torch.stack(augmented_imgs, dim = 0)
        labels = labels.repeat_interleave(3)

        imgs = augmented_imgs.to(device)
        labels = labels.to(device)

        features = model(imgs).unsqueeze(1)
        val_loss = criteria(features, labels)
        total_val_loss += val_loss.item()
    print('Evaluate: Val Loss = {}'.format(total_val_loss/len(val_dataloader)))
    #loss_val.append(total_val_loss/len(val_dataloader))

    if total_val_loss < best_val_loss:
      best_val_loss = total_val_loss
      torch.save(model.state_dict(), 'model_best_weights_1000.pt')

if __name__ == "__main__":

    args = parse_arg()

    train_paths, val_paths, test_paths = create_train_val_test_paths(partition_path=args.partition_path)

    paths = {
      'train': train_paths,
      'val': val_paths,
      'test': test_paths
    }
    train_dct, val_dct, test_dct = create_train_val_test_dct(paths = paths, dataset_dir = args.dataset_dir, identity_path=args.identity_path)

    dct = {'train': train_dct, 'val': val_dct, 'test': test_dct}

    train_dataset = MyDataset(dct=dct)
    val_dataset = MyDataset(dct=dct, split='val')
    test_dataset= MyDataset(dct=dct, split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader= DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)



    train_transforms = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.AutoAugment(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(args.img_size[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = SiameseModel()
    criterion = SupConLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    training(model, train_dataloader, val_dataloader, optimizer, criterion, 3, device)