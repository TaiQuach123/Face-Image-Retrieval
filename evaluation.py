from lib import *
from model import SiameseModel
from loss import SupConLoss
from PIL import Image
from tqdm import tqdm
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser('Siamese Network Traning For Face Embedding')
    parser.add_argument('--partition_path', default='./list_eval_partition.txt',
                      help='Partition file used for create train-val-test')
    parser.add_argument('--identity_path', default='./identity_CelebA.txt',
                      help = 'file contains label for each image')
    parser.add_argument('--dataset_dir', default='./img_align_celeba/')
    parser.add_argument('--img_size', default=(160,160),
                        help='img size usage')
    parser.add_argument('--k',default=9)
    args = parser.parse_args()
    return args
def evaluate_PK(test_dataloader, model, index, k=3):
    PreK=[]
    d=1
    for paths, labels in test_dataloader:
        _labels=labels.tolist()

        prek=[]
        for stt, _path in enumerate(paths):
            path=_path.split('\t')[0]
            query = model(val_transforms(Image.open(path)).unsqueeze(0)).detach().numpy()
            D, I = index.search(query, k=k)
            yes_labels=0
            for i, idx in enumerate(I[0]):
                retrieved_image_path = test_paths[idx][0]
                label = test_paths[idx][1]
                if label==_labels[stt]:
                    yes_labels+=1
            if len(test_dct[str(_labels[stt])]) <k :
                prek.append(yes_labels/len(test_dct[str(_labels[stt])]))
            else:
                prek.append(yes_labels/k)
        PreK.append(sum(prek)/len(prek))
        print("Batch {} P@K (P@{}):______ {}".format(d,k,round(sum(prek)/len(prek),2)))
        d=d+1

    return PreK


def evaluate_APK(test_dataloader,model, index, k=3):
    meanAPK=[]
    d=1
    for paths, labels in test_dataloader:
        _labels=labels.tolist()

        APK=[]
        for kk in range(1,k+1):
            prek=[]
            for stt, _path in enumerate(paths):
                path=_path.split('\t')[0]
                query = model(val_transforms(Image.open(path)).unsqueeze(0)).detach().numpy()
                D, I = index.search(query, k=kk)
                yes_labels=0
                for i, idx in enumerate(I[0]):
                    retrieved_image_path = test_paths[idx][0]
                    label = test_paths[idx][1]
                    if label==_labels[stt]:
                        yes_labels+=1
                if len(test_dct[str(_labels[stt])]) <kk :
                    prek.append(yes_labels/len(test_dct[str(_labels[stt])]))
                else:
                    prek.append(yes_labels/kk)
                APK.append(sum(prek)/len(prek))

        print("Batch {} AP@K (AP@{}):______ {}".format(d,k,round(sum(APK)/len(APK),2)))
        d=d+1
        meanAPK.append(round(sum(APK)/len(APK),2))

    return meanAPK


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

    test_dataset= MyDataset(dct=dct, split='test')

    test_dataloader= DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    val_transforms = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = SiameseModel()
    criterion = SupConLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model.load_state_dict(torch.load('./model_best_weights_1000.pt'))
    model.to('cuda')
    model.eval();

    test_paths=[]
    for i in test_dataset:
      test_paths.append((i[0].split('\t')[0], i[1]))

    os.mkdir('./test_embeddings')


    test_store = []
    i = 0
    for (img_path, identity) in test_paths:
        embedding = model(val_transforms(Image.open(img_path)).unsqueeze(0).to('cuda')).cpu().detach().numpy().squeeze(0)
        test_store.append(embedding)
        if len(test_store) == 500:
            embeddings = np.vstack(test_store)
            np.save('./test_embeddings/embeddings{}.npy'.format(i), embeddings)
            i += 1
            test_store = []

    if len(test_store) > 0:
        embeddings = np.vstack(test_store)
        np.save('./test_embeddings/embeddings{}.npy'.format(i), embeddings)
        
    model.to('cpu');
    i=i+1

    filenames = ['./test_embeddings/embeddings{}.npy'.format(j) for j in range(i)]
    test_final_embeddings = np.vstack([np.load(filenames[j]) for j in range(i)])
    np.save('./test_embeddings/final_embeddings.npy', test_final_embeddings)


    index = faiss.IndexFlatIP(512)
    index.add(test_final_embeddings)

    PreK=evaluate(test_dataloader,model,index,k=args.k)
  


