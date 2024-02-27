from lib import *
# import dataset
#import model


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


