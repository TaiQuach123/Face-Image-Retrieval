from lib import *
from model import SiameseModel
from utils import get_img_names
import h5py
import os
import argparse
    
def face_vecs_embedding(args):
    '''Store embedding vectors of face images'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hf = h5py.File(args.fname, 'w')
    
    # Load model weights
    print('Loading model weights........', end='')
    model = SiameseModel()
    if device == 'cuda':
        model.load_state_dict(torch.load(args.model_weights))
    elif device == 'cpu':
        model.load_state_dict(torch.load(args.model_weights, map_location=torch.device('cpu')))
        
        
    model.to(device)
    model.eval();
    print('Done')
    

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_names = get_img_names(args.dataset_dir)
    
    for img_name in tqdm(img_names):
        path = os.path.join(args.dataset_dir, img_name)
        img = Image.open(path)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        result = model(img)

        result = result.cpu().detach().numpy()
        # Store feature vectors
        hf.create_dataset(img_name, data= result)

    hf.close()
    print(f'Saved face embedding vectors: {args.fname}')


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')
    
    parser.add_argument('--dataset-dir', default='./img_align_celeba',
                        help='Path to image directory')
    parser.add_argument('--model-weights', default='model.pth',
                        help='Path to model weights')
    parser.add_argument('--fname', default='face_vecs.h5',
                        help='Name of face embedding file')

    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    face_vecs_embedding(args)
    