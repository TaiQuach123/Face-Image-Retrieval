from lib import *

class SiameseModel(nn.Module):
    def __init__(self, backbone='InceptionResnetV1'):
        super().__init__()
        if backbone=='InceptionResnetV1':
            self.backbone = InceptionResnetV1(pretrained='vggface2')
    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x, dim=1)
        return x

if __name__ == "__main__":
    model = SiameseModel()
    rand_img = torch.rand((5,3,160,160))
    print(model(rand_img).shape)
