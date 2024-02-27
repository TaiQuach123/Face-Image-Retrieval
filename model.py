from lib import *

class SiameseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2')
    def forward(self, x):
        x = self.backbone(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return x
