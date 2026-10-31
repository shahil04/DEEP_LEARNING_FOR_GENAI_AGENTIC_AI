
from siamese_model import SiameseNetwork
from utils import preprocess
import torch
import torch.nn.functional as F

def predict_similarity(img1, img2):
    model = SiameseNetwork()
    model.load_state_dict(torch.load("models/siamese_model.pth", map_location=torch.device('cpu')))
    model.eval()

    img1 = preprocess(img1)
    img2 = preprocess(img2)

    with torch.no_grad():
        output1, output2 = model(img1, img2)
        distance = F.pairwise_distance(output1, output2)
    return 1.0 - distance.item()
