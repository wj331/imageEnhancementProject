import torch
from ..model.retinex import create_model
from ..model.retinex.utils.options import parse
import time
import numpy as np


#continue bringing over into local model, then enhance image
def enhance_image_retinexformer(img_np):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize the image
    img_np = img_np / 255.0

    # Convert the image to a tensor
    img_tensor = torch.from_numpy(img_np).float()
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor.to(device).unsqueeze(0)

    #load the retinexformer
    opt = parse("C:/Users/wenji/OneDrive/Desktop/Y3S2/ATAP/Image Enhancement Project/image-enhancement-app/backend/services/retinexOptions/RetinexFormer_LOL_v2_synthetic.yml", is_train= False)
    opt['dist'] = False

    model_restoration = create_model(opt).net_g

    # Load pre-trained weights
    checkpoint = torch.load("C:/Users/wenji/OneDrive/Desktop/Y3S2/ATAP/Image Enhancement Project/image-enhancement-app/backend/services/retinexPretrainedWeights/LOL_v2_synthetic.pth", map_location="cpu")
    model_restoration.load_state_dict(checkpoint['params'], strict=True)

    # Set model to evaluation mode
    model_restoration.eval()
    with torch.no_grad():
        start = time.time()
        enhanced_image = model_restoration(img_tensor)
        end_time = (time.time() - start)
        print(f"Enhancement time: {end_time}")

    enhanced_image_np = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_image_np = np.clip(enhanced_image_np * 255, 0, 255).astype(np.uint8)
    return enhanced_image_np
