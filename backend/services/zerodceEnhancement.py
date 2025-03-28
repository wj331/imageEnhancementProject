import time
import torch
import numpy as np
from ..model import zerodce
import os

def enhance_image_zerodce(img_np):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize the image
    img_np = img_np / 255.0

    # Convert the image to a tensor
    img_tensor = torch.from_numpy(img_np).float()
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor.to(device).unsqueeze(0)

    # Load the Zero-DCE model
    DCE_net = zerodce.enhance_net_nopool().to(device)

    #load weights
    weight_path = r'C://Users//wenji//OneDrive//Desktop//Y3S2//ATAP//Image Enhancement Project//image-enhancement-app//backend//services//dceweights//best5.0.pth'
    # weight_path = r'C://Users//wenji//OneDrive//Desktop//Y3S2//ATAP//Image Enhancement Project//image-enhancement-app//backend//services//snapshots//Epoch199.pth'

    DCE_net.load_state_dict(torch.load(weight_path, map_location = device))
    DCE_net.eval()

    with torch.no_grad():
        start = time.time()
        _, enhanced_image, _ = DCE_net(img_tensor)
        end_time = (time.time() - start)
        print(f"Enhancement time: {end_time}")

    enhanced_image_np = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced_image_np = np.clip(enhanced_image_np * 255, 0, 255).astype(np.uint8)
    return enhanced_image_np