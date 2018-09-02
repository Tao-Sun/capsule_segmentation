import python.baseline.models.unet_hippo as unet_hippo
import python.baseline.models.unet_hippo_1 as unet_hippo_1

def get_model(name):
    if name == 'hippo':
        return unet_hippo
    elif name == 'hippo1':
        return unet_hippo_1
