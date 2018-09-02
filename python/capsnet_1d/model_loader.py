import python.capsnet_1d.models.capsnet_1d_hippo as capsnet_1d_hippo
import python.capsnet_1d.models.capsnet_1d_hippo_1 as capsnet_1d_hippo_1

def get_model(name):
    if name == 'hippo':
        return capsnet_1d_hippo
    elif name == 'hippo1':
        return capsnet_1d_hippo_1
