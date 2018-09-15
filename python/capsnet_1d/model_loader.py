import python.capsnet_1d.models.capsnet_1d_hippo as capsnet_1d_hippo
import python.capsnet_1d.models.capsnet_1d_hippo_1 as capsnet_1d_hippo_1
import python.capsnet_1d.models.capsnet_1d_hippo_2 as capsnet_1d_hippo_2
import python.capsnet_1d.models.capsnet_1d_hippo_3 as capsnet_1d_hippo_3
import python.capsnet_1d.models.capsnet_1d_hippo_4 as capsnet_1d_hippo_4
import python.capsnet_1d.models.capsnet_1d_mnist_1 as capsnet_1d_mnist_1


def get_model(name):
    if name == 'hippo':
        return capsnet_1d_hippo
    elif name == 'hippo1':
        return capsnet_1d_hippo_1
    elif name == 'hippo2':
        return capsnet_1d_hippo_2
    elif name == 'hippo3':
        return capsnet_1d_hippo_3
    elif name == 'hippo4':
        return capsnet_1d_hippo_4
    elif name == 'mnist_1':
        return capsnet_1d_mnist_1
