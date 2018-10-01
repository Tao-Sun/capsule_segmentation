import python.baseline.models.unet_hippo as unet_hippo
import python.baseline.models.unet_hippo_1 as unet_hippo_1
import python.baseline.models.unet_hippo_2 as unet_hippo_2
import python.baseline.models.unet_hippo_3 as unet_hippo_3
import python.baseline.models.unet_hippo_5 as unet_hippo_5
import python.baseline.models.unet_hippo_6 as unet_hippo_6
import python.baseline.models.unet_mnist_1 as unet_mnist_1
import python.baseline.models.unet_mnist_2 as unet_mnist_2
import python.baseline.models.unet_mnist_3 as unet_mnist_3
import python.baseline.models.unet_mnist_4 as unet_mnist_4
import python.baseline.models.unet_affnist_1 as unet_affnist_1


def get_model(name):
    if name == 'hippo':
        return unet_hippo
    elif name == 'hippo1':
        return unet_hippo_1
    elif name == 'hippo2':
        return unet_hippo_2
    elif name == 'hippo3':
        return unet_hippo_3
    elif name == 'hippo5':
        return unet_hippo_5
    elif name == 'hippo6':
        return unet_hippo_6
    elif name == 'mnist1':
        return unet_mnist_1
    elif name == 'mnist2':
        return unet_mnist_2
    elif name == 'mnist3':
        return unet_mnist_3
    elif name == 'mnist4':
        return unet_mnist_4
    elif name == 'affnist1':
        return unet_affnist_1

