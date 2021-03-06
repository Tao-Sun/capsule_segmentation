import python.capsnet_1d.models.capsnet_1d_hippo as capsnet_1d_hippo
import python.capsnet_1d.models.capsnet_1d_hippo_1 as capsnet_1d_hippo_1
import python.capsnet_1d.models.capsnet_1d_hippo_2 as capsnet_1d_hippo_2
import python.capsnet_1d.models.capsnet_1d_hippo_3 as capsnet_1d_hippo_3
import python.capsnet_1d.models.capsnet_1d_hippo_4 as capsnet_1d_hippo_4
import python.capsnet_1d.models.capsnet_1d_hippo_5 as capsnet_1d_hippo_5
import python.capsnet_1d.models.capsnet_1d_hippo_6 as capsnet_1d_hippo_6
import python.capsnet_1d.models.capsnet_1d_hippo_7 as capsnet_1d_hippo_7
import python.capsnet_1d.models.capsnet_1d_hippo_8 as capsnet_1d_hippo_8

import python.capsnet_1d.models.capsnet_1d_mnist_1 as capsnet_1d_mnist_1
import python.capsnet_1d.models.capsnet_1d_mnist_2 as capsnet_1d_mnist_2
import python.capsnet_1d.models.capsnet_1d_mnist_3 as capsnet_1d_mnist_3
import python.capsnet_1d.models.capsnet_1d_mnist_4 as capsnet_1d_mnist_4
import python.capsnet_1d.models.capsnet_1d_mnist_5 as capsnet_1d_mnist_5


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
    elif name == 'hippo5':
        return capsnet_1d_hippo_5
    elif name == 'hippo6':
        return capsnet_1d_hippo_6
    elif name == 'hippo7':
        return capsnet_1d_hippo_7
    elif name == 'hippo8':
        return capsnet_1d_hippo_8
    if name == 'mnist1':
        return capsnet_1d_mnist_1
    elif name == 'mnist2':
        return capsnet_1d_mnist_2
    elif name == 'mnist3':
        return capsnet_1d_mnist_3
    elif name == 'mnist4':
        return capsnet_1d_mnist_4
    elif name == 'mnist5':
        return capsnet_1d_mnist_5
