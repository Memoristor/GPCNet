# coding=utf-8

from datasets.base_datasets import ClassificationDataset

__all__ = ["CIFAR10Dataset", "CIFAR100Dataset"]


class CIFAR10Dataset(ClassificationDataset):
    """
    The CIFAR-10 dataset (Canadian Institute for Advanced Research, 10 classes) is a subset
    of the Tiny Images dataset and consists of 60000 32x32 color images. The images are
    labelled with one of 10 mutually exclusive classes: airplane, automobile (but not truck
    or pickup truck), bird, cat, deer, dog, frog, horse, ship, and truck (but not pickup
    truck). There are 6000 images per class with 5000 training and 1000 testing images per class.

    Reference:
    https://paperswithcode.com/dataset/cifar-10
    """

    CLASSES = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    def __init__(
        self,
        image_mean=(0.491, 0.482, 0.447),
        image_std=(0.247, 0.243, 0.262),
        **kwargs,
    ):
        super(CIFAR10Dataset, self).__init__(image_mean=image_mean, image_std=image_std, **kwargs)


class CIFAR100Dataset(ClassificationDataset):
    """
    The CIFAR-100 dataset (Canadian Institute for Advanced Research, 100 classes) is a subset of
    the Tiny Images dataset and consists of 60000 32x32 color images. The 100 classes in the CIFAR-100
    are grouped into 20 superclasses. There are 600 images per class. Each image comes with a "fine"
    label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).
    There are 500 training images and 100 testing images per class.

    Reference:
    https://paperswithcode.com/dataset/cifar-100
    """

    SUPER_CLASSES = {
        "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
        "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
        "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
        "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        "household_electrical_devices": [
            "clock",
            "keyboard",
            "lamp",
            "telephone",
            "television",
        ],
        "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
        "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
        "large_man-made_outdoor_things": [
            "bridge",
            "castle",
            "house",
            "road",
            "skyscraper",
        ],
        "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
        "large_omnivores_and_herbivores": [
            "camel",
            "cattle",
            "chimpanzee",
            "elephant",
            "kangaroo",
        ],
        "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
        "non-insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
        "people": ["baby", "boy", "girl", "man", "woman"],
        "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
    }

    CLASSES = ()

    def __init__(
        self,
        image_suffix="*.png",
        image_mean=(0.507, 0.487, 0.441),
        image_std=(0.267, 0.256, 0.276),
        **kwargs,
    ):
        super(CIFAR100Dataset, self).__init__(
            image_suffix=image_suffix,
            image_mean=image_mean,
            image_std=image_std,
            **kwargs,
        )

        # Get CLASSES
        sub_classes = list()
        for v in self.SUPER_CLASSES.values():
            sub_classes.extend(v)
        self.CLASSES = tuple(sub_classes)

        # get the number of classes
        self.num_classes = len(self.CLASSES)
