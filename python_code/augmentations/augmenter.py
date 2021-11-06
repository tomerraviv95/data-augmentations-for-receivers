from python_code.augmentations.augmenter1 import Augmenter1
from python_code.augmentations.augmenter2 import Augmenter2
from python_code.augmentations.augmenter3 import Augmenter3


class Augmenter:
    @staticmethod
    def augment(current_received, current_transmitted, type, h, snr):
        if type == 'reg':
            x, y = current_received, current_transmitted.reshape(1, -1)
        elif type == 'aug1':
            x, y = Augmenter1.augment(current_transmitted.reshape(1, -1), h, snr)
        elif type == 'aug2':
            x, y = Augmenter2.augment(current_received, current_transmitted.reshape(1, -1), h)
        elif type == 'aug3':
            x, y = Augmenter3.augment(current_received, current_transmitted)
        else:
            raise ValueError("No sucn augmentation method!!!")

        return x, y
