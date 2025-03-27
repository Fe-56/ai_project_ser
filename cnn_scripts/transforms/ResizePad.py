import torchvision.transforms.functional as F


class ResizePad(object):
    def __init__(self, size=(224, 224), fill=0, padding_mode='constant'):
        """
        Args:
            size (tuple): Desired output size as (height, width).
            fill (int): Pixel fill value for constant padding.
            padding_mode (str): Padding mode ('constant', 'edge', etc.).
        """
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # Get original dimensions (width, height)
        orig_width, orig_height = img.size
        target_height, target_width = self.size

        # Calculate scale factor to fit image inside target dimensions
        scale = min(target_width / orig_width, target_height / orig_height)
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        # Resize the image while preserving aspect ratio
        img = F.resize(img, (new_height, new_width))

        # Calculate padding values to center the image
        pad_left = (target_width - new_width) // 2
        pad_top = (target_height - new_height) // 2
        pad_right = target_width - new_width - pad_left
        pad_bottom = target_height - new_height - pad_top

        # Apply padding
        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom),
                    fill=self.fill, padding_mode=self.padding_mode)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(size={self.size}, fill={self.fill}, padding_mode={self.padding_mode})'
