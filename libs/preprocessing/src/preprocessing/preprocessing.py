from PIL import ImageFile
import torchvision.transforms as T
from torch import Tensor


def input_transform(image_size: tuple[float, float] = (322, 322)):
        """Pipeline for image transforming for preprocessing, image sides sizes is needed to be divisable by 14
        Returns function that can transform input images 
        """
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(image_size, antialias=True)
        ])

class Preprocessor():
    def __init__(self, image_size: tuple[float, float] = (322, 322)) -> None:
        self.transform = input_transform(image_size=image_size)
        
    def get_preprocessed_image(self, img: ImageFile, unsqueeze: bool = True) -> Tensor:
        """Main function for image preprocessing, use usqueeze if the batch forming is needed 
        Returns transformed Image Tensor that is ready for input to the VPR model (MegaLoc) 
        """
        if unsqueeze:  
            return self.transform(img).unsqueeze(0)
        else:
            return self.transform(img)