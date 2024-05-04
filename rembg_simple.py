from typing import Optional, Union, List
import os
import subprocess
import sys
import torch
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont
import numpy as np
from comfy_extras.chainner_models import model_loading
import folder_paths as comfy_paths
MODELS_DIR =  comfy_paths.models_dir


class cstr(str):
    class color:
        END = '\33[0m'
        BOLD = '\33[1m'
        ITALIC = '\33[3m'
        UNDERLINE = '\33[4m'
        BLINK = '\33[5m'
        BLINK2 = '\33[6m'
        SELECTED = '\33[7m'

        BLACK = '\33[30m'
        RED = '\33[31m'
        GREEN = '\33[32m'
        YELLOW = '\33[33m'
        BLUE = '\33[34m'
        VIOLET = '\33[35m'
        BEIGE = '\33[36m'
        WHITE = '\33[37m'

        BLACKBG = '\33[40m'
        REDBG = '\33[41m'
        GREENBG = '\33[42m'
        YELLOWBG = '\33[43m'
        BLUEBG = '\33[44m'
        VIOLETBG = '\33[45m'
        BEIGEBG = '\33[46m'
        WHITEBG = '\33[47m'

        GREY = '\33[90m'
        LIGHTRED = '\33[91m'
        LIGHTGREEN = '\33[92m'
        LIGHTYELLOW = '\33[93m'
        LIGHTBLUE = '\33[94m'
        LIGHTVIOLET = '\33[95m'
        LIGHTBEIGE = '\33[96m'
        LIGHTWHITE = '\33[97m'

        GREYBG = '\33[100m'
        LIGHTREDBG = '\33[101m'
        LIGHTGREENBG = '\33[102m'
        LIGHTYELLOWBG = '\33[103m'
        LIGHTBLUEBG = '\33[104m'
        LIGHTVIOLETBG = '\33[105m'
        LIGHTBEIGEBG = '\33[106m'
        LIGHTWHITEBG = '\33[107m'

        @staticmethod
        def add_code(name, code):
            if not hasattr(cstr.color, name.upper()):
                setattr(cstr.color, name.upper(), code)
            else:
                raise ValueError(f"'cstr' object already contains a code with the name '{name}'.")

    def __new__(cls, text):
        return super().__new__(cls, text)

    def __getattr__(self, attr):
        if attr.lower().startswith("_cstr"):
            code = getattr(self.color, attr.upper().lstrip("_cstr"))
            modified_text = self.replace(f"__{attr[1:]}__", f"{code}")
            return cstr(modified_text)
        elif attr.upper() in dir(self.color):
            code = getattr(self.color, attr.upper())
            modified_text = f"{code}{self}{self.color.END}"
            return cstr(modified_text)
        elif attr.lower() in dir(cstr):
            return getattr(cstr, attr.lower())
        else:
            raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)

def packages(versions=False):
    import sys
    import subprocess
    return [( r.decode().split('==')[0] if not versions else r.decode() ) for r in subprocess.check_output([sys.executable, '-s', '-m', 'pip', 'freeze']).split()]

def install_package(package, uninstall_first: Union[List[str], str] = None):
    if os.getenv("WAS_BLOCK_AUTO_INSTALL", 'False').lower() in ('true', '1', 't'):
        cstr(f"Preventing package install of '{package}' due to WAS_BLOCK_INSTALL env").msg.print()
    else:
        if uninstall_first is None:
            return

        if isinstance(uninstall_first, str):
            uninstall_first = [uninstall_first]

        cstr(f"Uninstalling {', '.join(uninstall_first)}..")
        subprocess.check_call([sys.executable, '-s', '-m', 'pip', 'uninstall', *uninstall_first])
        cstr("Installing package...").msg.print()
        subprocess.check_call([sys.executable, '-s', '-m', 'pip', '-q', 'install', package])

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Ivan:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "transparency": ("BOOLEAN", {"default": True},),
                "post_processing": ("BOOLEAN", {"default": False}),
                "only_mask": ("BOOLEAN", {"default": False},),
                "alpha_matting": ("BOOLEAN", {"default": False},),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255}),
                "alpha_matting_erode_size": ("INT", {"default": 10, "min": 0, "max": 255}),
                "background_color": (["none", "black", "white", "magenta", "chroma green", "chroma blue"],),
            },
        }

    RETURN_TYPES = ("IMAGE","IMAGE")
    RETURN_NAMES = ("u2net_images","u2net_human_images")
    FUNCTION = "image_rembg"

    CATEGORY = "rembg_simple"

    def __convertToBool(self, x):

        # Evaluate string representation of False types
        if type(x) == str:
            x = x.strip()
            if (x.lower() == 'false'
                or x.lower() == 'none'
                or x == '0'
                or x == '0.0'
                or x == '0j'
                or x == "''"
                or x == '""'
                or x == "()"
                or x == "[]"
                or x == "{}"
                or x.lower() == "decimal(0)"
                or x.lower() == "fraction(0,1)"
                or x.lower() == "set()"
                or x.lower() == "range(0)"
            ):
                return False
            else:
                return True

        # Anything else will be evaluated by the bool function
        return bool(x)

    def image_rembg(
        self,
        images,
        model="u2net",
        transparency=True,
        alpha_matting=False,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
        post_processing=False,
        only_mask=False,
        background_color="none",
        # putalpha = False,
    ):

        # ComfyUI will allow strings in place of booleans, validate the input.
        transparency = transparency if type(transparency) is bool else self.__convertToBool(transparency)
        alpha_matting = alpha_matting if type(alpha_matting) is bool else self.__convertToBool(alpha_matting)
        post_processing = post_processing if type(post_processing) is bool else self.__convertToBool(post_processing)
        only_mask = only_mask if type(only_mask) is bool else self.__convertToBool(only_mask)

        if "rembg" not in packages():
            install_package("rembg")

        from rembg import remove, new_session

        os.environ['U2NET_HOME'] = os.path.join(MODELS_DIR, 'rembg')
        os.makedirs(os.environ['U2NET_HOME'], exist_ok=True)
            
        # Set bgcolor
        bgrgba = None
        if background_color == "black":
            bgrgba = [0, 0, 0, 255]
        elif background_color == "white":
            bgrgba = [255, 255, 255, 255]
        elif background_color == "magenta":
            bgrgba = [255, 0, 255, 255]
        elif background_color == "chroma green":
            bgrgba = [0, 177, 64, 255]
        elif background_color == "chroma blue":
            bgrgba = [0, 71, 187, 255]
        else:
            bgrgba = None

        if transparency and bgrgba is not None:
            bgrgba[3] = 0

        batch_tensor1 = []
        batch_tensor2 = []
        for image in images:
            image = tensor2pil(image)
            batch_tensor1.append(pil2tensor(
                remove(
                    image,
                    session=new_session("u2net"),
                    post_process_mask=post_processing,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size,
                    only_mask=only_mask,
                    bgcolor=bgrgba,
                    # putalpha = putalpha,
                )
                .convert(('RGBA' if transparency else 'RGB'))))
        batch_tensor1 = torch.cat(batch_tensor1, dim=0)

        for image in images:
            image = tensor2pil(image)
            batch_tensor2.append(pil2tensor(
                remove(
                    image,
                    session=new_session("u2net_human_seg"),
                    post_process_mask=post_processing,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                    alpha_matting_erode_size=alpha_matting_erode_size,
                    only_mask=only_mask,
                    bgcolor=bgrgba,
                    # putalpha = putalpha,
                )
                .convert(('RGBA' if transparency else 'RGB'))))
        batch_tensor2 = torch.cat(batch_tensor2, dim=0)

        

        return (batch_tensor1,batch_tensor2)



# NODE MAPPING
NODE_CLASS_MAPPINGS = {

    "rembg_simple": Ivan
}
