from PIL import Image
import unidecode
from jellyfish import levenshtein_distance

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import time
import os
import numpy as np


class Reader:
    def __init__(self, weights):
        # list tinh & tp co dau
        self.tinh_list = ['An Giang', 'Bà Rịa - Vũng Tàu', 'Bắc Giang', 'Bắc Kạn', 'Bạc Liêu', 'Bắc Ninh', 'Bến Tre', 'Bình Định', 'Bình Dương', 'Bình Phước', 'Bình Thuận',
            'Cà Mau', 'Cao Bằng', 'Đắk Lắk', 'Đắk Nông', 'Điện Biên', 'Đồng Nai', 'Đồng Tháp', 'Gia Lai', 'Hà Giang', 'Hà Nam', 'Hà Tĩnh', 'Hải Dương', 'Hậu Giang', 'Hòa Bình', 'Hưng Yên',
            'Khánh Hòa', 'Kiên Giang', 'Kon Tum', 'Lai Châu', 'Lâm Đồng', 'Lạng Sơn', 'Lào Cai', 'Long An', 'Nam Định', 'Nghệ An', 'Ninh Bình', 'Ninh Thuận', 'Phú Thọ',
            'Quảng Bình', 'Quảng Nam', 'Quảng Ngãi', 'Quảng Ninh', 'Quảng Trị', 'Sóc Trăng', 'Sơn La', 'Tây Ninh', 'Thái Bình', 'Thái Nguyên', 'Thanh Hóa', 'Thừa Thiên Huế', 'Tiền Giang',
            'Trà Vinh', 'Tuyên Quang', 'Vĩnh Long', 'Vĩnh Phúc', 'Yên Bái', 'Phú Yên', 'Cần Thơ', 'Đà Nẵng', 'Hải Phòng', 'Hà Nội', 'TP Hồ Chí Minh']
        # list tinh & tp khong co dau
        self.provinces = [self.remove_accent(tinh).lower() for tinh in self.tinh_list]

        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['weights'] = weights
        self.config['cnn']['pretrained'] = False
        self.config['device'] = 'cpu'
        self.config['predictor']['beamsearch'] = False

        self.reader = Predictor(self.config)

    def read(self, image):
        """
        Recognise text from image
        :param image: ndarray of image
        :return: text
        """
        text = self.reader.predict(image)

        return text

    def remove_accent(self, text):

        return unidecode.unidecode(text)

    def postprocess_address(self, original_text, thresold):
        # preprocess text
        text = self.remove_accent(original_text)
        text = text.lower()

        # calculate editance between text with each of address in provinces list
        edits = [levenshtein_distance(text, address) for address in self.provinces]
        edits = np.array(edits)
        arg_min = np.argmin(edits)

        if edits[arg_min] < thresold:
            return self.tinh_list[arg_min]
        else:
            return original_text



if __name__ == "__main__":
    transformer_weight = os.path.join("./models", "transformerocr_new.pth")

    reader = Reader(transformer_weight)

    file_path = os.path.join("./cropped_images", "place_of_birth_272.jpg")
    img = Image.open(file_path)
    start = time.time()
    result = reader.read(img)
    final_result = reader.postprocess_address(result, 5)

    print("result: ", final_result)
    print("time:", time.time() - start)

