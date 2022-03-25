
import cv2
import numpy as np
import time


np.seterr(divide='ignore',invalid='ignore')
para = [
        # [3, 0.5],
        # [5, 1.0],
        # [7, 2.0],
        # [9, 2.0],
        # [11,3.0],
        # [13,3.8],
        # [15,4.2],
        # [17,4.8],
        [19,5.0],
        [21,6.0],
        [23,8.0],
        [25,9.0]
        ]


def normalization(data):
    isnan = np.isnan(data)
    data[isnan] = 0
    data[isnan] = np.mean(data)
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def SQI(input_img):
    # input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = np.zeros(input_img.shape, np.float)
    img_in = input_img.astype(np.float)
    size = len(para)
    scale_idx = 0
    for i in range(size):
        scale_idx += 1
        hsize = para[i][0]
        sigma = para[i][1]
        a = time.time()
        img_smo = cv2.GaussianBlur(img_in, (hsize,hsize), sigma)
        QI_cur = img_in / img_smo
        QI_cur = 1 / (1 + np.exp(-QI_cur))
        QI_cur = 255 * normalization(QI_cur)
        QI_cur = QI_cur.astype(np.uint8).astype(np.float)
        output_img = output_img + QI_cur

    
    output_img = output_img / scale_idx
    output_img = output_img.astype(np.uint8)
    return output_img

if __name__ == '__main__':
    sample_id = 0
    exposure = 6
    pic_num = 1
    src_file = '../data/sample_' + str(sample_id) + '/{:03d}'.format(exposure) + '/' + str(pic_num) + '.png'
    img_in = cv2.imread(src_file)
    st = time.time()
    img_out = SQI(img_in)
    ed = time.time()
    print("Cost:{:.5f}".format(ed-st))

    # cv2.imshow("result", img_out )
    # cv2.waitKey(0)