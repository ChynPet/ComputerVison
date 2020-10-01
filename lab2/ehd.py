import cv2
import numpy as np
import time

class EHD:

    def __init__(self, num_blocks_h, num_blocks_w, th_edge):
        self.num_blocks_h = num_blocks_h
        self.num_blocks_w = num_blocks_w
        self.th_edge = th_edge

        self.ver_edge_filter = np.array([[1, -1], [1, -1]], dtype=np.float64)
        self.hor_edge_filter = np.array([[1, 1], [-1, -1]], dtype=np.float64)
        self.dia45_edge_filter = np.array([[np.sqrt(2), 0], [0, -np.sqrt(2)]], dtype=np.float64)
        self.dia135_edge_filter = np.array([[0, np.sqrt(2)], [-np.sqrt(2), 0]], dtype=np.float64)
        self.nond_edge_filter = np.array([[2, -2], [-2, 2]], dtype=np.float64)

        self.filter_sizes = 2
        self.num_sub_image_h = 4
        self.num_sub_image_w = 4

        self.matrix_descriptor = None
        self.matrix_mean = None
        self.matrix_conv_image = None

        self.X_train_name = None
        self.local_X_train = None
        self.y_train = None

        self.X_test_name = None
        self.local_X_test = None
        self.y_test = None

        self.time_train = None
        self.time_test = None
        self.y_predict = None
        self.list_score = None
        self.score = None

    # block filtering function
    def filtering(self, image_block, filter):

        height, width = (int(i/2) for i in image_block.shape)
        sub_block0 = image_block[0:height, 0:width] * filter[0,0]
        sub_block1 = image_block[0:height, width:] * filter[0,1]
        sub_block2 = image_block[height:, 0:width] * filter[1,0]
        sub_block3 = image_block[height:, width:] * filter[1,1]
        res = sub_block0 + sub_block1 + sub_block2 + sub_block3

        return res

    # function for determining the type of edge in the image block
    def type_edge(self, image_block):

        mat_ver_edg = self.filtering(image_block, self.ver_edge_filter)
        mat_hor_edg = self.filtering(image_block, self.hor_edge_filter)
        mat_dia45_edg = self.filtering(image_block, self.dia45_edge_filter)
        mat_dia135_edg = self.filtering(image_block, self.dia135_edge_filter)
        mat_nond_edg = self.filtering(image_block, self.nond_edge_filter)

        mat = np.array([mat_ver_edg, mat_hor_edg, mat_dia45_edg, mat_dia135_edg, mat_nond_edg])
        m = np.max(np.abs(np.mean(mat, axis=(1, 2))))
        arg = np.argmax(np.mean(mat, axis=(1, 2)))

        if m > self.th_edge:
            return (mat[arg], m, arg+1)

        return (np.zeros(mat_ver_edg.shape), 0, 0)

    # creates descriptor
    def descriptor(self, image):

        height_block = int(image.shape[0] / self.num_blocks_h)
        width_block = int(image.shape[1] / self.num_blocks_w)

        if height_block % 2 != 0:
            height_block -= 1
        if width_block % 2 != 0:
            width_block -= 1

        self.matrix_descriptor = np.zeros((self.num_blocks_h, self.num_blocks_w), dtype=np.int)
        self.matrix_mean = np.zeros((self.num_blocks_h, self.num_blocks_w))

        height_conv_image = int(height_block / 2)
        width_conv_image = int(width_block / 2)
        self.matrix_conv_image = np.zeros((height_conv_image * self.num_blocks_h, width_conv_image * self.num_blocks_w))

        for i_im_block in np.arange(self.num_blocks_h):
            for j_im_block in np.arange(self.num_blocks_w):
                h_im_block_start = i_im_block * height_block
                w_im_block_start = j_im_block * width_block
                h_im_block_end = h_im_block_start + height_block
                w_im_block_end = w_im_block_start + width_block

                h_conv_im_start = i_im_block * height_conv_image
                h_conv_im_end = h_conv_im_start + height_conv_image
                w_conv_im_start = j_im_block * width_conv_image
                w_conv_im_end = w_conv_im_start + width_conv_image

                image_block = image[h_im_block_start:h_im_block_end, w_im_block_start:w_im_block_end]

                mat_conv_image, mean, type_edge = self.type_edge(image_block)

                self.matrix_descriptor[i_im_block, j_im_block] = type_edge
                self.matrix_mean[i_im_block, j_im_block] = mean
                self.matrix_conv_image[h_conv_im_start:h_conv_im_end, w_conv_im_start:w_conv_im_end] = mat_conv_image

        return self

    # calculates histograms in sub images
    def compute_histogram(self, descriptor):

        h_sub_im = int(self.num_blocks_h / self.num_sub_image_h)
        w_sub_im = int(self.num_blocks_w / self.num_sub_image_w)
        local_im = np.zeros((self.num_sub_image_h, self.num_sub_image_w, 5))

        for i in np.arange(self.num_sub_image_h):
            for j in np.arange(self.num_sub_image_w):
                h_start = i * h_sub_im
                w_start = j * w_sub_im
                h_end = h_start + h_sub_im
                w_end = w_start + w_sub_im

                hist = np.histogram(descriptor[h_start:h_end, w_start:w_end], bins=6)
                local_im[i, j] = hist[0][1:] / np.sum(hist[0])

        return local_im

    # calculates the distances between histograms sub-images
    def distance(self, local_A, local_B):

        list_dist = []
        for i in range(local_A.shape[0]):
            for j in range(local_A.shape[1]):
                list_dist.append( cv2.compareHist(local_A[i,j].astype(np.float32), local_B[i,j].astype(np.float32), cv2.HISTCMP_CHISQR))

        return list_dist

    def train(self, train_images):

        self.local_X_train = []

        self.y_train = []
        self.X_train_name = []
        self.time_train = []

        for name_image, target in train_images.items():
            im = cv2.imread(name_image, 0)

            start_time = time.time()
            descriptor = self.descriptor(im).matrix_descriptor
            local_hist = self.compute_histogram(descriptor)
            end_time = time.time()

            self.local_X_train.append(local_hist)
            self.y_train.append(target)
            self.X_train_name.append(name_image)
            self.time_train.append(np.round(end_time - start_time, 2))

        return self

    def test(self, test_images, threshold=0.5):

        self.local_X_test = []
        self.X_test_name = []
        self.y_test = []

        self.y_predict = []
        self.score = []
        self.list_score = []
        self.time_test = []
        for name_image, target in test_images.items():

            im = cv2.imread(name_image, 0)

            start_time = time.time()
            descriptor = self.descriptor(im).matrix_descriptor
            local_hist = self.compute_histogram(descriptor)
            end_time = time.time()
            list_dist = []

            start_reg = time.time()
            for l in self.local_X_train:
                locals_dist = self.distance(l, local_hist)
                list_dist.append(locals_dist)

            min_dist = np.min(np.mean(list_dist, axis=0))

            if min_dist < threshold:
                self.y_predict.append(1)
            else:
                self.y_predict.append(0)
            end_reg = time.time()

            arg = np.argmin(np.mean(list_dist, axis=1))
            self.list_score.append(list_dist[int(arg)])

            self.local_X_test.append(local_hist)
            self.X_test_name.append(name_image)
            self.y_test.append(target)

            self.score.append(min_dist)
            self.time_test.append((np.round(end_time - start_time, 7), np.round(end_reg - start_reg, 7)))

        return self
