import warnings
import numpy as np
import matplotlib
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras import models

matplotlib.use('tkagg')


class SudokuReader:
    def __init__(self, path_img, path_clf, debug=False):
        self.input_image = np.empty((1, 1, 3), dtype=np.uint8)
        self.input_edges = np.empty((1, 1), dtype=np.uint8)
        self.sudoku_img = np.empty((1, 1), dtype=np.uint8)
        self.sudoku_gray = np.empty((1, 1), dtype=np.uint8)
        self.sudoku_binary = np.empty((1, 1), dtype=np.uint8)
        self.lines = np.empty((1, 1), dtype=np.uint8)
        self.read_image_from_source(path_img)

        self.height_img = 0  # nb rows input image
        self.width_img = 0  # nb columns input image
        self.x0_sudoku = 0  # x coord of upper left pt of sudoku in input image
        self.y0_sudoku = 0  # y coord of upper left pt of sudoku in input image
        self.height_sudoku = 0  # nb rows sudoku image
        self.width_sudoku = 0  # nb columns sudoku image
        self.side_sudoku = 0  # nb rows = nb cols in rectified image
        self.number_candidates = []

        self.sudoku_field = np.zeros((9, 9), dtype=np.uint8)
        self.number_classifier = None
        self.load_model(path_clf)

        self.debug = debug  # show images in debug mode

    @staticmethod
    def order_rectangle_points(poly_candidate):
        pts_unsorted = np.reshape(poly_candidate, (4, 2))
        pts_sorted = np.empty((4, 2))
        coord_sum = np.sum(pts_unsorted, axis=1)
        pts_sorted[0] = pts_unsorted[np.argmin(coord_sum)]
        pts_sorted[2] = pts_unsorted[np.argmax(coord_sum)]
        coord_diff = np.diff(pts_unsorted, axis=1)
        pts_sorted[1] = pts_unsorted[np.argmax(coord_diff)]
        pts_sorted[3] = pts_unsorted[np.argmin(coord_diff)]
        return pts_sorted

    def read_image_from_source(self, path_src):
        self.input_image = cv.imread(path_src)
        long_side = max(self.input_image.shape[0], self.input_image.shape[1])
        scale_factor = 800 / long_side
        self.input_image = cv.resize(self.input_image, dsize=(0, 0), fx=scale_factor, fy=scale_factor)
        self.height_img, self.width_img = self.input_image.shape[0], self.input_image.shape[1]
        return None

    def load_model(self, path_model):
        self.number_classifier = models.load_model(path_model)
        return None

    def show_all_images(self):
        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs[0, 0].imshow(self.input_image)
        axs[0, 0].set_title('Input image')
        axs[0, 1].imshow(self.input_edges)
        axs[0, 1].set_title('Edge image')
        axs[1, 0].imshow(self.sudoku_gray, cmap='gray')
        axs[1, 0].set_title('Sudoku gray')
        axs[1, 1].imshow(self.sudoku_binary, cmap='gray')
        axs[1, 1].set_title(label='Sudoku binary')
        fig.suptitle('All (relevant) images')
        plt.show()
        return None

    def show_candidates(self):
        drawing = self.sudoku_img.copy()
        for candidate in self.number_candidates:
            drawing = cv.putText(drawing, str(candidate['number']),
                                 (int(candidate['x_center']), int(candidate['y_center'])),
                                 fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 0), thickness=4)
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(self.sudoku_img)
        axs[0].set_title('Sudoku contour')
        axs[1].imshow(drawing)
        axs[1].set_title('Detected numbers')
        plt.show()
        return None

    def show_solution_on_sudoku(self, field):
        drawing = self.sudoku_img.copy()
        step = self.side_sudoku // 9
        delta = self.side_sudoku // 36

        for row in range(0, 9):
            for col in range(0, 9):
                drawing = cv.putText(drawing, str(field[row, col]), (col * step + delta, (row + 1) * step - delta),
                                     fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=3, color=(255, 0, 0), thickness=3)
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].imshow(self.input_image)
        axs[0].set_title('Input image')
        axs[1].imshow(drawing)
        axs[1].set_title('One possible solution')
        plt.show()
        return None

    def compute_binary_image(self, gaussian_kernel_size=5, thres=1.0, block_size=5):
        blur = cv.GaussianBlur(self.sudoku_gray, (gaussian_kernel_size, gaussian_kernel_size), sigmaX=3.0)
        self.sudoku_binary = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
                                                  blockSize=block_size, C=thres)
        return None

    def otsu_thresholding(self, kernel_size=7):
        blur = cv.GaussianBlur(self.sudoku_gray, (kernel_size, kernel_size), 0)
        _, self.sudoku_binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        return None

    def open_image(self, kernel_shape=cv.MORPH_RECT, kernel_size=3):
        kernel = cv.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
        self.sudoku_binary = cv.morphologyEx(self.sudoku_binary, cv.MORPH_OPEN, kernel)
        return None

    def close_image(self, kernel_shape=cv.MORPH_RECT, kernel_size=5):
        kernel = cv.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
        self.sudoku_binary = cv.morphologyEx(self.sudoku_binary, cv.MORPH_CLOSE, kernel)
        return None

    def canny_edge_detection(self, kernel_size=5, thres_low=100, thres_upper=200):
        input_gray = cv.cvtColor(self.input_image, cv.COLOR_BGR2GRAY)
        input_blur = cv.GaussianBlur(input_gray, ksize=(kernel_size, kernel_size), sigmaX=1.0)
        self.input_edges = cv.Canny(input_blur, thres_low, thres_upper)
        return None

    def rectify_image_sudoku(self, source_pts):
        source_pts = np.array(source_pts, dtype=np.float32)
        a = np.array([0, 0])
        b = np.array([0, self.side_sudoku])
        c = np.array([self.side_sudoku, self.side_sudoku])
        d = np.array([self.side_sudoku, 0])
        destination_pts = np.array([a, b, c, d], dtype=np.float32)

        invtf = cv.getPerspectiveTransform(source_pts, destination_pts)
        self.sudoku_img = cv.warpPerspective(self.input_image, invtf, dsize=(self.side_sudoku, self.side_sudoku))
        return None

    def find_contour_sudoku(self, side_length):
        self.canny_edge_detection()
        contours, _ = cv.findContours(self.input_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        for candidate in contours:
            perimeter_candidate = cv.arcLength(candidate, True)
            poly_candidate = cv.approxPolyDP(candidate, epsilon=0.05 * perimeter_candidate, closed=True)
            if len(poly_candidate) == 4:
                x0, y0, w, h = [int(i) for i in cv.boundingRect(poly_candidate)]
                self.x0_sudoku = x0
                self.y0_sudoku = y0
                self.height_sudoku = h
                self.width_sudoku = w
                source_pts = self.order_rectangle_points(poly_candidate)
                self.side_sudoku = max(w, h, side_length)
                self.rectify_image_sudoku(source_pts)
                self.sudoku_gray = cv.cvtColor(self.sudoku_img, cv.COLOR_BGR2GRAY)
                # debug
                if self.debug:
                    output = self.input_image.copy()
                    fig, axs = plt.subplots(nrows=1, ncols=2)
                    for i in range(0, 4):
                        cv.circle(output, (int(source_pts[i][0]), int(source_pts[i][1])), 3, (0, 255, 0))
                    axs[0].imshow(output)
                    axs[0].set_title('Four points contour')
                    axs[1].imshow(self.sudoku_img)
                    axs[1].set_title('Cropped sudoku contour')
                    plt.show()
                # debug end
                return True
        print('No contour found which corresponds to a possible sudoku square.')
        return False

    def find_candidates(self):
        self.otsu_thresholding()
        self.close_image()
        nb_labels, labels, stats, centroids = cv.connectedComponentsWithStats(self.sudoku_binary, connectivity=8,
                                                                              ltype=cv.CV_32S)
        # debug
        if self.debug:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(self.sudoku_binary, cmap='gray')
            axs[0].set_title('Binary image')
            axs[1].imshow(labels)
            axs[1].set_title('Connected components')
            plt.show()

            output = self.sudoku_img.copy()
            for i in range(0, nb_labels):
                if self.is_candidate_size_realistic(stats[i]):
                    x = stats[i, cv.CC_STAT_LEFT]
                    y = stats[i, cv.CC_STAT_TOP]
                    w = stats[i, cv.CC_STAT_WIDTH]
                    h = stats[i, cv.CC_STAT_HEIGHT]
                    cx = centroids[i, 0]
                    cy = centroids[i, 1]
                    cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv.circle(output, (int(cx), int(cy)), 1, (255, 0, 0), 3)
            plt.imshow(output)
            plt.title('Candidates found')
            plt.show()
        # end debug
        for i in range(0, nb_labels):
            if self.is_candidate_size_realistic(stats[i]):
                self.number_candidates.append({'stats': stats[i],
                                               'x_center': centroids[i, 0],
                                               'y_center': centroids[i, 1]})
            else:
                continue

        if self.number_candidates:
            return True
        else:
            return False

    def is_candidate_size_realistic(self, stats):
        # assuming 1/20 * 1/81 * s**2 <= A_cand <= 1/81 * s**2
        # s being the side length of the sudoku square
        area_total = self.side_sudoku * self.side_sudoku
        w = stats[cv.CC_STAT_WIDTH]
        h = stats[cv.CC_STAT_HEIGHT]
        area_cand = w * h
        if h / w < 1.0 / 3.0 or 3.0 < h / w:
            return False
        elif area_cand / area_total < 0.0005 or 0.01 < area_cand / area_total:
            return False
        else:
            return True

    def crop_candidate(self, stats):
        x = stats[cv.CC_STAT_LEFT]
        y = stats[cv.CC_STAT_TOP]
        w = stats[cv.CC_STAT_WIDTH]
        h = stats[cv.CC_STAT_HEIGHT]
        s = max(w, h)
        x = x + w // 2 - s // 2
        y = y + h // 2 - s // 2
        # adding 20 percent of length at each side
        delta_s = s // 5
        x_left = max(x - delta_s, 0)
        x_right = min(x + delta_s + s, self.side_sudoku)
        y_up = max(y - delta_s, 0)
        y_down = min(y + delta_s + s, self.side_sudoku)
        img_cand = self.sudoku_gray[y_up:y_down, x_left:x_right]
        _, img_thres = cv.threshold(img_cand, 140, 255, cv.THRESH_BINARY_INV)
        img_thres = cv.resize(img_thres, dsize=(28, 28))
        img_thres = img_thres.astype(np.float32) / 255.0
        # debug
        if self.debug:
            fig, axs = plt.subplots(nrows=1, ncols=2)
            axs[0].imshow(img_cand, cmap='gray')
            axs[0].set_title('Candidate')
            axs[1].imshow(img_thres, cmap='gray')
            axs[1].set_title('Thres opencv')
            plt.show()
        # end debug
        return img_thres

    def get_position_in_sudoku(self, x_center, y_center, dist_x, dist_y):
        # check if contour shape might be much larger than sudoku field, which can lead to mapping error
        if 8 * dist_x < 7 * self.side_sudoku or 8 * dist_y < 7 * self.side_sudoku:
            warnings.warn('Possible wrong mapping of candidates to sudoku field.')
        idx_x = int(9 * x_center / self.side_sudoku)
        idx_y = int(9 * y_center / self.side_sudoku)
        assert idx_x < 9 and idx_y < 9, 'Computed index is too large; number does not fit into sudoku.'
        return idx_x, idx_y

    def fill_in_numbers(self):
        x_coords = np.array([candidate['stats'][cv.CC_STAT_LEFT] for candidate in self.number_candidates])
        y_coords = np.array([candidate['stats'][cv.CC_STAT_TOP] for candidate in self.number_candidates])
        dist_x = np.max(x_coords) - np.min(x_coords)
        dist_y = np.max(y_coords) - np.min(y_coords)
        for candidate in self.number_candidates:
            img_cand = self.crop_candidate(candidate['stats'])
            candidate_probs = self.number_classifier.predict(np.reshape(img_cand, (1, 28, 28, 1)))
            candidate_nb = np.argmax(candidate_probs)
            candidate['number'] = candidate_nb
            idx_x, idx_y = self.get_position_in_sudoku(candidate['x_center'], candidate['y_center'], dist_x, dist_y)
            self.sudoku_field[idx_y, idx_x] = candidate_nb
            # debug
            if self.debug:
                plt.imshow(img_cand)
                plt.title('Predicted nb {}'.format(candidate_nb))
                plt.show()
            # end debug
        return None

    def get_sudoku_field_from_image(self):
        if self.find_contour_sudoku(side_length=500):
            self.compute_binary_image(thres=2.3, block_size=5)
            if self.find_candidates():
                self.fill_in_numbers()
                if self.debug:
                    self.show_candidates()
                return True
            else:
                print('Found no numbers in sudoku.')
                fig, axs = plt.subplots(nrows=1, ncols=2)
                axs[0].imshow(self.input_image)
                axs[0].set_title('Input image')
                axs[1].imshow(self.sudoku_img)
                axs[1].set_title('Sudoku field found')
                return False
        else:
            print('Found no sudoku field in image.')
            plt.imshow(self.input_image)
            plt.title('Input image')
            return False
