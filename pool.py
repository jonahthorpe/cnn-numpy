import numpy as np
import cv2


class Pool:
    # TODO add different pool types

    def forward(self, image):
        """
        Down sample input via a 2x2 max pool
        :param numpy.ndarray image: numpy array of shape (height, width, feature maps)
        :return numpy.ndarray image: numpy array of shape (height, width, feature maps)
        """
        self.input = image

        height, width, channels = image.shape
        output = np.zeros((height // 2, width // 2, channels))

        for h in range(height // 2):
            for w in range(width // 2):
                im_region = image[(h * 2):(h * 2 + 2), (w * 2):(w * 2 + 2)]
                output[h, w] = np.amax(im_region, axis=(0, 1))

        return output

    def backward(self, d_loss_d_out, step):
        """
        Performs the backward pass of the convolution layer.
        :param numpy.ndarray d_loss_d_out: derivative of loss w.r.t the output of the forward pass. Same shape as the
        output of the forward pass
        :param float step: learning rate used in weight change. only here to keep params same for backward pass
        :return numpy.ndarray d_loss_d_input: derivative of loss w.r.t the input of the forward pass. Same shape as the
        input
        """
        d_loss_d_input = np.zeros(self.input.shape)
        # get every 2x2 region of the input image
        height, width, feature_maps = self.input.shape
        for h in range(height // 2):
            for w in range(width // 2):
                im_region = self.input[(h * 2):(h * 2 + 2), (w * 2):(w * 2 + 2)]
                im_region_h, im_region_w, _ = im_region.shape
                # get the max value from the region
                amax = np.amax(im_region, axis=(0, 1))

                # get the position of each value in the region
                for h2 in range(im_region_h):
                    for w2 in range(im_region_w):
                        for feature_map in range(feature_maps):
                            # if the value is the max value, find its position in the original input, and update output
                            if im_region[h2, w2, feature_map] == amax[feature_map]:
                                d_loss_d_input[h * 2 + h2, w * 2 + w2, feature_map] = d_loss_d_out[h, w, feature_map]

        return d_loss_d_input


if __name__ == '__main__':
    i1 = [
          [[2], [3], [1], [9]],
          [[4], [7], [3], [5]],
          [[8], [2], [2], [2]],
          [[1], [3], [4], [5]]
          ]
    image = cv2.imread("lenna.png")
    pool = Pool()
    print(pool.forward(np.array(i1)))

