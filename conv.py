import numpy as np
import cv2


class Conv:
    # TODO add different filter sizes
    # TODO add different stride sizes
    # TODO add bias

    def __init__(self, input_channels, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with dimensions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(input_channels, num_filters, 3, 3)/input_channels

    def forward(self, input):
        """
        Performs a forward pass of the conv layer on the inputted image.
        :param numpy.ndarray input: numpy array of shape (height, width, channels) or (height, width)
        :return numpy.ndarray output: numpy array of shape (height, width, filters)
        """
        input = self.pad_image(input)
        self.input = input
        height, width, channels = input.shape
        output = np.zeros((height - 2, width - 2, self.num_filters))

        # slide left to right, top to bottom
        for h in range(height - 2):
            for w in range(width - 2):
                # convolve over every channel
                for channel in range(channels):
                    # get the region of the image you are working on
                    im_region = input[h:(h + 3), w:(w + 3), channel]
                    # apply filter
                    output[h, w] += np.sum(im_region * self.filters[channel], axis=(1, 2))

        return output

    def backward(self, d_loss_d_out, step):
        """
        Performs the backward pass of the convolution layer.
        :param numpy.ndarray d_loss_d_out: derivative of loss w.r.t the output of the forward pass. Same shape as the
        output of the forward pass
        :param float step: learning rate used in weight change
        :return numpy.ndarray d_loss_d_input: derivative of loss w.r.t the input of the forward pass. Same shape as the
        input
        """
        channels, _, height_f, width_f = self.filters.shape
        height_l, width_l, filters = d_loss_d_out.shape
        d_loss_d_filters = np.zeros(self.filters.shape)
        d_loss_d_input = np.zeros(self.input.shape)

        # calculate the derivative of the loss w.r.t the filers
        # convolution between the input data and derivative of the loss w.r.t to the output
        for h_f in range(height_f):
            for w_f in range(width_f):
                for c in range(channels):
                    im_region = self.input[h_f: h_f + height_l, w_f: w_f + width_l, c]
                    for f in range(filters):
                        loss_region = d_loss_d_out[0: height_l, 0: width_l, f]
                        # de/df = de/do * do/df
                        d_loss_d_filters[c, f, h_f, w_f] += np.sum(loss_region * im_region)

        # calculate the derivative of the loss w.r.t the input
        # a full convolution between the derivative of the loss w.r.t to the output and filters rotated 180 degrees
        # pad with 0 so its a full convolution
        padded_loss = np.pad(d_loss_d_out, ((2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
        # iterate over the height/width of the loss + filter
        for h in range(height_l + height_f - 1):
            for w in range(width_l + width_f - 1):
                for c in range(channels):
                    # get the slice of data from the loss
                    loss_region = padded_loss[h: h + height_f, w: w + width_f]
                    # rotate filter
                    rotated_filter = np.rot90(np.rot90(self.filters[c], axes=(1, 2)), axes=(1, 2))
                    # print(np.sum(np.sum((np.rollaxis(im_region, 2, 0) * self.filters[c]), axis=(1, 2))))
                    # find convolution of the filter and loss
                    d_loss_d_input[h, w, c] = np.sum((np.rollaxis(loss_region, 2, 0) * rotated_filter))

        # Update filters
        self.filters -= step * d_loss_d_filters
        return d_loss_d_input

    def pad_image(self, image):
        return self.same_pad(image)

    def same_pad(self, image):
        """
        Performs same padding on the input image.
        :param numpy.ndarray image: numpy array of shape (height, width, channels) or (height, width)
        :return numpy.ndarray padded_image: numpy array of shape (height, width, channels)
        """
        # check for 2d array
        if len(image.shape) == 2:  # if 2d resize
            height, width = image.shape
            image.resize(height, width, 1)
        return np.pad(image, ((1, 1), (1, 1), (0, 0)), 'constant', constant_values=0)


if __name__ == '__main__':
    import time
    conv = Conv(3, 4)
    conv2 = Conv(4, 16)
    image_grey = cv2.cvtColor(cv2.imread("lenna.png"), cv2.COLOR_RGB2GRAY)
    image = cv2.imread("lenna.png")
    i1 = [
        [[0, 0, 0], [50, 0, 0], [0, 0, 0], [29, 0 ,0]],

        [[0, 0, 0], [80, 0, 0], [31, 0, 0], [2, 0, 0]],

        [[33, 0, 0], [90, 0, 0], [0, 0, 0], [75, 0, 0]],

        [[0, 0, 0], [9, 0, 0], [0, 0, 0], [95, 0, 0]]
    ]

    i2 = [[0, 50, 0, 29],
         [0, 80, 31, 2],
         [33, 90, 0, 75],
         [0, 9, 0, 95]]
    start = time.time()
    out = conv.forward(np.array(i2))
    out = conv2.forward(out)
    a = np.array(i1)
    #print(a)
    #print(a.shape)
    a_r = np.rollaxis(a, 2, 0)
    #print(a_r)
    #print(a_r.shape)
    end = time.time()

    # test rotating image
    print(image.shape)
    # change shape to be same as filter
    print(np.rollaxis(image, 2, 0).shape)
    # rotate
    print(np.rot90(np.rot90(np.rollaxis(image, 2, 0), axes=(1, 2)), axes=(1, 2)).shape)
    temp = np.rot90(np.rot90(np.rollaxis(image, 2, 0), axes=(1, 2)), axes=(1, 2))
    # reshape back to image
    print(np.rollaxis(temp, 2, 0).shape)
    temp = np.rollaxis(temp, 2, 0)
    print(np.rollaxis(temp, 2, 0).shape)
    # show image to see if rotated
    cv2.imshow("rotate", np.rollaxis(temp, 2, 0))
    cv2.waitKey()
