import cv2

class ImagePyramid:

    @classmethod
    def run(cls, src, ratio, layers):
        out = list([src,])
        for i in range(layers):
            src = cv2.resize(src, (0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
            out.append(src)
        return out

    def __call__(self, *args):
        return ImagePyramid.run(*args)


if __name__ == '__main__':
    src = cv2.imread('test1/000000039481.jpg', 0).astype(float)
    out = ImagePyramid()(src, 0.7071, 2)
    for i, img in enumerate(out):
        print(img.shape)
        cv2.imwrite('test1/{0}.png'.format(i), img)
