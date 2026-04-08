def kuwahara_filter(img, ksize=5):
    h, w = img.shape
    pad = ksize // 2
    output = img.copy()

    for i in range(pad, h-pad):
        for j in range(pad, w-pad):
            regions = []

            r = ksize // 2

            regions.append(img[i-r:i+1, j-r:j+1])
            regions.append(img[i-r:i+1, j:j+r+1])
            regions.append(img[i:i+r+1, j-r:j+1])
            regions.append(img[i:i+r+1, j:j+r+1])

            min_var = float('inf')
            best_mean = 0

            for reg in regions:
                mean, std = cv2.meanStdDev(reg)
                var = std[0][0] ** 2

                if var < min_var:
                    min_var = var
                    best_mean = mean[0][0]

            output[i, j] = best_mean

    return output