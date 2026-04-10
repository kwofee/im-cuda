#pragma once

// Otsu's binarization: calculates optimal threshold and binarizes the image.
class OtsuBinarizer {
public:
    OtsuBinarizer() = default;
    ~OtsuBinarizer() = default;

    // input: flat array of grayscale image (m x n) with values in range [0, 255]
    // output: flat array binarized (0 or 255)
    void apply(float *input, float *output, int m, int n);
};
