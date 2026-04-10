#pragma once

// Converts RGB image to HSV image.
// Both input and output are expected to have shape (m, n, 3) flattened.
class RGB2HSVConverter {
public:
    RGB2HSVConverter() = default;
    ~RGB2HSVConverter() = default;

    // m, n: dimensions of the image. The array size should be m * n * 3.
    void apply(float *input, float *output, int m, int n);
};
