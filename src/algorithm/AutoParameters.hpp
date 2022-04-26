//
// Created by joel on 26.04.22.
//

#ifndef APR_AUTOPARAMETERS_HPP
#define APR_AUTOPARAMETERS_HPP

#include "APRParameters.hpp"
#include "data_structures/Mesh/PixelData.hpp"

#include <stdio.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

template<typename T>
void compute_means(const std::vector<T>& data, float threshold, float& mean_back, float& mean_fore) {
    float sum_fore=0.f, sum_back=0.f;
    size_t count_fore=0, count_back=0;

#ifdef HAVE_OPENMP
#pragma omp parallel for default(none) shared(data, threshold) reduction(+:sum_fore, sum_back, count_fore, count_back)
#endif
    for(size_t idx = 0; idx < data.size(); ++idx) {
        if(data[idx] > threshold) {
            sum_fore += data[idx];
            count_fore++;
        } else {
            sum_back += data[idx];
            count_back++;
        }
    }
    mean_fore = sum_fore / count_fore;
    mean_back = sum_back / count_back;
}


/**
 * Compute threshold value by Li's iterative Minimum Cross Entropy method [1]
 *
 * Note: it is assumed that the input elements are non-negative (as is the case for the gradient and local intensity
 * scale). To apply the method to data that may be negative, subtract the minimum value from the input, and then add
 * that value to the computed threshold.
 *
 * [1] Li, C. H., & Tam, P. K. S. (1998). "An iterative algorithm for minimum cross entropy thresholding."
 *     Pattern recognition letters, 19(8), 771-776.
 */
template<typename T>
float threshold_li(const std::vector<T> &input) {

    if(input.empty()) { return 0.f; }     // return 0 if input is empty

    const T image_min = *std::min_element(input.begin(), input.end());
    const T image_max = *std::max_element(input.begin(), input.end());

    if(image_min == image_max) { return image_min; }  // if all inputs are equal, return that value

    float tolerance = 0.5f;   // tolerance 0.5 should suffice for integer inputs

    // For floating point inputs we set the tolerance to the lesser of 0.01 and a fraction of the data range
    // This could be improved, by instead taking e.g. half the smallest difference between any two non-equal elements
    if(std::is_floating_point<T>::value) {
        float range = image_max - image_min;
        tolerance = std::min(0.01f, range*1e-4f);
    }

    // Take the mean of the input as initial value
    float t_next = std::accumulate(input.begin(), input.end(), 0.f) / (float) input.size();
    float t_curr = -2.f * tolerance; //this ensures that the first iteration is performed, since t_next is non-negative

    // For integer inputs, we have to ensure a large enough initial value, such that mean_back > 0
    if(!std::is_floating_point<T>::value) {
        // if initial value is <1, try to increase it to 1.5 unless the range is too narrow
        if(t_next < 1.f) {
            t_next = std::min(1.5f, (image_min+image_max)/2.f);
        }
    }

    // Perform Li iterations until convergence
    while(std::abs(t_next - t_curr) > tolerance) {
        t_curr = t_next;

        // Compute averages of values above and below the current threshold
        float mean_back, mean_fore;
        compute_means(input, t_curr, mean_back, mean_fore);

        // Handle the edge case where all values < t_curr are 0
        if(mean_back == 0) {
            std::wcout << "log(0) encountered in APRConverter::threshold_li, returning current threshold" << std::endl;
            return t_curr;
        }

        // Update the threshold (one-point iteration)
        t_next = (mean_fore - mean_back) / (std::log(mean_fore) - std::log(mean_back));
    }
    return t_next;
}


template<typename S, typename T, typename U>
void autoParametersLiEntropy(APRParameters& par,
                             const PixelData<S>& image,
                             const PixelData<T>& localIntensityScale,
                             const PixelData<U>& grad,
                             const float bspline_offset,
                             const bool verbose=false) {

    std::vector<U> grad_subsampled;
    std::vector<T> lis_subsampled;

    {   // intentional scope
        /// First we extract the gradient and local intensity scale values at all locations where the image
        /// intensity exceeds the intensity threshold. This allows better adaptation in certain cases, e.g.
        /// when there is significant background AND signal noise/autofluorescence (provided that par.Ip_th
        /// is set appropriately).
        std::vector<U> grad_foreground(grad.mesh.size());
        std::vector<T> lis_foreground(localIntensityScale.mesh.size());

        const auto threshold = par.Ip_th + bspline_offset;
        size_t counter = 0;
        for(size_t idx = 0; idx < grad.mesh.size(); ++idx) {
            if(image.mesh[idx] > threshold) {
                grad_foreground[counter] = grad.mesh[idx];
                lis_foreground[counter] = localIntensityScale.mesh[idx];
                counter++;
            }
        }

        const size_t num_foreground_pixels = counter;

        grad_foreground.resize(num_foreground_pixels); //setting size to non-zero elements.
        lis_foreground.resize(num_foreground_pixels);

        /// Then we uniformly subsample these signals, as we typically don't need all elements to compute the thresholds
        const size_t num_elements = std::min((size_t)32*512*512, num_foreground_pixels); //arbitrary number.
        const size_t delta = num_foreground_pixels / num_elements;
        grad_subsampled.resize(num_elements);
        lis_subsampled.resize(num_elements);

#ifdef HAVE_OPENMP
#pragma omp parallel for default(shared)
#endif
        for(size_t idx = 0; idx < num_elements; ++idx) {
            grad_subsampled[idx] = grad_foreground[idx*delta];
            lis_subsampled[idx] = lis_foreground[idx*delta];
        }
    }

    /// Compute thresholds using Li's iterative minimum cross entropy algorithm
    par.grad_th = threshold_li(grad_subsampled);
    par.sigma_th = threshold_li(lis_subsampled);

    if(verbose) {
        std::cout << "Automatic parameter tuning found sigma_th = " << par.sigma_th <<
                     " and grad_th = " << par.grad_th << std::endl;
    }
}

#endif //APR_AUTOPARAMETERS_HPP
