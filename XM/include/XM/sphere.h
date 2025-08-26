#ifndef SPHERE_H
#define SPHERE_H

#include <Utils/memory.h>
#include <XM/manifold.h>

/**
 * @brief Projects a vector v onto the tangent space of the sphere at point x.
 *
 * @param projected_v The projected vector.
 * @param v The vector to project.
 * @param x The point on the sphere.
 * @param n The dimension of the sphere.
 */
template <typename T>
__global__ void sphereProjectionKernal(T* projected_v, const T* v, const T* x, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Compute the dot product of v and x
        T dot_product = 0;
        for (size_s j = 0; j < n; ++j) {
            dot_product += v[j] * x[j];
        }
        // Project v onto the tangent space
        projected_v[i] = v[i] - dot_product * x[i];
    }
}

/**
 * @brief Retracts a point x along a tangent vector u to the sphere.
 *
 * @param new_x The new point on the sphere.
 * @param x The point on the sphere.
 * @param u The tangent vector.
 * @param n The dimension of the sphere.
 */
template <typename T>
__global__ void sphereRetractionKernal(T* new_x, const T* x, const T* u, size_s n) {
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Move x along the tangent vector u
        new_x[i] = x[i] + u[i];
    }
    __syncthreads();

    if (i == 0) {
        // Normalize the new point to project it back onto the sphere
        T norm = 0;
        for (size_s j = 0; j < n; ++j) {
            norm += new_x[j] * new_x[j];
        }
        norm = sqrt(norm);
        for (size_s j = 0; j < n; ++j) {
            new_x[j] /= norm;
        }
    }
}

/**
 * @brief Batched projection for the Oblique manifold.
 *
 * @param projected_v The projected vectors.
 * @param v The vectors to project.
 * @param x The points on the spheres.
 * @param n The dimension of the spheres.
 * @param num_spheres The number of spheres.
 */
template <typename T>
__global__ void obliqueProjectionKernal(T* projected_v, const T* v, const T* x, size_s n, size_s num_spheres) {
    // Iterate over each sphere in the batch
    for(size_s s = 0; s < num_spheres; ++s) {
        size_s i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            // Compute the dot product of the current sphere's vectors
            T dot_product = 0;
            for (size_s j = 0; j < n; ++j) {
                dot_product += v[s * n + j] * x[s * n + j];
            }
            // Project the vector onto the tangent space of the current sphere
            projected_v[s * n + i] = v[s * n + i] - dot_product * x[s * n + i];
        }
    }
}

/**
 * @brief Batched retraction for the Oblique manifold.
 *
 * @param new_x The new points on the spheres.
 * @param x The points on the spheres.
 * @param u The tangent vectors.
 * @param n The dimension of the spheres.
 * @param num_spheres The number of spheres.
 */
template <typename T>
__global__ void obliqueRetractionKernal(T* new_x, const T* x, const T* u, size_s n, size_s num_spheres) {
    // Iterate over each sphere in the batch
    for(size_s s = 0; s < num_spheres; ++s) {
        size_s i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            // Move the point on the current sphere along the tangent vector
            new_x[s * n + i] = x[s * n + i] + u[s * n + i];
        }
        __syncthreads();

        if (i == 0) {
            // Normalize the new point to project it back onto the current sphere
            T norm = 0;
            for (size_s j = 0; j < n; ++j) {
                norm += new_x[s * n + j] * new_x[s * n + j];
            }
            norm = sqrt(norm);
            for (size_s j = 0; j < n; ++j) {
                new_x[s * n + j] /= norm;
            }
        }
    }
}


/**
 * @brief Computes the inner product of two tangent vectors on the sphere.
 *
 * @param cublas_H The cuBLAS handle.
 * @param u The first tangent vector.
 * @param v The second tangent vector.
 * @return The inner product of the two tangent vectors.
 */
template <typename T>
inline double sphereInnerProduct(DeviceBlasHandle& cublas_H, const DeviceDnTen<T>& u, const DeviceDnTen<T>& v) {
    double result = 0;
    CHECK_CUBLAS(cublasDdot(cublas_H.cublas_handle, u.total_size, u.vals, 1, v.vals, 1, &result));
    return result;
}

/**
 * @brief An instance of the Manifold struct for the unit sphere.
 */
template <typename T>
Manifold<T> SphereManifold = {
    .retraction = [](DeviceDnTen<T>& new_x, const DeviceDnTen<T>& x, const DeviceDnTen<T>& u, double lr) {
        // Launch the sphere retraction kernel
        sphereRetractionKernal<<<(x.total_size + 1023) / 1024, 1024>>>(new_x.vals, x.vals, u.vals, x.total_size);
    },
    .projection = [](DeviceDnTen<T>& projected_v, const DeviceDnTen<T>& x, const DeviceDnTen<T>& v) {
        // Launch the sphere projection kernel
        sphereProjectionKernal<<<(x.total_size + 1023) / 1024, 1024>>>(projected_v.vals, v.vals, x.vals, x.total_size);
    },
    .inner_product = [](DeviceBlasHandle& cublas_H, const DeviceDnTen<T>& u, const DeviceDnTen<T>& v) {
        // Compute the inner product on the sphere
        return sphereInnerProduct(cublas_H, u, v);
    }
};

/**
 * @brief An instance of the Manifold struct for the Oblique manifold.
 */
template <typename T>
Manifold<T> ObliqueManifold = {
    .retraction = [](DeviceDnTen<T>& new_x, const DeviceDnTen<T>& x, const DeviceDnTen<T>& u, double lr) {
        // Launch the batched oblique retraction kernel
        obliqueRetractionKernal<<<(x.dimensions[0] + 1023) / 1024, 1024>>>(new_x.vals, x.vals, u.vals, x.dimensions[0], x.dimensions[1]);
    },
    .projection = [](DeviceDnTen<T>& projected_v, const DeviceDnTen<T>& x, const DeviceDnTen<T>& v) {
        // Launch the batched oblique projection kernel
        obliqueProjectionKernal<<<(x.dimensions[0] + 1023) / 1024, 1024>>>(projected_v.vals, v.vals, x.vals, x.dimensions[0], x.dimensions[1]);
    },
    .inner_product = [](DeviceBlasHandle& cublas_H, const DeviceDnTen<T>& u, const DeviceDnTen<T>& v) {
        // The inner product on the Oblique manifold is the sum of the inner products on each sphere,
        // which is equivalent to the standard Euclidean inner product on the flattened vectors.
        return sphereInnerProduct(cublas_H, u, v);
    }
};

#endif // SPHERE_H