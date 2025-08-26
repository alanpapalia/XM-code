#ifndef MANIFOLD_H
#define MANIFOLD_H

#include <Utils/memory.h>
#include <functional>

/**
 * @brief A struct to hold the function pointers for a given manifold.
 * 
 * This struct is used to pass the manifold operations to the trust region solver.
 * This allows the solver to be generic and work with any manifold that implements these operations.
 */
template <typename T>
struct Manifold {
    /**
     * @brief Retracts a point on the manifold along a tangent vector.
     * 
     * @param new_x The new point on the manifold after retraction.
     * @param x The point on the manifold to retract from.
     * @param u The tangent vector to retract along.
     * @param lr The learning rate or step size.
     */
    std::function<void(DeviceDnTen<T>&, const DeviceDnTen<T>&, const DeviceDnTen<T>&, double)> retraction;

    /**
     * @brief Projects a vector from the ambient space onto the tangent space of the manifold.
     * 
     * @param projected_v The projected vector in the tangent space.
     * @param x The point on the manifold at which to project.
     * @param v The vector in the ambient space to project.
     */
    std::function<void(DeviceDnTen<T>&, const DeviceDnTen<T>&, const DeviceDnTen<T>&)> projection;

    /**
     * @brief Computes the inner product of two tangent vectors on the manifold.
     * 
     * @param cublas_H The cuBLAS handle.
     * @param u The first tangent vector.
     * @param v The second tangent vector.
     * @return The inner product of the two tangent vectors.
     */
    std::function<double(DeviceBlasHandle&, const DeviceDnTen<T>&, const DeviceDnTen<T>&)> inner_product;
};

#endif // MANIFOLD_H
