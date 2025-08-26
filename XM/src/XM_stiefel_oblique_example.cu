#include <iostream>
#include <vector>
#include <XM/trustregion.h>
#include <XM/sphere.h>

int main() {
    // Define the dimensions of the manifolds
    size_s n = 3; // Dimension of the ambient space
    size_s p = 2; // Number of columns for the Stiefel manifold
    size_s m = 5; // Number of spheres for the Oblique manifold

    // Create the variables
    opt_var R({n, p});
    opt_var X({n, m});

    // Initialize the variables with random data
    R.rand();
    X.rand();

    // Create the manifolds
    Manifold<datatype> stiefel_manifold = StiefelManifold<datatype>;
    Manifold<datatype> oblique_manifold = ObliqueManifold<datatype>;

    // Create the vector of variables and manifolds
    std::vector<opt_var*> vars = {&R, &X};
    std::vector<Manifold<datatype>> manifolds = {stiefel_manifold, oblique_manifold};

    // Define the cost function (a simple sum of squares)
    opt_var C({n, n});
    C.eye();

    // Set the optimization parameters
    double lam = 0.0;
    double gradtol = 1e-6;
    double linesearch_step = 0.0;
    opt_var v({n, p + m});
    double primal_value = 0.0;
    double maxtime = 100.0;

    // Call the trust region solver
    XMtrustregion(C, vars, manifolds, lam, gradtol, linesearch_step, v, &primal_value, maxtime);

    // Print the results
    std::cout << "Final primal value: " << primal_value << std::endl;
    std::cout << "Optimized R:" << std::endl;
    R.print();
    std::cout << "Optimized X:" << std::endl;
    X.print();

    return 0;
}
