#include <iostream>
#include <Utils/memory.h>
#include <Dense/matmul.h>
#include <Dense/trace.h>
#include <Dense/transpose.h>
#include <Dense/matdiagmul.h>
#include <Dense/matdot.h>
#include <Dense/matdivide.h>
#include <Dense/batchedQR.h>

#include <chrono>
#include <vector>
#include <fstream>
#include <Optimization/optimization.h>
#include <XM/manifold.h>

using namespace std::chrono;

/**
 * @brief An instance of the Manifold struct for the Stiefel manifold.
 * 
 * The Stiefel manifold is the set of all orthonormal k-frames in R^n.
 * The retraction is based on QR decomposition.
 * The projection is based on the formula: proj_x(v) = v - x * sym(x^T * v), where sym(A) = (A + A^T)/2.
 */
template <typename T>
Manifold<T> StiefelManifold = {
    .retraction = [](DeviceDnTen<T>& new_x, const DeviceDnTen<T>& x, const DeviceDnTen<T>& u, double lr) {
        // Update the point with a scaled tangent vector
        CHECK_CUBLAS(cublasDaxpy(new_x.cublas_handle, x.total_size, &lr, u.vals, 1, x.vals, 1));
        // Perform QR decomposition to retract back to the manifold
        batchedQR(new_x, x, 3, x.total_size / 3);
    },
    .projection = [](DeviceDnTen<T>& projected_v, const DeviceDnTen<T>& x, const DeviceDnTen<T>& v) {
        // Project the ambient vector v onto the tangent space at x
        opt_var RTgradR({3,3*x.dimensions[0]});
        opt_var RTgradR_sym({3,3*x.dimensions[0]});
        DnMatDnMatBatch(x.cublas_handle,RTgradR,x,v,3,x.dimensions[0],3,x.dimensions[0]/3,CUBLAS_OP_T,CUBLAS_OP_N);
        symBatched(RTgradR_sym.vals,RTgradR.vals,RTgradR.dimensions[0],x.dimensions[0]/3);
        CHECK_CUBLAS(cublasDcopy(x.cublas_handle, v.total_size, v.vals, 1, projected_v.vals, 1));
        DnMatDnMatBatch(x.cublas_handle,projected_v,x,RTgradR_sym,x.dimensions[0],3,3,x.dimensions[0]/3,CUBLAS_OP_N,CUBLAS_OP_N,-1.0,1.0);
    },
    .inner_product = [](DeviceBlasHandle& cublas_H, const DeviceDnTen<T>& u, const DeviceDnTen<T>& v) {
        // The inner product on the Stiefel manifold is the standard Euclidean inner product.
        double result = 0;
        CHECK_CUBLAS(cublasDdot(cublas_H.cublas_handle, u.total_size, u.vals, 1, v.vals, 1, &result));
        return result;
    }
};

/**
 * @brief An instance of the Manifold struct for the positive-orthant manifold.
 * 
 * The positive-orthant manifold is the set of all vectors with positive entries.
 * The retraction is based on the exponential map.
 * The projection is the identity, as the tangent space is the entire Euclidean space.
 */
template <typename T>
Manifold<T> PositiveManifold = {
    .retraction = [](DeviceDnTen<T>& new_x, const DeviceDnTen<T>& x, const DeviceDnTen<T>& u, double lr) {
        // Retract along the tangent vector using the exponential map
        positiveManifoldRetractionKernal<<<(x.total_size + 1023) / 1024, 1024>>>(new_x.vals, x.vals, u.vals, x.total_size, lr);
    },
    .projection = [](DeviceDnTen<T>& projected_v, const DeviceDnTen<T>& x, const DeviceDnTen<T>& v) {
        // The tangent space of the positive orthant is the entire Euclidean space,
        // so the projection is just the identity.
        CHECK_CUBLAS(cublasDcopy(x.cublas_handle, v.total_size, v.vals, 1, projected_v.vals, 1));
    },
    .inner_product = [](DeviceBlasHandle& cublas_H, const DeviceDnTen<T>& u, const DeviceDnTen<T>& v) {
        // The inner product on the positive-orthant manifold is the standard Euclidean inner product.
        double result = 0;
        CHECK_CUBLAS(cublasDdot(cublas_H.cublas_handle, u.total_size, u.vals, 1, v.vals, 1, &result));
        return result;
    }
};

template <typename T>
__global__ void positiveManifoldRetractionKernal(T* news, T* olds, T* grad, size_s n, double lr){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        news[i] = olds[i] * exp(lr*grad[i]/olds[i]);
    }  
}

template <typename T>
__global__ void ObjectiveLambdaKernal(T* news, T* olds, size_s n){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        news[i] = (olds[i] * olds[i] - 1)*(olds[i] * olds[i] - 1);
    }  
}

template <typename T>
__global__ void GradLambdaKernal(T* news, T* olds, size_s n){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        news[i] = (olds[i+1] * olds[i+1] - 1)*olds[i+1];
    }  
}

template <typename T>
__global__ void HessLambdaKernal(T* news, T* olds, T* oldsu, size_s n){
    size_s i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n){
        news[i] = (3 * olds[i+1] * olds[i+1] - 1) * oldsu[i+1];
    }  
}

// template <typename T>
// __global__ void compute_sum(T* v_s, T* s0, double* result, size_s size) {
//     size_s idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         double temp = (v_s[idx] * v_s[idx]) / (s0[idx] * s0[idx]);
//         atomicAdd(result, temp);
//     }
// }

// template <typename T>
// __global__ void ProductManifoldInnerKernal(T* As, T* Bs, T* s0, double* result, size_s n){
//     size_s i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i<n){
//         result
//     }  
// }

double ProductManifoldInner(DeviceBlasHandle& CUOPT_blas_handle, opt_var &AR,opt_var &BR, opt_var &As, opt_var &Bs){
    double result1 = 0;
    double result2 = 0;
    CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, AR.total_size, AR.vals, 1, BR.vals, 1, &result1));
    CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, As.total_size, As.vals, 1, Bs.vals, 1, &result2));
    
    return result1+result2;
}


/**
 * @brief The main trust region solver.
 * 
 * This function implements a trust region algorithm for optimization on a product of manifolds.
 * 
 * @param C The cost matrix.
 * @param vars A vector of variables to optimize.
 * @param manifolds A vector of manifolds corresponding to the variables.
 * @param lam The regularization parameter.
 * @param gradtol The tolerance for the gradient norm.
 * @param linesearch_step The initial step size for the line search.
 * @param v A temporary variable used in the solver.
 * @param primal_value A pointer to store the final primal value.
 * @param maxtime The maximum time to run the solver.
 */
void XMtrustregion(opt_var &C, const std::vector<opt_var*>& vars, const std::vector<Manifold<datatype>>& manifolds, const double lam, double &gradtol, double linesearch_step , opt_var &v , double* primal_value, const double maxtime){
    // initialize
    DeviceBlasHandle CUOPT_blas_handle;
    CUOPT_blas_handle.activate();
    // DeviceSolverDnHandle CUOPT_cuslover_handle;
    // CUOPT_cuslover_handle.activate();
    
    // size_s n = 5;
    // std::vector<datatype> Q_h(9*n*n);
    // for(size_l i = 0; i<9*n*n; ++i){
    //     Q_h[i] = static_cast<float>(rand()) / RAND_MAX;
    // }
    // opt_var A({3*n,3*n});
    // A.SynchronizeHostToDevice(Q_h.data());
    // opt_var C({3*n,3*n});
    // DnMatDnMat(CUOPT_blas_handle,C,A,A,CUBLAS_OP_N,CUBLAS_OP_T);

    // size_s n = 0;
    // std::vector<datatype> Q_h;
    // loadMatrixFromBin("../assets/matrix_data_20.bin", Q_h, n); 
    // opt_var C({3*n,3*n});
    // C.SynchronizeHostToDevice(Q_h.data());

    

    size_s o = vars[0]->dimensions[1];
    size_s n = C.dimensions[0]/3;
    size_s dim =  n * (3 * o - 6) + n - 1;
    double delta_bar = sqrt(dim);
    double delta = delta_bar / 8.0;

    // copy R
    opt_var R({3*n,o});
    CHECK_CUDA(cudaMemcpy(R.vals, vars[0]->vals, R.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));
    
    //s = ones(n-1,1);
    //s_ex = [1,s];
    // SUPERRRRRRRRR UGLY!!!!!!!!!!
    opt_var s_ex({n}); //because we reserve the first element to be 1
    std::vector<datatype> s_ex_h(n,1);
    s_ex.SynchronizeHostToDevice(s_ex_h.data());
    opt_var s; //because we reserve the first element to be 1
    s.vals = s_ex.vals+1;
    s.num_dims = 1;
    s.dimensions = new size_s[1];
    s.dimensions[0] = n-1;
    s.total_size = n-1;
    // copy s
    CHECK_CUDA(cudaMemcpy(s.vals, vars[1]->vals, s.total_size * sizeof(datatype), cudaMemcpyDeviceToDevice));

    opt_var p_s_ex({n}); //because we reserve the first element to be 1
    std::vector<datatype> p_s_ex_h(n,1);
    p_s_ex.SynchronizeHostToDevice(p_s_ex_h.data());
    p_s_ex.setValueAt(0,0);
    opt_var p_s; //because we reserve the first element to be 1
    p_s.vals = p_s_ex.vals+1;
    p_s.num_dims = 1;
    p_s.dimensions = new size_s[1];
    p_s.dimensions[0] = n-1;
    p_s.total_size = n-1;

    

    // the cost is sR * C * sR^T
    auto objc = [&C,&CUOPT_blas_handle, &CsR, &n, &slamobj,&lam, &vars](opt_var& sR_point_T,opt_var& s_point) {
        double result = 0;
        double result_lambda = 0;

        // This is a placeholder for a more general cost function.
        // You would need to modify this to work with your specific problem.
        for (size_t i = 0; i < vars.size(); ++i) {
            opt_var* var = vars[i];
            opt_var C_var({C.dimensions[0], var->dimensions[1]});
            DnMatDnMat(CUOPT_blas_handle, C_var, C, *var);
            double var_result = 0;
            CHECK_CUBLAS(cublasDdot(CUOPT_blas_handle.cublas_handle, C_var.total_size, C_var.vals, 1, var->vals, 1, &var_result));
            result += var_result;
        }

        ObjectiveLambdaKernal<<<(n + 1024 - 2) / 1024,1024>>>(slamobj.vals,s_point.vals,n-1);   
        CHECK_CUBLAS(cublasDasum(CUOPT_blas_handle.cublas_handle, slamobj.total_size, slamobj.vals, 1, &result_lambda));
        return result + lam * result_lambda;
    };
    

    //function [gradR,grads] = grad(AAT,R,s)
    //    n = size(R,1)/3;
    //    s_ex = [1;s];
    //    sR = R.*kron(s_ex,ones(3,1));
    //    dfdsR = 2 * AAT * sR;
    //    gradR = dfdsR.*kron(s_ex,ones(3,1));
    //    grads_pre = dfdsR.*R;
    //    grads = zeros(n-1,1);
    //    for i = 1:n-1
    //        grads(i) = sum(grads_pre(3*i+1:3*i+3,:),'all');
    //    end
    // end
    auto grad = [&C,&CUOPT_blas_handle,&vars,&n,&slamgrad,&lam](opt_var& R_point, opt_var& s_point, opt_var& sR_point){
        // This is a placeholder for a more general gradient calculation.
        // You would need to modify this to work with your specific problem.
        for (size_t i = 0; i < vars.size(); ++i) {
            opt_var* var = vars[i];
            opt_var grad_var({var->dimensions[0], var->dimensions[1]});
            DnMatDnMat(CUOPT_blas_handle, grad_var, C, *var, CUBLAS_OP_N, CUBLAS_OP_N, 2.0, 0.0);
            // You would need to store the gradient for each variable.
        }

        GradLambdaKernal<<<(n + 1024 - 2) / 1024,1024>>>(slamgrad.vals,s_point.vals,n-1);   
        double glam = 4 * lam;
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, slamgrad.total_size, &glam, slamgrad.vals, 1, vars[1]->vals, 1));
        return;
    };


    // function [hR,hs] = hess(AAT,R,s,Ru,su)
    //     % Hank: can you check if my gradient is correct?
    //     n = size(R,1)/3;
    //     o = size(R,2);
    //     s_ex = [1;s]; % add first scale = 1
    //     s_ex = kron(s_ex,ones(3,1));
    //     sR = R.*s_ex;  
    //     su_ex = [0;su]; % add first scale = 1
    //     su_ex = kron(su_ex,ones(3,1));   
    //     sRu = Ru .* s_ex;
    //     suR = R .* su_ex;
    //     hR = (2*AAT*(sRu+suR)).*s_ex+(2*AAT*sR).*su_ex;
    //     hs_pre = (2*AAT*(sRu+suR)).*R + (2*AAT*sR).*Ru;
    //     hs = zeros(n-1,1);
    //     for i = 1:n-1
    //        hs(i) = sum(hs_pre(3*i+1:3*i+3,:),'all');
    //     end
    // end
    auto ehess = [&C,&CUOPT_blas_handle,&vars,&n,&slamhess,&lam](opt_var& R_point, opt_var& s_point, opt_var& Ru_point, opt_var& su_point){
        // This is a placeholder for a more general Hessian calculation.
        // You would need to modify this to work with your specific problem.
        for (size_t i = 0; i < vars.size(); ++i) {
            opt_var* var = vars[i];
            opt_var hess_var({var->dimensions[0], var->dimensions[1]});
            // You would need to implement the Hessian calculation for each variable.
        }

        HessLambdaKernal<<<(n + 1024 - 2) / 1024,1024>>>(slamhess.vals,s_point.vals,su_point.vals,n-1); 
        double hlam = 4 * lam;
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, slamhess.total_size, &hlam, slamhess.vals, 1, vars[1]->vals, 1));
    };


    // function [rhR,rhs] = ehess2rhess(ehessR,ehesss,egradR,egrads,R,s,Ru,su)
    //     n = size(egradR,1)/3;
    //     rhR = zeros(size(egradR));
    //     for i = 1:n
    //         rhR(3*i-2:3*i,:) = ehessR(3*i-2:3*i,:) - Ru(3*i-2:3*i,:)*sym(R(3*i-2:3*i,:)'*egradR(3*i-2:3*i,:));
    //         rhR(3*i-2:3*i,:) = rhR(3*i-2:3*i,:) - R(3*i-2:3*i,:)*sym(R(3*i-2:3*i,:)'*rhR(3*i-2:3*i,:));
    //     end
    //     rhs = ehesss .* (s.^2) - su .* egrads .* s;
    // end

    // input R
    opt_var RTgradR({3,3*n});
    opt_var RTgradR_sym({3,3*n});   
    opt_var rhr({o,3*n});
    opt_var rhs({n-1});
    opt_var RTrhr({3,3*n});
    opt_var RTrhr_sym({3,3*n}); 
    opt_var sus({n-1});
    opt_var suhss({n-1});
    auto ehess2rhess = [&CUOPT_blas_handle,&RTgradR,&RTgradR_sym,&rhr,&rhs,&RTrhr,&RTrhr_sym,&sus,&suhss](opt_var& ehessR, opt_var& ehesss, opt_var& egradR, opt_var& egrads, opt_var& R_point, opt_var& s_point, opt_var& Ru_point, opt_var& su_point){
        size_s n = R_point.dimensions[1]/3;
        DnMatDnMatBatch(CUOPT_blas_handle,RTgradR,R_point,egradR,3,R_point.dimensions[0],3,n,CUBLAS_OP_T,CUBLAS_OP_N);
        symBatched(RTgradR_sym.vals,RTgradR.vals,RTgradR.dimensions[0],n);
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, ehessR.total_size, ehessR.vals, 1, rhr.vals, 1));
        CHECK_CUDA(cudaDeviceSynchronize());
        DnMatDnMatBatch(CUOPT_blas_handle,rhr,Ru_point,RTgradR_sym,Ru_point.dimensions[0],3,3,n,CUBLAS_OP_N,CUBLAS_OP_N,-1.0,1.0);
        
        DnMatDnMatBatch(CUOPT_blas_handle,RTrhr,R_point,rhr,3,R_point.dimensions[0],3,n,CUBLAS_OP_T,CUBLAS_OP_N);
        symBatched(RTrhr_sym.vals,RTrhr.vals,RTrhr.dimensions[0],n);
        DnMatDnMatBatch(CUOPT_blas_handle,rhr,R_point,RTrhr_sym,R_point.dimensions[0],3,3,n,CUBLAS_OP_N,CUBLAS_OP_N,-1.0,1.0);
        DnMatDnMatDot(rhs,ehesss,s_point,2);
        DnMatDnMatDot(sus,su_point,s_point,1);
        DnMatDnMatDot(suhss,sus,egrads,1);
        // rhs = rhs - suhss;
        double one = 1.0;
        CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, rhs.total_size, &one, suhss.vals, 1, rhs.vals, 1));
        return;
    };

    // function [rgradR,rgrads] = projection(egradR,egrads,R,s)
    //     n = size(egradR,1)/3;
    //     rgradR = zeros(size(egradR));
    //     for i = 1:n
    //         rgradR(3*i-2:3*i,:) = egradR(3*i-2:3*i,:) - R(3*i-2:3*i,:)*sym(R(3*i-2:3*i,:)'*egradR(3*i-2:3*i,:));
    //     end
    //     rgrads = (s.^2).*egrads;
    // end
    auto projection = [&](opt_var& R_point, opt_var& s_point, opt_var& R_gradient, opt_var& s_gradient){
        manifolds[0].projection(rgrad_r, R_point, R_gradient);
        manifolds[1].projection(rgrad_s, s_point, s_gradient);
    };

    auto retraction = [&](opt_var& R_point, opt_var& s_point, opt_var& rR_gradient, opt_var& rs_gradient, double lr){
        manifolds[0].retraction(new_R_T, R_point, rR_gradient, lr);
        manifolds[1].retraction(new_s, s_point, rs_gradient, lr);
    };

    opt_var R_T({o,3*n});
    transpose(R_T.vals,R.vals,R.dimensions[0],R.dimensions[1]);
    opt_var grad_r_T({o,3*n});  


    opt_var sR_new({3*n,o});
    std::cout<<"start linesearch"<<std::endl;
    if(linesearch_step != 0){
        // do linesearch here
        double f0 = objc(sR,s);
        double alpha = linesearch_step;
        // decent R is from input v
        opt_var decent_dir_R({3*n,o});
        CHECK_CUDA(cudaMemcpyAsync(decent_dir_R.vals + 3*n*(o-1), v.vals, sizeof(datatype) * v.total_size, cudaMemcpyDeviceToDevice));
        opt_var decent_dir_R_T({o,3*n});
        transpose(decent_dir_R_T.vals,decent_dir_R.vals,decent_dir_R.dimensions[0],decent_dir_R.dimensions[1]);
        // decent s is 0
        opt_var decent_dir_s({n-1});

        retraction(R_T,s,decent_dir_R_T,decent_dir_s,-alpha);
        transpose(R_T.vals,R.vals,R.dimensions[0],R.dimensions[1]);
        transpose(new_R.vals,new_R_T.vals,new_R_T.dimensions[0],new_R_T.dimensions[1]);
        dnmat_mul_spdiag_batch(sR_new,new_R,s_ex,3);
        double f = objc(sR_new,s);
        while(f>f0){
            alpha = alpha/2;
            retraction(R_T,s,decent_dir_R_T,decent_dir_s,-alpha);
            transpose(R_T.vals,R.vals,R.dimensions[0],R.dimensions[1]);
            transpose(new_R.vals,new_R_T.vals,new_R_T.dimensions[0],new_R_T.dimensions[1]);
            dnmat_mul_spdiag_batch(sR_new,new_R,s_ex,3);
            f = objc(sR_new,s);
            if(alpha<1e-20){
                printf("linesearch failed! BM stopped! \n");
                s.vals = nullptr;
                p_s.vals = nullptr;
                new_s.vals = nullptr;
                *primal_value = -1;
                return;
            }
        }

        if(f0-f>0){
            printf("linesearch decrease %1.3e\n",f0-f);
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R_T.total_size, new_R_T.vals, 1, R_T.vals, 1));
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R.total_size, new_R.vals, 1, R.vals, 1));
        }
        else{
            printf("linesearch failed! BM stopped! \n");
            s.vals = nullptr;
            p_s.vals = nullptr;
            new_s.vals = nullptr;
            *primal_value = -1;
            return;
        }

    }


    // opt_var sR_T({o,3*n});
    // opt_var sR_TsR({o,o});
    // transpose(sR_T.vals,sR.vals,sR.dimensions[0],sR.dimensions[1]);
    // DnMatDnMat(CUOPT_blas_handle,sR_TsR,sR_T,sR); 
    // sR_TsR.print();
    size_s max_inner_iter = 1000;
    size_s max_outer_iter = 1000;
    size_s totalite = 0;
  
    double* loss = new double[max_outer_iter];
    double* gradnorm = new double[max_outer_iter];
    loss[0] = objc(sR,s);
    
    // 1 for nagative curvature
    // 2 for exceed trust region
    // 3 for reached norm tolerance
    // 5 for numerical issue
    // 6 for max iteration
    int endreason = 6;
    // 1 for tr region shrink
    // 2 for tr region expand
    // 3 for tr region REJ
    // 4 for tr region remain
    int trstatus = 4;
    int shrink_count = 0;

    opt_var rhsds({n-1});
    opt_var rsds({n-1});
    opt_var psds({n-1});
    opt_var vsds({n-1});
    opt_var hvsds({n-1});
    opt_var rgradsds({n-1});

    opt_var r_R({o,3*n});
    opt_var r_s({n-1});
    opt_var p_R({o,3*n});
    opt_var p_R_T({3*n,o});
    opt_var v_R_T({3*n,o});
    size_s i = 0;
    size_s k = 0;
    auto start = high_resolution_clock::now();
    for(k = 0; k<max_outer_iter;++k){

        opt_var v_R({o,3*n});
        opt_var v_s({n-1});

        opt_var hv_R({o,3*n});
        opt_var hv_s({n-1});
        opt_var bestR(R);
        opt_var bests(s);
        double bestloss = loss[k];
        double* rdotr = new double[max_inner_iter];
        double* lossqu = new double[max_inner_iter];
        lossqu[0] = 0;
        double* lossqu1 = new double[max_inner_iter];
        lossqu1[0] = 0;
        grad(R,s_ex,sR);
        transpose(grad_r_T.vals,grad_r.vals,grad_r.dimensions[0],grad_r.dimensions[1]);
        transpose(R_T.vals,R.vals,R.dimensions[0],R.dimensions[1]);
        projection(R_T,s,grad_r_T,grad_s);
        DnMatDnMatDivide(rgradsds,rgrad_s,s,1);
        // // r_R = rgradR;
        // // r_s = rgrads;
        // // p_R = -r_R;
        // // p_s = -r_s;
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_r.total_size, rgrad_r.vals, 1, r_R.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_s.total_size, rgrad_s.vals, 1, r_s.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_r.total_size, rgrad_r.vals, 1, p_R.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, rgrad_s.total_size, rgrad_s.vals, 1, p_s.vals, 1));
        double neg_one = -1.0;
        cublasDscal(CUOPT_blas_handle.cublas_handle, p_R.total_size, &neg_one, p_R.vals, 1);
        cublasDscal(CUOPT_blas_handle.cublas_handle, p_s.total_size, &neg_one, p_s.vals, 1);
        DnMatDnMatDivide(rsds,r_s,s,1);
        rdotr[0] = ProductManifoldInner(CUOPT_blas_handle,r_R,r_R,rsds,rsds);
        gradnorm[k] = sqrt(rdotr[0]);
        
        if(k>0){   
        switch (trstatus)
        {
        case 1:
            std::cout << "TR- " ;
            break;
        case 2: 
            std::cout << "TR+ ";
            break;
        case 3:
            std::cout << "REJ " ;
            break;
        case 4:
            std::cout << "TR " ;
            break;
        }
        }
        printf("%d   %d   %1.3e   %1.3e",k, i+1,loss[k],gradnorm[k]);
        if(k>0){   
        switch (endreason)
        {
        case 1:
            std::cout << "   nagative curvature" << std::endl;
            break;
        case 2: 
            std::cout << "   exceed trust region" << std::endl;
            break;
        case 3:
            std::cout << "   reached norm tolerance" << std::endl;
            break;
        case 5:
            std::cout << "   numerical issue" << std::endl;
            break;
        case 6:
            std::cout << "   max iteration" << std::endl;
            break;
        }
        }else{
            printf("\n");
        }
        if(endreason == 5){
            printf("Terminate because of rdotr touched machine precise\n");
            break;
        }

        if(gradnorm[k] < gradtol){
            printf("Terminate because of small gradient norm\n");
            gradtol /= 10;
            break;
        }

        auto middle_end_time = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(middle_end_time - start);
        if(duration.count() > maxtime){
            printf("Terminate because of time limit\n");
            break;
        }
        endreason = 6;
        trstatus = 4;
        double* vdotv_record = new double[max_inner_iter];
        double* vdotp_record = new double[max_inner_iter];
        double* pdotp_record = new double[max_inner_iter];
        vdotv_record[0] = 0;
        vdotp_record[0] = 0;
        pdotp_record[0] = rdotr[0];
        endreason = 6;
        DnMatDnMat(CUOPT_blas_handle,CsR,C,sR,CUBLAS_OP_N,CUBLAS_OP_N,2.0,0.0); 
        CHECK_CUDA(cudaDeviceSynchronize());

        // double violation = 0;
        // double another_alpha = 0;

        for(i = 0; i<max_inner_iter;++i){
            transpose(p_R_T.vals,p_R.vals,p_R.dimensions[0],p_R.dimensions[1]);
            ehess(R,s_ex,p_R_T,p_s_ex);
            transpose(hrT.vals,hr.vals,hr.dimensions[0],hr.dimensions[1]);
            ehess2rhess(hrT,hs,grad_r_T,grad_s,R_T,s,p_R,p_s);
            //calculate inner
            DnMatDnMatDivide(rhsds,rhs,s,2);
            double alpha = rdotr[i] / ProductManifoldInner(CUOPT_blas_handle,p_R,rhr,p_s,rhsds);
            
            double vdotv = vdotv_record[i];
            double pdotp = pdotp_record[i];
            double vdotp = vdotp_record[i];
            
            if (rdotr[i] < 1e-15) {
                printf("gradient residual is very small!\n");
                endreason = 5; 
                break;
            }           
            if(alpha <=0){
                double sqrt_val = sqrt(vdotp * vdotp + pdotp * (delta * delta - vdotv));
                double tau = (-vdotp + sqrt_val) / pdotp;
                // v_R = v_R + tau * p_R;
                // v_s = v_s + tau * p_s;
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &tau, p_R.vals, 1, v_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &tau, p_s.vals, 1, v_s.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &tau, rhr.vals, 1, hv_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &tau, rhs.vals, 1, hv_s.vals, 1));
                endreason = 1;
                break;
            }
            if(vdotv + 2*alpha*vdotp + alpha*alpha*pdotp > delta * delta){
                double sqrt_val = sqrt(vdotp * vdotp + pdotp * (delta * delta - vdotv));
                double tau = (-vdotp + sqrt_val) / pdotp;
                // v_R = v_R + tau * p_R;
                // v_s = v_s + tau * p_s;
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &tau, p_R.vals, 1, v_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &tau, p_s.vals, 1, v_s.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &tau, rhr.vals, 1, hv_R.vals, 1));
                CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &tau, rhs.vals, 1, hv_s.vals, 1));
                endreason = 2;
                break;
            }
            // v_R = v_R + alpha * p_R;
            // r_R = r_R + alpha * rhR;
            // v_s = v_s + alpha * p_s;
            // r_s = r_s + alpha * rhs;
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_R.total_size, &alpha, p_R.vals, 1, v_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, v_s.total_size, &alpha, p_s.vals, 1, v_s.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, r_R.total_size, &alpha, rhr.vals, 1, r_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, r_s.total_size, &alpha, rhs.vals, 1, r_s.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_R.total_size, &alpha, rhr.vals, 1, hv_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, hv_s.total_size, &alpha, rhs.vals, 1, hv_s.vals, 1));

            // retraction(R_T,s,v_R,v_s,1);
            // transpose(new_R.vals,new_R_T.vals,new_R_T.dimensions[0],new_R_T.dimensions[1]);
            // dnmat_mul_spdiag_batch(sR,new_R,new_s_ex,3);
            // double lossgt = objc(sR);
            // if(lossgt < bestloss){
            //     CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R.total_size, new_R.vals, 1, bestR.vals, 1));
            //     CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_s.total_size, new_s.vals, 1, bests.vals, 1));
                
            //     bestloss = lossgt;
            // }
            // // recover RT
            // transpose(R_T.vals,R.vals,R.dimensions[0],R.dimensions[1]);
            
            DnMatDnMatDivide(rsds,r_s,s,1);
            rdotr[i+1] = ProductManifoldInner(CUOPT_blas_handle,r_R,r_R,rsds,rsds);
            if(sqrt(rdotr[i+1]) < gradnorm[k] * min(gradnorm[k],0.1)){
                endreason = 3;
                break;
            }
            // beta = rdotr(end)/rdotr(end-1);
            // p_R = -r_R + beta * p_R;
            // p_s = -r_s + beta * p_s;
            double beta = rdotr[i+1] / rdotr[i];
            CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_R.total_size, &beta, p_R.vals, 1));
            CHECK_CUBLAS(cublasDscal(CUOPT_blas_handle.cublas_handle, p_s.total_size, &beta, p_s.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, p_R.total_size, &neg_one, r_R.vals, 1, p_R.vals, 1));
            CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, p_s.total_size, &neg_one, r_s.vals, 1, p_s.vals, 1));
            // vdotv_record(end+1) = vdotv + 2*alpha*vdotp + alpha^2*pdotp;
            // vdotp_record(end+1) = beta*(vdotp+alpha*pdotp);
            // pdotp_record(end+1) = beta^2*pdotp + rdotr(end);
            vdotv_record[i+1] = vdotv + 2*alpha*vdotp + alpha*alpha*pdotp;
            vdotp_record[i+1] = beta*(vdotp+alpha*pdotp);
            pdotp_record[i+1] = beta*beta*pdotp + rdotr[i+1];

            // DnMatDnMatDivide(vsds,v_s,s,1);
            // DnMatDnMatDivide(hvsds,hv_s,s,1);
            // DnMatDnMatDivide(rgradsds,rgrad_s,s,1);
            // lossqu[i+1] = ProductManifoldInner(CUOPT_blas_handle,v_R,hv_R,vsds,hvsds)/2 + ProductManifoldInner(CUOPT_blas_handle,v_R,rgrad_r,vsds,rgradsds);
            // lossqu1[i+1] = ProductManifoldInner(CUOPT_blas_handle,v_R,hv_R,vsds,hvsds);
            // if(lossqu[i+1] >= lossqu[i]){
            //     s.print();
            //     printf("error! loss_qu is increasing\n");
            // }
            // DnMatDnMatDivide(psds,p_s,s,1);
            // DnMatDnMatDivide(rsds,r_s,s,1);
            // violation = ProductManifoldInner(CUOPT_blas_handle,p_R,r_R,psds,rsds)/sqrt(pdotp_record[i+1]*rdotr[i+1]);
            // another_alpha = ProductManifoldInner(CUOPT_blas_handle,p_R,r_R,psds,rsds);
            // opt_var another_r(rgrad_r);
            // double one = 1.0;
            // CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, another_r.total_size, &one, hv_R.vals, 1, another_r.vals, 1));
            // CHECK_CUBLAS(cublasDaxpy(CUOPT_blas_handle.cublas_handle, another_r.total_size, &neg_one, r_R.vals, 1, another_r.vals, 1));
            
        }
        
        totalite += i + 1;
        DnMatDnMatDivide(vsds,v_s,s,2);
        double loss_qu = ProductManifoldInner(CUOPT_blas_handle,v_R,hv_R,vsds,hv_s)/2 + ProductManifoldInner(CUOPT_blas_handle,v_R,rgrad_r,vsds,rgrad_s);
        if(loss_qu >= 0){
            printf("error! loss_qu is larger than 0\n");
            break;
        }
        retraction(R_T,s,v_R,v_s,1);
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_R_T.total_size, new_R_T.vals, 1, R_T.vals, 1));
        CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, new_s.total_size, new_s.vals, 1, s.vals, 1));
        transpose(R.vals,R_T.vals,R_T.dimensions[0],R_T.dimensions[1]);
        dnmat_mul_spdiag_batch(sR,R,s_ex,3);
        loss[k+1] = objc(sR,s);

        double rou = (loss[k+1] - loss[k]) / loss_qu;
        //std::cout << "rou: " << rou ;
        if(rou < 0.25){
            delta = delta * 0.25;
            trstatus = 1;
            shrink_count ++;
        }else if(rou > 0.75 && endreason <= 2){
            delta = min(delta * 2,delta_bar);
            trstatus = 2;
            shrink_count = 0;
        }else{
            shrink_count = 0;
        }
        if(shrink_count > 3){
            delta = delta * 1e-3;
            shrink_count = 0;
            printf("delta shrinked to %1.3e\n",delta);
            if(delta < 1e-20){
                printf("delta is too small, BM stopped!\n");
                break;
            }
        }
        if(loss[k+1] > bestloss || rou < 0.1){
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, R.total_size, bestR.vals, 1, R.vals, 1));
            CHECK_CUBLAS(cublasDcopy(CUOPT_blas_handle.cublas_handle, s.total_size, bests.vals, 1, s.vals, 1));
            loss[k+1] = bestloss;
            dnmat_mul_spdiag_batch(sR,R,s_ex,3);
            trstatus = 3;
        }
        
    }
    std::cout <<std::endl<< "Total iteration:     " << totalite <<std::endl;
    auto end = high_resolution_clock::now();
        std::cout << "Time taken by function1: "
            << duration_cast<milliseconds>(end - start).count() << " ms" << std::endl;
    *primal_value = loss[k];
    // record result
    CHECK_CUDA(cudaMemcpy(R_result.vals,R.vals,R.total_size * sizeof(datatype),cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(s_result.vals,s.vals,s.total_size * sizeof(datatype),cudaMemcpyDeviceToDevice));

    // must equal null, or there will be a memory leak
    s.vals = nullptr;
    p_s.vals = nullptr;
    new_s.vals = nullptr;
}