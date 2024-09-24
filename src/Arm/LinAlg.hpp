#ifndef LINALG_HPP
#define LINALG_HPP

#include "Tensor.hpp"
#include <tuple>

namespace HyperAlg {


    // vectorize this
    constexpr Tensor outer(Tensor x, Tensor y){
        if (x.getNdim() != 1 || y.getNdim() != 1) {
            throw std::invalid_argument("Tensors must be 1D vectors!");
        }

        unsigned int n = x.getShape()[0];
        unsigned int m = y.getShape()[0];

        Tensor z({n, m});

        for(unsigned int i = 0; i < n; i++){
            //Tensor w = x[{i}] * y; 
            for(unsigned int j = 0; j < m; j++){
                z[{i, j}] = x[{i}] * y[{j}];
            }
        }

        return z;
    }

    constexpr Tensor norm(Tensor x) {

        if (x.getNdim() != 1) {
            throw std::invalid_argument("Tensor must be 1D vector!");
        }

        double y = 0; 

        auto a = x.getData(); 
        int i = 0;

        if(x.getLen() > 8){
            for (; i <= x.getLen() - 8; i += 8) {
                asm volatile(
                    "ldr d0, [%[b], #0]      \n"

                    "ldp q1, q2, [%[a], #0]      \n"
                    "ldp q3, q4, [%[a], #32]     \n"

                    "fmul v5.2d, v1.2d, v1.2d\n"
                    "fmul v6.2d, v2.2d, v2.2d\n"
                    "fmul v7.2d, v3.2d, v3.2d\n"
                    "fmul v8.2d, v4.2d, v4.2d\n"

                    "faddp.2d d10, v5   \n"  // Pairwise addition of elements in v5, result in v5
                    "fadd d0, d0, d10  \n"  // Accumulate v5 result (scalar) into d0

                    "faddp.2d d12, v6   \n"  // Pairwise addition of elements in v6, result in v6
                    "fadd d0, d0, d12  \n"  // Accumulate v6 result (scalar) into d0

                    "faddp.2d d14, v7   \n"  // Pairwise addition of elements in v7, result in v7
                    "fadd d0, d0, d14  \n"  // Accumulate v7 result (scalar) into d0

                    "faddp.2d d16, v8   \n"  // Pairwise addition of elements in v8, result in v8
                    "fadd d0, d0, d16  \n"  // Accumulate v8 result (scalar) into d0



                    "str d0, [%[b], #0]      \n"
                    :
                    : [b] "r"(&y), [a] "r"(a + i)
                    : 
                );
            }
        }

        for(; i < x.getLen(); i++){ // for remaining elements.
            y += a[i] * a[i];
        }

        return std::sqrt(y);
    }

    constexpr Tensor chol(Tensor x){
        unsigned int n = x.getShape()[0];
        Tensor L({n, n});

        for(unsigned int i = 0; i < n; i++){
            for(unsigned int j = 0; j <= i; j++){
                double sum = 0; 
                if(i == j){
                    for(unsigned int k = 1; k < j; k++){
                        sum += L[{i, k}] * L[{i,k}];
                    }
                    L[{i,j}] = std::sqrt(x[{i,j}] - sum);
                }
                else {
                    for(unsigned int k = 1; k < j; k++){
                        sum += L[{j,k}] * L[{i,k}];
                    }
                    L[{i,j}] = std::sqrt(x[{i,j}] - sum) / L[{j,j}];
                }
            }
        }
        return L;
    }

    // https://johnfoster.pge.utexas.edu/numerical-methods-book/LinearAlgebra_LU.html
    // W UT Austin 
    constexpr std::tuple<Tensor, Tensor> lu(Tensor x){
        unsigned int n = x.getShape()[0];
        Tensor L = eye(n);
        Tensor U = x;


        for(unsigned int i = 0; i < n; i++){
            for(unsigned int j = i + 1; j < n; j++){
                L[{j, i}] = U[{j, i}] / U[{i, i}];
                    for(unsigned int k = 0; k < n; k++){
                        U[{j, k}] = U[{j, k}] - L[{j, i}] * U[{i, k}];
                    }
            }
        }
        return {L, U};
    }

    constexpr std::tuple<Tensor, Tensor> qr(Tensor x){
        if (x.getNdim() != 2) {
            throw std::invalid_argument("Tensor must be 2D matrix!");
        }

        unsigned int n = x.getShape()[0];
        unsigned int m = x.getShape()[1];

        Tensor q = eye(n);
        Tensor r({n, m});

        int t = std::min(n - 1, m);


        for(unsigned int i = 0; i < t; i++){
            Tensor y({n});
            for(unsigned int j = 0; j < n; j++){
                y[{j}] = x[{j, i}];
            }

            double normY = norm(y)[{0}];

            y[{i}] += std::copysign(1.0, y[{i}]) * normY;

            Tensor h = eye(n) - 2 * outer(y,y) / vdot(y,y);


            for(unsigned int i = 0; i < 3; i++){
              for(unsigned int j = 0; j < 3; j++){
                    std::cout <<h[{i, j}] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";


            std::cout << "\n";

            q =  q.transpose().matmul(h);
        }

        return {q, q.transpose().matmul(x)};
    }
}

#endif