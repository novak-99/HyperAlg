#ifndef TENSOR_CPP
#define TENSOR_CPP

#include <initializer_list>
#include <vector>
#include <cstring>
#include <type_traits>
#include <stdexcept>

#include <cmath>
#include <iostream>
#include <arm_neon.h>

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


namespace HyperAlg{

    class Tensor{
        public:

            // OK this is kinda bad
            constexpr Tensor(const Tensor& x) noexcept : itemSize(sizeof(double)), ndim(x.getNdim()), len(x.getLen()) {
                data = new double[len];
                shape = new unsigned int[ndim];
                strides = new unsigned int[ndim];

                // auto xData = x.getData(); 

                // int i = 0;
                // // data = (double*)std::memcpy(data, x.getData(), itemSize * len);
                // if(len > 8){
                //     auto time1 = high_resolution_clock::now();
                //     for (; i <= len - 8; i += 8) {
                //         asm volatile(
                //             "ldp q0, q1, [%[b], #0]      \n"
                //             "ldp q2, q3, [%[b], #32]     \n"
                //             "ldp q4, q5, [%[b], #64]      \n"
                //             "ldp q6, q7, [%[b], #96]     \n"

                //             "stp q0, q1, [%[a], #0]      \n"
                //             "stp q2, q3, [%[a], #32]     \n"
                //             "stp q4, q5, [%[a], #64]      \n"
                //             "stp q6, q7, [%[a], #96]     \n"
                //             :
                //             : [a] "r"(data + i), [b] "r"(xData + i)
                //             : 
                //         );
                //     }
                //     auto time2 = high_resolution_clock::now();
                //     duration<double, std::milli> ms_double = time2 - time1; // 0.6ms (ours) vs. 3ms (Eigen)
                //     std::cout << ms_double.count() << "ms\n";
                // }
                // else { 
                //     data = (double*)memcpy(data, x.getData(), itemSize * len);
                // }
                
                data = (double*)memcpy(data, x.getData(), itemSize * len);
                shape = (unsigned int*)std::memcpy(shape, x.getShape(), sizeof(unsigned int) * ndim);
                strides = (unsigned int*)std::memcpy(strides, x.getStrides(), sizeof(unsigned int) * ndim);


            }

            constexpr Tensor(double* data, unsigned int* dims, int ndim) noexcept : itemSize(sizeof(double)), ndim(ndim) {
                
                createTensorStructure(dims);

                this->data = data;
                len = sizeof(data) / itemSize;
            }

            constexpr Tensor(double* data, const std::vector<unsigned int>& dims) noexcept : itemSize(sizeof(double)), ndim(dims.size()) {

                createTensorStructure(dims);

                this->data = data;
                len = sizeof(data) / itemSize;
            }


            // initializes tensor given only shapes
            constexpr Tensor(const std::initializer_list<unsigned int>& dims) noexcept : itemSize(sizeof(double)), ndim(dims.size()) {

                len = 1;
                for(auto it = dims.begin(); it != dims.end(); it++) len *= *it;
                createTensorStructure(dims);

                data = new double[len](); // yea good luck with setting all of this lol
                // maybe change from emtpy l8r..
            }

            // initializes tensor given only a signle element
            constexpr Tensor(const double element) noexcept : itemSize(sizeof(double)), ndim(1), len(1) {


                shape = new unsigned int[1];
                strides = new unsigned int[1];
                data = new double[1]; 

                shape[0] = 1; 
                strides[0] = itemSize;
                data[0] = element;

            }

            constexpr Tensor(const std::vector<double>& vector) noexcept : itemSize(sizeof(double)), ndim(1), len(vector.size()) {
                strides = new unsigned int[ndim];
                strides[ndim - 1] = itemSize;

                shape = new unsigned int[ndim];
                shape[0] = vector.size();
                
                data = new double[len];

                int i = 0; 
                for(auto it = vector.begin(); it != vector.end(); it++){
                    data[i] = *it;
                    i++; 
                }
            }


            // // initializes tensor given initializer list
            // template <typename T>
            // constexpr Tensor(const std::vector<T>& ndarray) noexcept : itemSize(sizeof(double)) {
            //     std::vector<double> flattenedArray;
            //     std::vector<unsigned int> dims;

            //     processArray(ndarray, flattenedArray);
            //     calculateDims(ndarray, dims);

            //     len = flattenedArray.size();

            //     data = new double[len];

            //     for(int i = 0; i < len; i++){
            //         data[i] = flattenedArray[i];
            //     }

            //     ndim = dims.size();

            //     createTensorStructure(dims);
            // }

            template <typename T>
            constexpr Tensor(const std::vector<std::vector<T>>& ndarray) noexcept : itemSize(sizeof(double)) {
                std::vector<double> flattenedArray;
                std::vector<unsigned int> dims;

                processArray(ndarray, flattenedArray);
                calculateDims(ndarray, dims);

                len = flattenedArray.size();

                data = new double[len];

                for(int i = 0; i < len; i++){
                    data[i] = flattenedArray[i];
                }

                ndim = dims.size();

                createTensorStructure(dims);
            }

            // // initializes tensor given initializer list
            // template <class T>
            // constexpr Tensor(const std::vector<std::vector<T>>& ndarray) noexcept : itemSize(sizeof(double)) {
            //     std::vector<double> flattenedArray;
            //     std::vector<unsigned int> dims;

            //     processArray(ndarray, flattenedArray);
            //     calculateDims(ndarray, dims);

            //     len = flattenedArray.size();

            //     data = new double[len];

            //     for(int i = 0; i < len; i++){
            //         data[i] = flattenedArray[i];
            //     }

            //     ndim = dims.size();

            //     createTensorStructure(dims);
            // }


            constexpr ~Tensor() noexcept {
                delete data; 
                delete strides;
                delete shape;
            }

            constexpr double* getData() const noexcept {
                return data; 
            }

            constexpr unsigned int* getShape() const noexcept {
                return shape; 
            }

            constexpr unsigned int* getStrides() const noexcept {
                return strides; 
            }

            constexpr unsigned int getNdim() const noexcept {
                return ndim;
            }

            constexpr unsigned int getLen() const noexcept {
                return len;
            }

            constexpr double& operator[](const std::vector<unsigned int>& indices) noexcept {
                // make sure indices size !> ndim.
                unsigned int offSet = 0; 
                for(int i = 0; i < ndim; i++){
                    offSet += indices[i] * strides[i];
                } 

                return data[offSet/itemSize];
            }

            constexpr double& operator[](const std::vector<unsigned int>& indices) const noexcept {
                // make sure indices size !> ndim.
                // make sure inside of bounds
                unsigned int offSet = 0; 
                for(int i = 0; i < ndim; i++){
                    offSet += indices[i] * strides[i];
                } 

                return data[offSet/itemSize];
            }

            inline void operator=(const Tensor& x) noexcept {
                data = new double[len];
                shape = new unsigned int[ndim];
                strides = new unsigned int[ndim];

                data = (double*)memcpy(data, x.getData(), itemSize * len);
                shape = (unsigned int*)std::memcpy(shape, x.getShape(), sizeof(unsigned int) * ndim);
                strides = (unsigned int*)std::memcpy(strides, x.getStrides(), sizeof(unsigned int) * ndim);
            }

            constexpr Tensor& operator+=(const Tensor& x) noexcept {

                auto a = data; 
                auto b = x.getData(); 
                int i = 0;

                if(len > 8){
                    for (; i <= len - 8; i += 8) {
                        asm volatile(
                            "ldp q0, q1, [%[a], #0]      \n"
                            "ldp q2, q3, [%[a], #32]     \n"
                            "ldp q4, q5, [%[b], #0]      \n"
                            "ldp q6, q7, [%[b], #32]     \n"

                            "fadd.2d v0, v0, v4     \n"
                            "fadd.2d v1, v1, v5     \n"
                            "fadd.2d v2, v2, v6     \n"
                            "fadd.2d v3, v3, v7     \n"

                            "stp q0, q1, [%[a], #0]      \n"
                            "stp q2, q3, [%[a], #32]     \n"
                            :
                            : [a] "r"(a + i), [b] "r"(b + i)
                            : 
                        );
                    }
                }

                for(; i < len; i++){ // for remaining elements.
                    a[i] -= b[i];
                }

                return *this;
            }   
            constexpr Tensor& operator-=(const Tensor& x) noexcept {

                auto a = data; 
                auto b = x.getData(); 
                int i = 0;

                if(len > 8){
                    for (; i <= len - 8; i += 8) {
                        asm volatile(
                            "ldp q0, q1, [%[a], #0]      \n"
                            "ldp q2, q3, [%[a], #32]     \n"
                            "ldp q4, q5, [%[b], #0]      \n"
                            "ldp q6, q7, [%[b], #32]     \n"

                            "fsub.2d v0, v0, v4     \n"
                            "fsub.2d v1, v1, v5     \n"
                            "fsub.2d v2, v2, v6     \n"
                            "fsub.2d v3, v3, v7     \n"

                            "stp q0, q1, [%[a], #0]      \n"
                            "stp q2, q3, [%[a], #32]     \n"
                            :
                            : [a] "r"(a + i), [b] "r"(b + i)
                            : 
                        );
                    }
                }

                for(; i < len; i++){ // for remaining elements.
                    a[i] -= b[i];
                }

                return *this;
            }   

            constexpr Tensor& operator*=(const Tensor& x) noexcept {

                auto a = data; 
                auto b = x.getData(); 
                int i = 0;

                if(len == 1){
                    if(len > 8){
                        for (; i <= len - 8; i += 8) {
                            asm volatile(
                                "ldp q0, q1, [%[b], #0]      \n"
                                "ldp q2, q3, [%[b], #32]     \n"

                                "ldr d8, [%[a], #0]      \n"

                                "fmul.2d v0, v0, v8[0]     \n"
                                "fmul.2d v1, v1, v8[0]     \n"
                                "fmul.2d v2, v2, v8[0]     \n"
                                "fmul.2d v3, v3, v8[0]     \n"

                                "stp q0, q1, [%[b], #0]      \n"
                                "stp q2, q3, [%[b], #32]     \n"
                                :
                                : [a] "r"(a), [b] "r"(b + i)
                                : 
                            );
                        }
                    }

                    for(; i < len; i++){ // for remaining elements.
                        b[i] *= a[0];
                    }
                }

                else if(x.getLen() == 1){
                    if(len > 8){
                        for (; i <= len - 8; i += 8) {
                            asm volatile(
                                "ldp q0, q1, [%[a], #0]      \n"
                                "ldp q2, q3, [%[a], #32]     \n"

                                "ldr d8, [%[b], #0]      \n"

                                "fmul.2d v0, v0, v8[0]     \n"
                                "fmul.2d v1, v1, v8[0]     \n"
                                "fmul.2d v2, v2, v8[0]     \n"
                                "fmul.2d v3, v3, v8[0]     \n"

                                "stp q0, q1, [%[a], #0]      \n"
                                "stp q2, q3, [%[a], #32]     \n"
                                :
                                : [a] "r"(a + i), [b] "r"(b)
                                : 
                            );
                        }
                    }

                    for(; i < len; i++){ // for remaining elements.
                        a[i] *= b[0];
                    }
                }

                else{

                    if(len > 8){
                        for (; i <= len - 8; i += 8) {
                            asm volatile(
                                "ldp q0, q1, [%[a], #0]      \n"
                                "ldp q2, q3, [%[a], #32]     \n"
                                "ldp q4, q5, [%[b], #0]      \n"
                                "ldp q6, q7, [%[b], #32]     \n"

                                "fmul.2d v0, v0, v4     \n"
                                "fmul.2d v1, v1, v5     \n"
                                "fmul.2d v2, v2, v6     \n"
                                "fmul.2d v3, v3, v7     \n"

                                "stp q0, q1, [%[a], #0]      \n"
                                "stp q2, q3, [%[a], #32]     \n"
                                :
                                : [a] "r"(a + i), [b] "r"(b + i)
                                : 
                            );
                        }
                    }

                    for(; i < len; i++){ // for remaining elements.
                        a[i] *= b[i];
                    }

                }

                return *this;
            }   

            constexpr Tensor& operator/=(const Tensor& x) noexcept {

                auto a = data; 
                auto b = x.getData(); 
                int i = 0;

                if(len == 1){
                    if(len > 8){
                        for (; i <= len - 8; i += 8) {
                            asm volatile(
                                "ldp q0, q1, [%[b], #0]      \n"
                                "ldp q2, q3, [%[b], #32]     \n"

                                "ldr d8, [%[a], #0]      \n"
                                "ldr d9, [%[a], #0]      \n"

                                "fdiv.2d v0, v0, v4     \n"
                                "fdiv.2d v1, v1, v4     \n"
                                "fdiv.2d v2, v2, v4     \n"
                                "fdiv.2d v3, v3, v4     \n"

                                "stp q0, q1, [%[b], #0]      \n"
                                "stp q2, q3, [%[b], #32]     \n"
                                :
                                : [a] "r"(a), [b] "r"(b + i)
                                : 
                            );
                        }
                    }

                    for(; i < len; i++){ // for remaining elements.
                        b[i] /= a[0];
                    }
                }

                else if(x.getLen() == 1){
                    if(len > 8){
                        for (; i <= len - 8; i += 8) {
                            asm volatile(
                                "ldp q0, q1, [%[a], #0]      \n"
                                "ldp q2, q3, [%[a], #32]     \n"

                                "ldr d8, [%[b], #0]      \n"
                                "ldr d9, [%[b], #0]      \n"

                                "fdiv.2d v0, v0, v4     \n"
                                "fdiv.2d v1, v1, v4     \n"
                                "fdiv.2d v2, v2, v4     \n"
                                "fdiv.2d v3, v3, v4     \n"

                                "stp q0, q1, [%[a], #0]      \n"
                                "stp q2, q3, [%[a], #32]     \n"
                                :
                                : [a] "r"(a + i), [b] "r"(b)
                                : 
                            );
                        }
                    }

                    for(; i < len; i++){ // for remaining elements.
                        a[i] /= b[0];
                    }
                }

                else{

                    if(len > 8){
                        for (; i <= len - 8; i += 8) {
                            asm volatile(
                                "ldp q0, q1, [%[a], #0]      \n"
                                "ldp q2, q3, [%[a], #32]     \n"
                                "ldp q4, q5, [%[b], #0]      \n"
                                "ldp q6, q7, [%[b], #32]     \n"

                                "fdiv.2d v0, v0, v4     \n"
                                "fdiv.2d v1, v1, v5     \n"
                                "fdiv.2d v2, v2, v6     \n"
                                "fdiv.2d v3, v3, v7     \n"

                                "stp q0, q1, [%[a], #0]      \n"
                                "stp q2, q3, [%[a], #32]     \n"
                                :
                                : [a] "r"(a + i), [b] "r"(b + i)
                                : 
                            );
                        }
                    }

                    for(; i < len; i++){ // for remaining elements.
                        a[i] /= b[i];
                    }

                }

                return *this;
            }   

            constexpr Tensor eye(const int n) const {
                Tensor x({(unsigned int)(n), (unsigned int)(n)});

                auto a = x.getData();
                
                int i = 0; 
                for(; i < n - 8; i+=8){
                    asm volatile(
                        "ldp q0, q1, #1      \n"
                        "ldp q2, q3, #1      \n"

                        "stp q0, q1, [%[a], #0]      \n"
                        "stp q2, q3, [%[a], #32]     \n"

                        :
                        : [a] "r"(a + n * i + i)
                        : 
                    );

                }

                for(; i < n; i++){ // for remaining elements.
                    a[i * n + i] = 1; 
                }

                return x;
            }

            constexpr Tensor vdot(const Tensor& x) const {
                
                if (ndim != 1 || x.getNdim() != 1) {
                    throw std::invalid_argument("Tensors must be 1D vectors!");
                }

                double z = 0; 

                auto a = data;
                auto b = x.getData();

                int i = 0; 

                if(len > 8){
                    for (; i <= len - 8; i += 8) {
                        asm volatile(
                            "ldp q0, q1, [%[a], #0]      \n"
                            "ldp q2, q3, [%[a], #32]     \n"
                            "ldp q4, q5, [%[b], #0]      \n"
                            "ldp q6, q7, [%[b], #32]     \n"

                            "ldr d8, [%[c], #0]          \n"

                            "fmul.2d v0, v0, v4     \n"
                            "fmul.2d v1, v1, v5     \n"
                            "fmul.2d v2, v2, v6     \n"
                            "fmul.2d v3, v3, v7     \n"

                            "fadd.2d v0, v0, v1    \n"
                            "fadd.2d v0, v0, v2    \n"
                            "fadd.2d v0, v0, v3    \n"

                            "faddp.2d d0, v0          \n"

                            "fadd d8, d8, d0             \n"

                            "str d8, [%[c], #0]          \n"
                            :
                            : [a] "r"(a + i), [b] "r"(b + i), [c] "r"(&z)
                            : 
                        );
                    }
                }

                for(; i < len; i++){ // for remaining elements.
                    z += a[i] * b[i];
                }        

                return Tensor(z);
            }
            

        inline void matmul2x2(double* a, double* b, double* c) const {
            asm volatile ( 

                "ldp q0, q1, [%[a], #0]\n"    

                "ldp q2, q3, [%[b], #0]\n"    

                "movi v4.2d, #0 \n"   
                "movi v5.2d, #0 \n"  

                "fmla v4.2d, v2.2d, v0.d[0]\n"
                "fmla v4.2d, v3.2d, v0.d[1]\n"

                "fmla v5.2d, v2.2d, v1.d[0]\n"
                "fmla v5.2d, v3.2d, v1.d[1]\n"

                "stp q4, q5, [%[c], #0]\n"

                :
                : [a] "r" (a), [b] "r" (b), [c] "r" (c)
                :
            );
            
        }

        inline void matmul4x4(double* a, double* b, double* c) const {
            asm volatile ( 

                "ldp q0, q1, [%[a], #0]\n"    
                "ldp q2, q3, [%[a], #32]\n" 
                "ldp q4, q5, [%[a], #64]\n"    
                "ldp q6, q7, [%[a], #96]\n"    

                "ldp q8, q9, [%[b], #0]\n"    
                "ldp q10, q11, [%[b], #32]\n" 
                "ldp q12, q13, [%[b], #64]\n"    
                "ldp q14, q15, [%[b], #96]\n"   

                "movi v16.2d, #0 \n"   
                "movi v17.2d, #0 \n"  
                "movi v18.2d, #0 \n"   
                "movi v19.2d, #0 \n"  
                "movi v20.2d, #0 \n"   
                "movi v21.2d, #0 \n"  
                "movi v22.2d, #0 \n"   
                "movi v23.2d, #0 \n"  

                "fmla v16.2d, v8.2d, v0.d[0]\n"
                "fmla v16.2d, v10.2d, v0.d[1]\n"
                "fmla v16.2d, v12.2d, v1.d[0]\n"
                "fmla v16.2d, v14.2d, v1.d[1]\n"

                "fmla v17.2d, v9.2d, v0.d[0]\n"
                "fmla v17.2d, v11.2d, v0.d[1]\n"
                "fmla v17.2d, v13.2d, v1.d[0]\n"
                "fmla v17.2d, v15.2d, v1.d[1]\n"

                "fmla v18.2d, v8.2d, v2.d[0]\n"
                "fmla v18.2d, v10.2d, v2.d[1]\n"
                "fmla v18.2d, v12.2d, v3.d[0]\n"
                "fmla v18.2d, v14.2d, v3.d[1]\n"

                "fmla v19.2d, v9.2d, v2.d[0]\n"
                "fmla v19.2d, v11.2d, v2.d[1]\n"
                "fmla v19.2d, v13.2d, v3.d[0]\n"
                "fmla v19.2d, v15.2d, v3.d[1]\n"

                "fmla v20.2d, v8.2d, v4.d[0]\n"
                "fmla v20.2d, v10.2d, v4.d[1]\n"
                "fmla v20.2d, v12.2d, v5.d[0]\n"
                "fmla v20.2d, v14.2d, v5.d[1]\n"

                "fmla v21.2d, v9.2d, v4.d[0]\n"
                "fmla v21.2d, v11.2d, v4.d[1]\n"
                "fmla v21.2d, v13.2d, v5.d[0]\n"
                "fmla v21.2d, v15.2d, v5.d[1]\n"

                "fmla v22.2d, v8.2d, v6.d[0]\n"
                "fmla v22.2d, v10.2d, v6.d[1]\n"
                "fmla v22.2d, v12.2d, v7.d[0]\n"
                "fmla v22.2d, v14.2d, v7.d[1]\n"

                "fmla v23.2d, v9.2d, v6.d[0]\n"
                "fmla v23.2d, v11.2d, v6.d[1]\n"
                "fmla v23.2d, v13.2d, v7.d[0]\n"
                "fmla v23.2d, v15.2d, v7.d[1]\n"
                
                

                // "fmla v5.2d, v2.2d, v1.d[0]\n"
                // "fmla v5.2d, v3.2d, v1.d[1]\n"

                // "stp q4, q5, [%[c], #0]\n"

                "stp q16, q17, [%[c], #0]\n"    
                "stp q18, q19, [%[c], #32]\n" 
                "stp q20, q21, [%[c], #64]\n"    
                "stp q22, q23, [%[c], #96]\n"   

                :
                : [a] "r" (a), [b] "r" (b), [c] "r" (c)
                :
            );
            
        } 

        constexpr Tensor transpose() const {
            if (ndim != 2) {
                throw std::invalid_argument("Tensor must be a 2D matrix!");
            }

            unsigned int n = shape[0]; 
            unsigned int m = shape[1];

            Tensor y({m, n});

            auto a = data; 
            auto b = y.getData();

            for(int i = 0; i < n; i++){
                for(int j = 0; j < m; j++){
                    b[j * m + i] = a[i * n + j];
                }
            }

            return y; 


        }

        constexpr Tensor matmul(const Tensor& x) const {
            if (ndim != 2 || x.getNdim() != 2) {
                throw std::invalid_argument("Tensors must be 2D matrices!");
            }
            if(shape[1] != x.shape[0]) {
                throw std::invalid_argument("Incompatible dimensions for matrix multiplication!");
            }

            unsigned int n = shape[0]; 
            unsigned int l = shape[1];
            unsigned int m = x.shape[1];

            Tensor y({n, m});

            auto a = data; 
            auto b = x.getData();
            auto c = y.getData();

            // VECTORIZE THIS
            // MAYBE ADD AN EXTRA INITIALIZER TO NOT SET data TO 0 => truf


            // 122MS...
            // Trying to make it faster... 
            //const double OFFSETS[3] = {double(n * itemSize), double(2 * n * itemSize), double(3 * n * itemSize)};

            // for (int i = 0; i <= n - 4; i+=4) {  
            //     for (int j = 0; j <= m - 4; j += 4) {  

            //             double *c_ptr0 =  &c[i * m + j];
            //             double *c_ptr1 =  &c[(i + 1) * m + j];
            //             double *c_ptr2 = &c[(i + 2) * m + j];
            //             double *c_ptr3 = &c[(i + 3) * m + j];

            //             asm volatile (
            //                 "movi v16.2d, #0 \n"   
            //                 "movi v17.2d, #0 \n"  
            //                 "movi v18.2d, #0 \n"   
            //                 "movi v19.2d, #0 \n"  
            //                 "movi v20.2d, #0 \n"   
            //                 "movi v21.2d, #0 \n"  
            //                 "movi v22.2d, #0 \n"   
            //                 "movi v23.2d, #0 \n"  
            //                 :
            //                 :
            //                 :
            //             );

            //         for (int k = 0; k <= l - 4; k+= 4) { 
            //             double *a_ptr0 = &a[i * l + k];
            //             double *a_ptr1 = &a[(i + 1)* l + k];
            //             double *a_ptr2 = &a[(i + 2) * l + k];
            //             double *a_ptr3 = &a[(i + 3 )* l + k];

            //             double *b_ptr0 = &b[k * m + j];
            //             double *b_ptr1 = &b[(k + 1) * m + j];
            //             double *b_ptr2 = &b[(k + 2) * m + j];
            //             double *b_ptr3 = &b[(k + 3) * m + j];

            //             //matmul4x4(a_ptr, b_ptr, c_ptr);


            //             asm volatile ( 
            //                 "ldp q0, q1, [%[a0], #0]\n"  
            //                 "ldp q2, q3, [%[a1], #0]\n" 
            //                 "ldp q4, q5, [%[a2], #0]\n"  
            //                 "ldp q6, q7, [%[a3], #0]\n"  

            //                 "ldp q8, q9, [%[b0], #0]\n"    
            //                 "ldp q10, q11, [%[b1], #0]\n" 
            //                 "ldp q12, q13, [%[b2], #0]\n"    
            //                 "ldp q14, q15, [%[b3], #0]\n"   

            //                 "fmla v16.2d, v8.2d, v0.d[0]\n"
            //                 "fmla v16.2d, v10.2d, v0.d[1]\n"
            //                 "fmla v16.2d, v12.2d, v1.d[0]\n"
            //                 "fmla v16.2d, v14.2d, v1.d[1]\n"

            //                 "fmla v17.2d, v9.2d, v0.d[0]\n"
            //                 "fmla v17.2d, v11.2d, v0.d[1]\n"
            //                 "fmla v17.2d, v13.2d, v1.d[0]\n"
            //                 "fmla v17.2d, v15.2d, v1.d[1]\n"

            //                 "fmla v18.2d, v8.2d, v2.d[0]\n"
            //                 "fmla v18.2d, v10.2d, v2.d[1]\n"
            //                 "fmla v18.2d, v12.2d, v3.d[0]\n"
            //                 "fmla v18.2d, v14.2d, v3.d[1]\n"

            //                 "fmla v19.2d, v9.2d, v2.d[0]\n"
            //                 "fmla v19.2d, v11.2d, v2.d[1]\n"
            //                 "fmla v19.2d, v13.2d, v3.d[0]\n"
            //                 "fmla v19.2d, v15.2d, v3.d[1]\n"

            //                 "fmla v20.2d, v8.2d, v4.d[0]\n"
            //                 "fmla v20.2d, v10.2d, v4.d[1]\n"
            //                 "fmla v20.2d, v12.2d, v5.d[0]\n"
            //                 "fmla v20.2d, v14.2d, v5.d[1]\n"

            //                 "fmla v21.2d, v9.2d, v4.d[0]\n"
            //                 "fmla v21.2d, v11.2d, v4.d[1]\n"
            //                 "fmla v21.2d, v13.2d, v5.d[0]\n"
            //                 "fmla v21.2d, v15.2d, v5.d[1]\n"

            //                 "fmla v22.2d, v8.2d, v6.d[0]\n"
            //                 "fmla v22.2d, v10.2d, v6.d[1]\n"
            //                 "fmla v22.2d, v12.2d, v7.d[0]\n"
            //                 "fmla v22.2d, v14.2d, v7.d[1]\n"

            //                 "fmla v23.2d, v9.2d, v6.d[0]\n"
            //                 "fmla v23.2d, v11.2d, v6.d[1]\n"
            //                 "fmla v23.2d, v13.2d, v7.d[0]\n"
            //                 "fmla v23.2d, v15.2d, v7.d[1]\n"

            //                 :
            //                 : [a0] "r" (a_ptr0), [a1] "r" (a_ptr1), [a2] "r" (a_ptr2), [a3] "r" (a_ptr3), [b0] "r" (b_ptr0), [b1] "r" (b_ptr1), [b2] "r" (b_ptr2), [b3] "r" (b_ptr3)
            //                 :
            //             );
            //         }
            //         asm volatile(
            //             "stp q16, q17, [%[c0], #0]\n"    
            //             "stp q18, q19, [%[c1], #0]\n" 
            //             "stp q20, q21, [%[c2], #0]\n"    
            //             "stp q22, q23, [%[c3], #0]\n"   
            //             :
            //             : [c0] "r" (c_ptr0), [c1] "r" (c_ptr1), [c2] "r" (c_ptr2), [c3] "r" (c_ptr3)
            //             :
            //         );
            //     }
            // }
            // int leftover_rows_a = n % 4;  // Leftover rows in a
            // int leftover_cols_b = m % 4;  // Leftover columns in b
            // int leftover_l = l % 4;       // Leftover columns in a / rows in b

            // // Handle leftover rows of a
            // if (leftover_rows_a > 0) {
            //     for (int i = n - leftover_rows_a; i < n; ++i) {
            //         for (int j = 0; j < m; ++j) {
            //             float sum = 0.0f;
            //             for (int k = 0; k < l; ++k) {
            //                 sum += a[i * l + k] * b[k * m + j];
            //             }
            //             c[i * m + j] += sum;
            //         }
            //     }
            // }

            // // Handle leftover columns of b
            // if (leftover_cols_b > 0) {
            //     for (int i = 0; i < n; ++i) {
            //         for (int j = m - leftover_cols_b; j < m; ++j) {
            //             float sum = 0.0f;
            //             for (int k = 0; k < l; ++k) {
            //                 sum += a[i * l + k] * b[k * m + j];
            //             }
            //             c[i * m + j] += sum;
            //         }
            //     }
            // }

            // // Handle leftover inner loop dimension
            // if (leftover_l > 0) {
            //     for (int i = 0; i < n; ++i) {
            //         for (int j = 0; j < m; ++j) {
            //             float sum = 0.0f;
            //             for (int k = l - leftover_l; k < l; ++k) {
            //                 sum += a[i * l + k] * b[k * m + j];
            //             }
            //             c[i * m + j] += sum;
            //         }
            //     }
            // }

            for(unsigned int i = 0; i < n; i++){
                for(unsigned int j = 0; j < m; j++){
                    for(unsigned int k = 0; k < l; k++){
                        // y[{i, j}] += (*this)[{i, k}] * x[{j, k}];
                        c[i * n + j] += (a[i * n + k] * b[j * m + k]);
                    }
                }
            }


            return y;  
        }

        // inline Tensor matmul(const Tensor& x) const {
        //     std::cout << "here " << "\n";
        //     if (ndim != 2 || x.getNdim() != 2) {
        //         throw std::invalid_argument("Tensors must be 2D matrices!");
        //     }
        //     if(shape[1] != x.shape[0]) {
        //         throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
        //     }

        //     unsigned int n = shape[0]; 
        //     unsigned int l = shape[1];
        //     unsigned int m = x.shape[0];

        //     Tensor y({n, m});

        //     auto a = data; 
        //     auto b = x.getData();
        //     auto c = y.getData();

        //     // VECTORIZE THIS
        //     // MAYBE ADD AN EXTRA INITIALIZER TO NOT SET data TO 0 => truf
        //     // MAYBE CHANGE (*this) structure.. => lol nah

        //     // les put a pin in this.
        //     for (int i = 0; i < n; i++) {  
        //         for (int j = 0; j < m; j += 38) {  
        //             double *ptr_c = &c[i * m + j];
        //             asm volatile (
        //                 "movi v17.2d, #0 \n" 
        //                 "movi v18.2d, #0 \n"   
        //                 "movi v19.2d, #0 \n"
        //                 "movi v20.2d, #0 \n"

        //                 "movi v21.2d, #0 \n" 
        //                 "movi v22.2d, #0 \n"   
        //                 "movi v23.2d, #0 \n"
        //                 "movi v24.2d, #0 \n"
                        
        //                 "movi v25.2d, #0 \n"
        //                 "movi v26.2d, #0 \n"
        //                 "movi v27.2d, #0 \n"

        //                 : // No output
        //                 :
        //                 :
        //             );

        //             for (int k = 0; k < l; k++) {  
        //                 double *ptr_a = &a[i * l + k];

        //                 double *ptr_b = &b[k * m + j];  

        //                 asm volatile (
        //                     // "ldr d0, [%[ptr_a]]\n"         
        //                     "ldr d0, [%[ptr_a]]\n"         

        //                     "ldp q1, q2, [%[ptr_b], #0]\n"    
        //                     "ldp q3, q4, [%[ptr_b], #32]\n"
        //                     "ldp q5, q6, [%[ptr_b], #64]\n"
        //                     "ldp q7, q8, [%[ptr_b], #96]\n"

        //                     "ldp q9, q10, [%[ptr_b], #128]\n"
        //                     "ldp q11, q12, [%[ptr_b], #160]\n"
        //                     "ldp q13, q14, [%[ptr_b], #192]\n"
        //                     "ldp q15, q16, [%[ptr_b], #224]\n"

        //                     "ldp q28, q29, [%[ptr_b], #224]\n"
        //                     "ldr q30, [%[ptr_b], #288]\n"

        //                     "fmla v17.2d, v1.2d, v0.d[0]\n"
        //                     "fmla v18.2d, v2.2d, v0.d[0]\n"
        //                     "fmla v19.2d, v3.2d, v0.d[0]\n"
        //                     "fmla v20.2d, v4.2d, v0.d[0]\n"

        //                     "fmla v21.2d, v5.2d, v0.d[0]\n"
        //                     "fmla v22.2d, v6.2d, v0.d[0]\n"
        //                     "fmla v23.2d, v7.2d, v0.d[0]\n"
        //                     "fmla v24.2d, v8.2d, v0.d[0]\n"

        //                     "fmla v17.2d, v9.2d, v0.d[0]\n"
        //                     "fmla v18.2d, v10.2d, v0.d[0]\n"
        //                     "fmla v19.2d, v11.2d, v0.d[0]\n"
        //                     "fmla v20.2d, v12.2d, v0.d[0]\n"

        //                     "fmla v21.2d, v13.2d, v0.d[0]\n"
        //                     "fmla v22.2d, v14.2d, v0.d[0]\n"
        //                     "fmla v23.2d, v15.2d, v0.d[0]\n" 
        //                     "fmla v24.2d, v16.2d, v0.d[0]\n" 

        //                     "fmla v25.2d, v28.2d, v0.d[0]\n" 
        //                     "fmla v26.2d, v29.2d, v0.d[0]\n" 
        //                     "fmla v27.2d, v30.2d, v0.d[0]\n" 
        //                     : // No output
        //                     : [ptr_a] "r" (ptr_a), [ptr_b] "r" (ptr_b)
        //                     :
        //                 );
        //             }

        //             // Store the result in C[i, j] and C[i, j+1]
        //             asm volatile (
        //                 "stp q17, q18, [%[ptr_c], #0]\n"
        //                 "stp q19, q20, [%[ptr_c], #32]\n"
        //                 "stp q21, q22, [%[ptr_c], #64]\n"
        //                 "stp q23, q24, [%[ptr_c], #96]\n"
        //                 "stp q25, q26, [%[ptr_c], #128]\n"
        //                 "str q27, [%[ptr_c], #160]\n"
        //                 :
        //                 : [ptr_c] "r" (ptr_c)
        //                 : 
        //             );
        //         }
        //     }

        //     return y;
        // }


            constexpr Tensor cross(const Tensor& x) const {
                
                if (ndim != 1 || x.getNdim() != 1) {
                    throw std::invalid_argument("Tensors must be 1D vectors!");
                }

                if (shape[0] != 3 || x.getShape()[0] != 3) {
                    std::cout << shape[1] << "\n";
                    std::cout << x.getShape()[1] << "\n";
                    throw std::invalid_argument("Vectors must be in R^3!");
                }

                Tensor y({3});

                auto a = data; 
                auto b = x.getData();
                auto c = y.getData();


                 c[0] = a[1] * b[2] - a[2] * b[1]; 
                 c[1] = a[0] * b[2] - a[2] * b[0];
                 c[2] = a[0] * b[1] - a[1] * b[0];

                 return y;
            }

            // constexpr Tensor dot(const Tensor& x) const noexcept {
            //     Tensor y(*this);
            //     return x;
            // }
    
        private:
            double* data; 
            unsigned int* shape; 
            unsigned int* strides;

            size_t itemSize;
            unsigned int ndim;
            unsigned int len;

            template <class T>
            constexpr void processArray(const std::vector<T>& ndarray, std::vector<double>& flattenedArray) const noexcept {
                for(auto& component : ndarray){
                    if constexpr (std::is_arithmetic<T>::value) {
                        flattenedArray.push_back(component);
                    }   
                    else{
                        processArray(component, flattenedArray);
                    }
                }
            }


            template <class T>
            constexpr void calculateDims(const std::vector<T>& ndarray, std::vector<unsigned int>& dims) const noexcept {
                dims.push_back(ndarray.size());

                for(auto& component : ndarray){
                    if constexpr (std::is_arithmetic<T>::value) {
                        break;
                    }   
                    else{
                        calculateDims(component, dims);
                    }
                    break;
                }
            }

                // if constexpr (!std::is_arithmetic<T>::value) {  
                //     if (ndarray.size() != 0) {
                //         calculateDims(*ndarray.begin(), dims);
                //     }
                // }

            // this will create the ndarray's shape and strides -- hence "tensor structure"
            constexpr void createTensorStructure(const std::vector<unsigned int>& dims) noexcept {
                shape = new unsigned int[ndim];

                int i = ndim - 1; 

                shape[i] = dims[i];
                strides = new unsigned int[ndim];

                strides[i] = itemSize;

                if(ndim > 1){
                    i--;
                    for (; i >= 0; i--) {
                        unsigned int prevDim = dims[i + 1];
                        strides[i] = strides[i + 1] * prevDim;
                        
                        shape[i] = dims[i];
                    }
                }
            }

            constexpr void createTensorStructure(unsigned int* dims) noexcept {
                shape = new unsigned int[ndim];

                int i = ndim - 1; 

                shape[i] = dims[i];
                strides = new unsigned int[ndim];

                strides[i] = itemSize;

                if(ndim > 1){
                    i--;
                    for (; i >= 0; i--) {
                        unsigned int prevDim = dims[i + 1];
                        strides[i] = strides[i + 1] * prevDim;
                        
                        shape[i] = dims[i];
                    }
                }
            }

            // Im gonna put a pin in this.
            // constexpr Tensor& operator[](const std::vector<unsigned int>& indices) noexcept {
            //     // make sure indices size !> ndim.
            //     // ill think of some cool exception to throw 

            //     std::vector<unsigned int> dims;
            //     unsigned int numIndexedDims = indices.size();

            //     for(int i = numIndexedDims; i < ndim; i++){
            //         dims.push_back(i);
            //     }

            //     unsigned int idx = 0; 

            //     for(int i = 0; i < numIndexedDims; i++){
            //         idx += indices[i] * strides[i];
            //     }
    
            //     double data[len - idx];

            //     for(int i = idx; i < len; i++){
            //         data[i - idx] = this->data[i];
            //     }

            //     return Tensor(data, dims);
                
            // }

            // constexpr Tensor operator[](const std::vector<unsigned int>& indices) const noexcept {
                
            // }

    };

    constexpr Tensor operator+(const Tensor& x) noexcept {
        return x;
    }

    constexpr Tensor operator-(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i <= y.getLen() - 8; i += 8) {
                asm volatile(
                    "ldp q0, q1, [%[a], #0]      \n"
                    "ldp q2, q3, [%[a], #32]     \n"
                    "ldp q4, q5, [%[a], #64]      \n"
                    "ldp q6, q7, [%[a], #96]     \n"

                    "fneg.2d v0, v0     \n"
                    "fneg.2d v1, v1     \n"
                    "fneg.2d v2, v2     \n"
                    "fneg.2d v3, v3     \n"
                    "fneg.2d v4, v4     \n"
                    "fneg.2d v5, v5     \n"
                    "fneg.2d v6, v6     \n"
                    "fneg.2d v7, v7     \n"
                    // "fdiv.2d v2, v2, v6     \n"
                    // "fdiv.2d v3, v3, v7     \n"

                    "stp q0, q1, [%[a], #0]      \n"
                    "stp q2, q3, [%[a], #32]     \n"
                    "stp q4, q5, [%[a], #64]      \n"
                    "stp q6, q7, [%[a], #96]     \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
            }
        }

        for(; i < y.getLen(); i++){ // for remaining elements.
            a[i] *= -1;
        }

        return y;
    }

    constexpr Tensor operator+(const Tensor& x, const Tensor& y) noexcept {
        Tensor z(x); 
        z += y;
        return z; 
    }

    constexpr Tensor operator-(const Tensor& x, const Tensor& y) noexcept {
        Tensor z(x); 
        z -= y;
        return z; 
    }

    constexpr Tensor operator*(const Tensor& x, const Tensor& y) noexcept {
        Tensor z(x); 
        z *= y;
        return z; 
    }

    constexpr Tensor operator/(const Tensor& x, const Tensor& y) noexcept {
        Tensor z(x); 
        z /= y;
        return z; 
    }

    constexpr Tensor add(const Tensor& x, const Tensor& y) noexcept {
        return x + y;
    }

    constexpr Tensor sub(const Tensor& x, const Tensor& y) noexcept {
        return x - y;
    }

    constexpr Tensor mul(const Tensor& x, const Tensor& y) noexcept {
        return x * y;
    }

    constexpr Tensor div(const Tensor& x, const Tensor& y) noexcept {
        return x / y;
    }

    constexpr Tensor vdot(const Tensor& x, const Tensor& y) noexcept {
        return x.vdot(y);
    }

    constexpr Tensor transpose(const Tensor& x) noexcept {
        return x.transpose();
    }

    constexpr Tensor matmul(const Tensor& x, const Tensor& y) noexcept {
        return x.matmul(y);
    }

    constexpr Tensor cross(const Tensor& x, const Tensor& y) noexcept {
        return x.cross(y);
    }

    constexpr Tensor eye(const int n) {
        Tensor x({(unsigned int)(n), (unsigned int)(n)});

        auto a = x.getData();
        
        int i = 0; 
        for(; i < n - 8; i+=8){
            asm volatile(
                "movi v0.2d, #0 \n"   
                "movi v1.2d, #0 \n"  
                "movi v2.2d, #0 \n"  
                "movi v3.2d, #0 \n"  

                "stp q0, q1, [%[a], #0]      \n"
                "stp q2, q3, [%[a], #32]     \n"

                :
                : [a] "r"(a + n * i + i)
                : 
            );

        }

        for(; i < n; i++){ // for remaining elements.
            a[i * n + i] = 1; 
        }

        return x;
    }

    inline Tensor exp(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

    constexpr Tensor exp2(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _exp2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

    constexpr Tensor expm1(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _expm1    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _expm1    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _expm1    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _expm1    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _expm1    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _expm1    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _expm1    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _expm1    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

constexpr Tensor log(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

constexpr Tensor log10(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log10    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log10    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log10    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log10    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log10    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log10    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log10    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log10    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

constexpr Tensor log2(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _log2    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

    constexpr Tensor sin(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sin    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sin    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sin    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sin    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sin    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sin    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sin    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sin    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

constexpr Tensor cos(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cos    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cos    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cos    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cos    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cos    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cos    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cos    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cos    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

constexpr Tensor tan(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tan    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tan    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tan    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tan    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tan    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tan    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tan    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tan    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

    constexpr Tensor asin(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asin    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i), [b] "r"(a + i)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asin    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 1), [b] "r"(a + i + 1)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asin    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 2), [b] "r"(a + i + 2)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asin    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 3), [b] "r"(a + i + 3)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asin    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 4), [b] "r"(a + i + 4)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asin    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 5), [b] "r"(a + i + 5)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asin    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 6), [b] "r"(a + i + 6)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asin    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 7), [b] "r"(a + i + 7)
                    :
                );
            }
        }

        return y;
    }

    constexpr Tensor acos(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acos    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i), [b] "r"(a + i)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acos    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 1), [b] "r"(a + i + 1)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acos    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 2), [b] "r"(a + i + 2)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acos    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 3), [b] "r"(a + i + 3)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acos    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 4), [b] "r"(a + i + 4)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acos    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 5), [b] "r"(a + i + 5)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acos    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 6), [b] "r"(a + i + 6)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acos    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 7), [b] "r"(a + i + 7)
                    :
                );
            }
        }

        return y;
    }

    constexpr Tensor atan(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atan    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i), [b] "r"(a + i)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atan    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 1), [b] "r"(a + i + 1)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atan    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 2), [b] "r"(a + i + 2)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atan    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 3), [b] "r"(a + i + 3)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atan    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 4), [b] "r"(a + i + 4)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atan    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 5), [b] "r"(a + i + 5)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atan    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 6), [b] "r"(a + i + 6)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atan    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 7), [b] "r"(a + i + 7)
                    :
                );
            }
        }

        return y;
    }

    constexpr Tensor atan2(const Tensor& x, const Tensor& y) noexcept {
        Tensor z(x);

        auto a = x.getData();
        auto b = y.getData();
        auto c = z.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i++) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "ldr d1, [%[b], #0]     \n"
                    "bl _atan2    \n"  
                    "str d0, [%[c], #0]      \n"
                    :
                    : [a] "r"(a + i), [b] "r"(b + i), [c] "r"(c + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "ldr d1, [%[b], #0]     \n"
                    "bl _atan2    \n"  
                    "str d0, [%[c], #0]      \n"
                    :
                    : [a] "r"(a + i + 1), [b] "r"(b + i + 1), [c] "r"(c + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "ldr d1, [%[b], #0]     \n"
                    "bl _atan2    \n"  
                    "str d0, [%[c], #0]      \n"
                    :
                    : [a] "r"(a + i + 2), [b] "r"(b + i + 2), [c] "r"(c + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "ldr d1, [%[b], #0]     \n"
                    "bl _atan2    \n"  
                    "str d0, [%[c], #0]      \n"
                    :
                    : [a] "r"(a + i + 3), [b] "r"(b + i + 3), [c] "r"(c + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "ldr d1, [%[b], #0]     \n"
                    "bl _atan2    \n"  
                    "str d0, [%[c], #0]      \n"
                    :
                    : [a] "r"(a + i + 4), [b] "r"(b + i + 4), [c] "r"(c + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "ldr d1, [%[b], #0]     \n"
                    "bl _atan2    \n"  
                    "str d0, [%[c], #0]      \n"
                    :
                    : [a] "r"(a + i + 5), [b] "r"(b + i + 5), [c] "r"(c + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "ldr d1, [%[b], #0]     \n"
                    "bl _atan2    \n"  
                    "str d0, [%[c], #0]      \n"
                    :
                    : [a] "r"(a + i + 6), [b] "r"(b + i + 6), [c] "r"(c + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "ldr d1, [%[b], #0]     \n"
                    "bl _atan2    \n"  
                    "str d0, [%[c], #0]      \n"
                    :
                    : [a] "r"(a + i + 7), [b] "r"(b + i + 7), [c] "r"(c + i + 7)
                    : 
                );
            }
        }

        return z;
    }

    constexpr Tensor sinh(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sinh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sinh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sinh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sinh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sinh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sinh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sinh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _sinh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

    constexpr Tensor cosh(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cosh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cosh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cosh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cosh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cosh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cosh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cosh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _cosh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

    constexpr Tensor tanh(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tanh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tanh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 1)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tanh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 2)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tanh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 3)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tanh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 4)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tanh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 5)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tanh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 6)
                    : 
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _tanh    \n"  
                    "str d0, [%[a], #0]      \n"
                    :
                    : [a] "r"(a + i + 7)
                    : 
                );
            }
        }

        return y;
    }

    constexpr Tensor asinh(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asinh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i), [b] "r"(a + i)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asinh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 1), [b] "r"(a + i + 1)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asinh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 2), [b] "r"(a + i + 2)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asinh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 3), [b] "r"(a + i + 3)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asinh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 4), [b] "r"(a + i + 4)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asinh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 5), [b] "r"(a + i + 5)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asinh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 6), [b] "r"(a + i + 6)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _asinh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 7), [b] "r"(a + i + 7)
                    :
                );
            }
        }

        return y;
    }

    constexpr Tensor acosh(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acosh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i), [b] "r"(a + i)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acosh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 1), [b] "r"(a + i + 1)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acosh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 2), [b] "r"(a + i + 2)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acosh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 3), [b] "r"(a + i + 3)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acosh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 4), [b] "r"(a + i + 4)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acosh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 5), [b] "r"(a + i + 5)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acosh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 6), [b] "r"(a + i + 6)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _acosh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 7), [b] "r"(a + i + 7)
                    :
                );
            }
        }

        return y;
    }

    constexpr Tensor atanh(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atanh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i), [b] "r"(a + i)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atanh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 1), [b] "r"(a + i + 1)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atanh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 2), [b] "r"(a + i + 2)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atanh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 3), [b] "r"(a + i + 3)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atanh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 4), [b] "r"(a + i + 4)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atanh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 5), [b] "r"(a + i + 5)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atanh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 6), [b] "r"(a + i + 6)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _atanh    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 7), [b] "r"(a + i + 7)
                    :
                );
            }
        }

        return y;
    }

    constexpr Tensor round(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _round    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i), [b] "r"(a + i)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _round    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 1), [b] "r"(a + i + 1)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _round    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 2), [b] "r"(a + i + 2)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _round    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 3), [b] "r"(a + i + 3)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _round    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 4), [b] "r"(a + i + 4)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _round    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 5), [b] "r"(a + i + 5)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _round    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 6), [b] "r"(a + i + 6)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _round    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 7), [b] "r"(a + i + 7)
                    :
                );
            }
        }

        return y;
    }

    constexpr Tensor abs(const Tensor& x) noexcept {
        Tensor y(x);

        auto a = y.getData();

        int i = 0;
        if(y.getLen() > 8){
            for (; i < y.getLen(); i+=8) {
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _abs    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i), [b] "r"(a + i)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _abs    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 1), [b] "r"(a + i + 1)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _abs    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 2), [b] "r"(a + i + 2)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _abs    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 3), [b] "r"(a + i + 3)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _abs    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 4), [b] "r"(a + i + 4)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _abs    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 5), [b] "r"(a + i + 5)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _abs    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 6), [b] "r"(a + i + 6)
                    :
                );
                asm volatile(
                    "ldr d0, [%[a], #0]     \n"
                    "bl _abs    \n"  
                    "str d0, [%[b], #0]      \n"
                    :
                    : [a] "r"(a + i + 7), [b] "r"(a + i + 7)
                    :
                );
            }
        }

        return y;
    }

    // constexpr Tensor operator+(const Tensor& x, const Tensor& y) noexcept {
    //     auto a = x.getData(); 
    //     auto b = y.getData(); 

    //     double* c = new double[x.getLen()]();
    //     int i = 0;

    //     if(x.getLen() > 8){
    //         for (; i <= x.getLen() - 8; i += 8) {
    //             asm volatile(
    //                 "ldp q0, q1, [%[a], #0]      \n"
    //                 "ldp q2, q3, [%[a], #32]     \n"
    //                 "ldp q4, q5, [%[b], #0]      \n"
    //                 "ldp q6, q7, [%[b], #32]     \n"

    //                 "fadd.2d v0, v0, v4     \n"
    //                 "fadd.2d v1, v1, v5     \n"
    //                 "fadd.2d v2, v2, v6     \n"
    //                 "fadd.2d v3, v3, v7     \n"

    //                 "stp q0, q1, [%[c], #0]      \n"
    //                 "stp q2, q3, [%[c], #32]     \n"
    //                 :
    //                 : [a] "r"(a + i), [b] "r"(b + i), [c] "r"(c + i)
    //                 : 
    //             );
    //         }

    //         for(; i < x.getLen(); i++){ // for remaining elements.
    //             c[i] = a[i] + b[i];
    //         }
    //     }
    //     return Tensor(c, x.getShape(), x.getNdim());

    // }

}

#endif // TENSOR_CPP