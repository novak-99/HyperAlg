#include <iostream>

#include <Tensor>
#include <LinAlg>
using namespace HyperAlg;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

int main(){


    // std::vector<unsigned int> dimensions = {2,2,2};

    // std::vector<std::vector<std::vector<double>>> v = {{{2,2.0,3.0}, {2.0,2.0,3}}};
    // Tensor t( v );
    // t += t;
    // std::cout << t[{0, 0, 0}] << "\n";

    // std::cout << t[{0, 0, 0}] << "\n";


    // const int N = 1000;
    // std::vector<std::vector<double>> mat_(N);
    // std::vector<double> vec_(N);
    // for(int i = 0; i < N; i++){
    //     mat_[i] = vec_;
    // }

    // //     std::vector<std::vector<double>> mat_2(2);
    // // std::vector<double> vec_2(N);
    // // for(int i = 0; i < N; i++){
    // //     mat_2[i] = vec_2;
    // // }

    // auto v1 = mat_;
    // // auto v2 = mat_2;

    // Tensor t1(v1);
    // Tensor t2(v1);

    // std::vector<std::vector<double>> v1 = {
    //     {0, 1, 2, 3, 4, 5, 6, 7},
    //     {8, 9, 10, 11, 12, 13, 14, 15},
    //     {16, 17, 18, 19, 20, 21, 22, 23},
    //     {24, 25, 26, 27, 28, 29, 30, 31},
    //     {32, 33, 34, 35, 36, 37, 38, 39},
    //     {40, 41, 42, 43, 44, 45, 46, 47},
    //     {48, 49, 50, 51, 52, 53, 54, 55},
    //     {56, 57, 58, 59, 60, 61, 62, 63}
    // };

    // Tensor t1(v1);
    // Tensor t2(v1);

    // std::vector<std::vector<double>> A = {
    //     {1, 2, 3, 4, 5},
    //     {6, 7, 8, 9, 10},
    //     {11, 12, 13, 14, 15},
    //     {16, 17, 18, 19, 20}
    // };

    // // Define matrix B (5x4)
    // std::vector<std::vector<double>> B = {
    //     {1, 2, 3, 4},
    //     {5, 6, 7, 8},
    //     {9, 10, 11, 12},
    //     {13, 14, 15, 16},
    //     {17, 18, 19, 20}
    // };

    // Tensor t1(A);
    // Tensor t2(B);



//     auto time1 = high_resolution_clock::now();
//     t1.transpose();
//     auto time2 = high_resolution_clock::now();

// //     std::cout << t1.getShape()[0] << " " << t1.getShape()[1] << "\n";
// //     std::cout << t2.getShape()[0] << " " << t2.getShape()[1] << "\n";
// //     std::cout << a << "\n";
// //     std::cout << t1.matmul(t2)[{0, 0}] << "\n";
//     duration<double, std::milli> ms_double = time2 - time1;
//     std::cout << ms_double.count() << "ms\n";




//     return 0; 
// }

    // std::vector<std::vector<double>> A = {{0.5, 0.75, 0.5}, {1.0,0.5,0.75}, {0.25,0.25,0.25}};
    // Tensor t1(A);
    // auto [q, r] = qr(t1);

    // Tensor t1 = eye(3);


    std::vector<std::vector<double>> v = {{4,3}, {6,3}};
    Tensor t1(v);
    auto [l, u] = lu(t1);

    for(unsigned int i = 0; i < 2; i++){
        for(unsigned int j = 0; j < 2; j++){
            std::cout << u[{i, j}] << " ";
        }
        std::cout << "\n";
    }

    // std::cout << norm(t1)[{0}] << "\n";
}