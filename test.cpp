#include <iostream>

#include <Tensor>
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


    const int N = 1e+8;
    std::vector<double> v(N, 1);

    Tensor a(v);
    Tensor b(v);

    auto time1 = high_resolution_clock::now();
    exp(a);
    auto time2 = high_resolution_clock::now();

    duration<double, std::milli> ms_double = time2 - time1;
    std::cout << ms_double.count() << "ms\n";

    // for(unsigned int i = 0; i < x.getLen(); i++) std::cout << x[{i}] << "\n";

//     return 0; 
// }
}