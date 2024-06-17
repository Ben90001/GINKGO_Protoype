#include <ginkgo/ginkgo.hpp>
#include <iostream>

int main() {
    std::cout<<"-------------------------------Ah shit, here we go again:---------------------------------------"<<std::endl;
    using Mtx = gko::matrix::Csr<double, int>;
    using Vec = gko::matrix::Dense<double>;
    
    auto exec = gko::ReferenceExecutor::create();   // g++ works upon till here (gcc,clang,g++-14 do NOT: probably just not compiling if not needed or sth like that?)
    auto mtx = gko::initialize<Mtx>({{0.0, 1.0}, {1.0, 0.0}}, exec);
    gko::write(std::cout, mtx);
    auto vec = gko::initialize<Vec>({2.0, 3.0}, exec);
    gko::write(std::cout, vec);
    auto result = vec->clone();                     //"->": dereference vec and call its method clone()
    mtx->apply(vec, result);
    gko::write(std::cout, result);

    std::cout<<"------------------------------At last we emerge victorious!-------------------------------------"<<std::endl;
    return 0;
}