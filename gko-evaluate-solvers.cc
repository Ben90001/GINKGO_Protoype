#include<ginkgo/ginkgo.hpp>

#include<iostream>
#include<chrono>


 
template <class MatrixType,typename CoefficientFunction, typename BoundaryTypeFunction>
std::unique_ptr<MatrixType> diffusion_matrix (const size_t n, const size_t d,
    CoefficientFunction diffusion_coefficient,
    BoundaryTypeFunction dirichlet_boundary,
    std::shared_ptr<gko::ReferenceExecutor> exec)
{
  // relevant types
  //using MatrixEntry = double;
  using mtx = MatrixType;


  // prepare grid information
  std::vector<std::size_t> sizes(d+1,1);
  for (int i=1; i<=d; ++i) sizes[i] = sizes[i-1]*n;         /// sizes={1,n,n^2,n^3,...,n^d} => time needed for total concentration equilibirum distance^d
  double mesh_size = 1.0/n;
  int N = sizes[d];

  // create matrix entries
  //gko::matrix_data<double,size_t> mtx_data{gko::dim<2,size_t>(N,N)};     //temporary COO representation (!might be unefficient) @changed
  gko::matrix_data<> mtx_data{gko::dim<2>(N)};              ///@changed @perfomance->passing size_t as template parameter to dim significant slowdown (why??)
  for (std::size_t index=0; index<sizes[d]; index++)        ///each grid cell
  {
    // create multiindex from row number                    ///fancy way of doing 3 for loops over n -> more powerful: works for all d
    std::vector<std::size_t> multiindex(d,0);
    auto copiedindex=index;
    for (int i=d-1; i>=0; i--)
    {                                                       ///start from the back!
      multiindex[i] = copiedindex/sizes[i];                 ///implicit size_t cast? Yes: returns only how ofter size fites into cindex
      copiedindex = copiedindex%sizes[i];                   ///basically returns the (missing) rest of the checking above 
    }
    
    //std::cout << "index=" << index;
    //for (int i=0; i<d; ++i) std::cout << " " << multiindex[i];
    //std::cout << std::endl;

    // the current cell
    std::vector<double> center_position(d);                 ///scaled up multigrid/cell-position
    for (int i=0; i<d; ++i) 
      center_position[i] = multiindex[i]*mesh_size;
    double center_coefficient = diffusion_coefficient(center_position);
    double center_matrix_entry = 0.0;

    // loop over all neighbors
    for (int i=0; i<d; i++)
    {
      // down neighbor
      if (multiindex[i]>0)
      {
        // we have a neighbor cell
        std::vector<double> neighbor_position(center_position);
        neighbor_position[i] -= mesh_size;
        double neighbor_coefficient = diffusion_coefficient(neighbor_position);
        double harmonic_average = 2.0/( (1.0/neighbor_coefficient) + (1.0/center_coefficient) );
        //pA->entry(index,index-sizes[i]) = -harmonic_average;
        mtx_data.nonzeros.emplace_back(index,index-sizes[i], -harmonic_average);                ///@changed
        center_matrix_entry += harmonic_average;
      }
      else
      {
        // current cell is on the boundary in this direction
        std::vector<double> neighbor_position(center_position);
        neighbor_position[i] = 0.0;
        if (dirichlet_boundary(neighbor_position))
          center_matrix_entry += center_coefficient*2.0;
      }

      // up neighbor
      if (multiindex[i]<n-1)
      {
        // we have a neighbor cell
        std::vector<double> neighbor_position(center_position);
        neighbor_position[i] += mesh_size;
        double neighbor_coefficient = diffusion_coefficient(neighbor_position);
        double harmonic_average = 2.0/( (1.0/neighbor_coefficient) + (1.0/center_coefficient) );
        //pA->entry(index,index+sizes[i]) = -harmonic_average;                                    
        mtx_data.nonzeros.emplace_back(index,index+sizes[i], -harmonic_average);                ///@changed
        center_matrix_entry += harmonic_average;
      }
      else
      {
        // current cell is on the boundary in this direction
        std::vector<double> neighbor_position(center_position);
        neighbor_position[i] = 1.0;
        if (dirichlet_boundary(neighbor_position))
          center_matrix_entry += center_coefficient*2.0;
      }
    }

    // finally the diagonal entry
    //pA->entry(index,index) = center_matrix_entry;     //# easyer if more positive, time would be added here
    mtx_data.nonzeros.emplace_back(index,index, center_matrix_entry);                           ///@changed
  }
  //create matrix from data
  //auto stats = pA->compress();
  size_t nnz = (2*d+1)*N;
  //auto pA = gko::share(mtx::create(exec,gko::dim<2>(N), nnz/*, mtx::strategy_type::strategy_type("classical")*/)); //? @optimization -> better overload? more exact nnz_size?
  auto pA = mtx::create(exec);                          ///@optimize (line below included)
  pA->read(mtx_data);

  return pA;
}







int main() {

    std::cout<<"-------------------------------STARTING:gko-evaluate-solvers---------------------------------------"<<std::endl;


    //using mtx_entry = double;
    using mtx = gko::matrix::Csr<>;
    using vec = gko::matrix::Dense<double>;

    const size_t n = 20;
    const size_t d = 3;
    size_t N = 1;
    for(int i=0; i<d;i++) N*=n;

    //defining executer
    const auto exec = gko::ReferenceExecutor::create();                                 //? parameters needed? @optimize

    //create sparse matrix
    auto diffusion_coefficient = [](const std::vector<double>& x) { return 1.0; };
    auto dirichlet_boundary = [](const std::vector<double>& x) { return true; };
    //synchronize before timing
    exec->synchronize();
    auto time_start = std::chrono::steady_clock::now();
    auto pA = gko::share(diffusion_matrix<mtx>(n,d,diffusion_coefficient,dirichlet_boundary,exec));
    //synchronize before timing
    exec->synchronize();
    auto time_stop = std::chrono::steady_clock::now();
    auto time_to_generate = std::chrono::duration_cast<std::chrono::nanoseconds>(time_stop - time_start);
    std::cout << "Time it took to generate the matrix:  " << time_to_generate.count()/1000000 << "."<< time_to_generate.count()%1000000 << "ms" << std::endl; 


    //initialize vectors
    auto x = vec::create(exec,gko::dim<2,size_t>(N,1));
    auto result = vec::create(exec,gko::dim<2,size_t>(N,1));

    gko::matrix_data<> vec_data{gko::dim<2>(N,1)}; 
    /*for(size_t i=0; i<N;i++){
        vec_data.nonzeros.emplace_back(i,1UL,1.0);
    };*/
    x->read(vec_data);
    
   
    //synchronize before timing
    exec->synchronize();
    time_start = std::chrono::steady_clock::now();
    pA->apply(x, result);
    //synchronize before timing
    exec->synchronize();
    time_stop = std::chrono::steady_clock::now();
    time_to_generate = std::chrono::duration_cast<std::chrono::nanoseconds>(time_stop - time_start);
    std::cout << "Time it took to apply A on x:  " << time_to_generate.count()/1000000 <<"."<< time_to_generate.count()%1000000<< "ms" << std::endl;
    std::cout << "result:  "<< std::endl;
    //gko::write(std::cout, result);

    std::cout<<"-------------------------------FINISHED:gko-evaluate-solvers---------------------------------------"<<std::endl;
    return 0;
}

/*// Ginkgo Hello World
    using Mtx = gko::matrix::Csr<double, int>;
    using Vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{0.0, 1.0}, {1.0, 0.0}}, exec);
    auto vec = gko::initialize<Vec>({2.0, 3.0}, exec);
    auto result = vec->clone();
    mtx->apply(vec, result);
    gko::write(std::cout, result);
*/
/*
 Ginkgo Matrix assembly
    using Mtx = gko::matrix::Csr<double, int>;
    using Vec = gko::matrix::Dense<double>;
    using dim = gko::dim<2>;
    using matrix_data = gko::matrix_data<double, int>;
    auto exec = gko::ReferenceExecutor::create();
    matrix_data data{dim{10, 10}};
    auto mtx = Mtx::create(exec);
    auto x = Vec::create(exec, dim{10, 1});
    auto y = Vec::create(exec, dim{10, 1});
    for (int row = 0; row < 10; row++) {
        data.nonzeros.emplace_back(row, row, 2.0);
        data.nonzeros.emplace_back(row, (row + 1) % 10, -1.0);
        data.nonzeros.emplace_back(row, (row + 9) % 10, -1.0);
        x->at(row, 0) = 1.0; // only works on CPU executors
    }
    mtx->read(data);
    mtx->apply(x, y);
    gko::write(std::cout, y);
*/



 