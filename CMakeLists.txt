cmake_minimum_required(VERSION 3.16)

#NOTE: apple clang 15 is tested according to GINKGO Github, so this should not be necessary
#NOTE: GINKGO is compiled using apple clang, using gcc would require reinstalling it
#set(CMAKE_C_COMPILER "/usr/local/bin/gcc-14")
#set(CMAKE_CXX_COMPILER "/usr/local/bin/gcc-14")

project(gko_protoype)

find_package(Ginkgo REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#include(FetchContent)
#FetchContent_Declare(
#	ginkgo
#	URL https://github.com/ginkgo-project/ginkgo/releases/tag/v1.7.0
#)
#set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#FetchContent_MakeAvailable(googletest)


enable_testing()

add_executable(
	gko-evaluate-solvers
	gko-evaluate-solvers.cc
)

target_link_libraries(
	gko-evaluate-solvers
	Ginkgo::ginkgo
)

#include(Ginkgo)
#gtest_discover_tests(main)
