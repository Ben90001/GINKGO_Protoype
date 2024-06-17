cmake_minimum_required(VERSION 3.16)
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
	main
	main.cc
)

target_link_libraries(
	main
	Ginkgo::ginkgo
)

#include(Ginkgo)
#gtest_discover_tests(main)
