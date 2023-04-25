cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(leaktest)

find_package(Torch REQUIRED)

add_executable(leaktest leaktest.cpp)
target_link_libraries(leaktest "${TORCH_LIBRARIES}" nvidia-ml)
set_property(TARGET leaktest PROPERTY CXX_STANDARD 14)
