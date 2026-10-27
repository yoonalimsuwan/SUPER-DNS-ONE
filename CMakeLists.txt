cmake_minimum_required(VERSION 3.20)
project(sesi_apple_inference LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")

# Objective-C++ ARC
set_source_files_properties(src/sesi_inference.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc")

add_library(sesi_inference STATIC src/sesi_inference.mm)
target_include_directories(sesi_inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sesi_inference PUBLIC
    "-framework CoreML"
    "-framework Foundation"
)

add_executable(sesi_demo examples/main.cpp)
target_link_libraries(sesi_demo PRIVATE sesi_inference)
