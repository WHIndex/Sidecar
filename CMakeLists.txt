cmake_minimum_required(VERSION 3.14)
project(Sidecar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)

set(SIDECAR_SOURCE
    src/benchmark/microbench.cpp
)

function(add_sidecar_benchmark target controller_enabled)
    add_executable(${target} ${SIDECAR_SOURCE})
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/competitor
    )
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE
        ENABLE_CONTROLLER=${controller_enabled}
        USE_TBB
    )
    target_compile_options(${target} PRIVATE
        -Wall
        -Wextra
        -mpopcnt
        -faligned-new
        -march=native
    )
    target_link_libraries(${target} PRIVATE
        Threads::Threads
        TBB::tbb
        OpenMP::OpenMP_CXX
    )
endfunction()

add_sidecar_benchmark(sidecar_benchmark 1)
add_sidecar_benchmark(sidecar_baseline 0)

install(TARGETS sidecar_benchmark sidecar_baseline DESTINATION bin)
