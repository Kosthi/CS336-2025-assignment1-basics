cmake_minimum_required(VERSION 3.10)

project(bpe_train)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

add_library(bpe SHARED bpe.cpp)
add_executable(bpe_cpp bpe.cpp)

# OpenMP configuration
if (APPLE)
    # Attempt to find PyTorch's OpenMP library to avoid runtime conflicts
    # CMAKE_CURRENT_SOURCE_DIR is .../cpp
    set(TORCH_OMP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/lib/python3.11/site-packages/torch/lib/libomp.dylib")
    get_filename_component(TORCH_OMP_PATH_ABS "${TORCH_OMP_PATH}" ABSOLUTE)
    
    message(STATUS "Checking for Torch OpenMP at: ${TORCH_OMP_PATH_ABS}")
    
    if (EXISTS "${TORCH_OMP_PATH_ABS}")
        message(STATUS "Found Torch OpenMP: ${TORCH_OMP_PATH_ABS}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor" "-fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${TORCH_OMP_PATH_ABS}")
        
        # We still need headers. Try Homebrew or standard locations.
        if (EXISTS "/opt/homebrew/opt/libomp/include")
            include_directories("/opt/homebrew/opt/libomp/include")
        elseif (EXISTS "/usr/local/opt/libomp/include")
            include_directories("/usr/local/opt/libomp/include")
        endif()
        
        target_compile_options(bpe PUBLIC ${OpenMP_CXX_FLAGS})
        target_link_libraries(bpe PUBLIC "${TORCH_OMP_PATH_ABS}")
        
        # Force the library to use the absolute path to Torch's libomp.dylib
        # This prevents runtime conflicts when DYLD_LIBRARY_PATH includes other OpenMP versions
        add_custom_command(TARGET bpe POST_BUILD
            COMMAND install_name_tool -change "@rpath/libomp.dylib" "${TORCH_OMP_PATH_ABS}" $<TARGET_FILE:bpe>
            COMMENT "Fixing OpenMP library path for bpe"
        )
        
        target_compile_options(bpe_cpp PUBLIC ${OpenMP_CXX_FLAGS})
        target_link_libraries(bpe_cpp PUBLIC "${TORCH_OMP_PATH_ABS}")
        
        add_custom_command(TARGET bpe_cpp POST_BUILD
            COMMAND install_name_tool -change "@rpath/libomp.dylib" "${TORCH_OMP_PATH_ABS}" $<TARGET_FILE:bpe_cpp>
            COMMENT "Fixing OpenMP library path for bpe_cpp"
        )
    else()
        message(STATUS "Torch OpenMP not found, falling back to system/Homebrew OpenMP")
        if (EXISTS "/opt/homebrew/opt/libomp")
            set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
            list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
        elseif (EXISTS "/usr/local/opt/libomp")
            set(OpenMP_ROOT "/usr/local/opt/libomp")
            list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/libomp")
        endif ()
        
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        if (OpenMP_CXX_FOUND)
            target_link_libraries(bpe PUBLIC OpenMP::OpenMP_CXX)
            target_link_libraries(bpe_cpp PUBLIC OpenMP::OpenMP_CXX)
        endif ()
    endif()
else()
    # Linux / Ubuntu configuration
    message(STATUS "Configuring for Linux/Ubuntu...")
    
    find_package(OpenMP REQUIRED)
    
    if(OpenMP_CXX_FOUND)
        message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
        # Ensure -fopenmp is used if needed (usually handled by OpenMP::OpenMP_CXX)
        target_link_libraries(bpe PUBLIC OpenMP::OpenMP_CXX)
        target_link_libraries(bpe_cpp PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(FATAL_ERROR "OpenMP not found. Please install it (e.g., sudo apt-get install libomp-dev on Ubuntu).")
    endif()
endif()
