cmake_minimum_required(VERSION 3.29)

set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)

project(afdd_test_pc)

# ========== 1. 查找 ONNX Runtime ==========
if (NOT DEFINED ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT /opt/onnxruntime)   # 按需修改
endif ()

find_path(ORT_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS ${ONNXRUNTIME_ROOT}/include)

find_library(ORT_LIBRARY
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_ROOT}/lib)

if (NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(FATAL_ERROR "Cannot find onnxruntime in ${ONNXRUNTIME_ROOT}")
endif ()

# ========== 2. C/C++ 标准 ==========
set(CMAKE_C_STANDARD 11)     # ai_model.c
set(CMAKE_CXX_STANDARD 14)   # 其余源码

# ========== 3. 可执行文件 ==========
add_executable(afdd_test_pc
        main.cpp
        ai_model.c
        onnx_wrapper.hpp        # 如果 ai_model 改成 C++ 则改为 .cpp
)

# 把 ai_model.c 当成 C++
set_source_files_properties(ai_model.c PROPERTIES LANGUAGE CXX)

# 预定义宏
target_compile_definitions(afdd_test_pc PRIVATE TARGET_PC)

# ========== 4. 头文件 / 库 ==========
target_include_directories(afdd_test_pc PRIVATE ${ORT_INCLUDE_DIR})

target_link_libraries(afdd_test_pc PRIVATE
        ${ORT_LIBRARY}
        pthread                # Linux 下 ORT 依赖
        dl                     # 某些版本需要
)

# ========== 5. 大静态数组专用：large model + 关闭 PIE ==========
if (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    # 关闭 position-independent code / executable
    set_property(TARGET afdd_test_pc PROPERTY POSITION_INDEPENDENT_CODE OFF)

    # 编译选项
    target_compile_options(afdd_test_pc PRIVATE
            -mcmodel=large      # 允许 >2 GB 静态段
            -fno-pie            # 不生成 PIE
    )

    # 链接选项
    target_link_options(afdd_test_pc PRIVATE
            -mcmodel=large
            -no-pie
    )
endif ()


add_custom_command(
        TARGET afdd_test_pc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/ai_model.c
        /home/manu/mnt/ST2000DM005-2U91/workspace/collection_firmware/App/cdc_acm/src/ai_model.c
        COMMENT "Copying model file ai_model.c ..."
)

add_custom_command(
        TARGET afdd_test_pc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/ai_model.h
        /home/manu/mnt/ST2000DM005-2U91/workspace/collection_firmware/App/cdc_acm/inc/ai_model.h
        COMMENT "Copying model file ai_model.h ..."
)

# 增加栈大小到256MB（根据需要调整）
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(afdd_test_pc PRIVATE
            -Wl,-z,stack-size=268435456  # 256MB
    )
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_options(afdd_test_pc PRIVATE
            /STACK:268435456  # 256MB
    )
endif ()