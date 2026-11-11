cmake_minimum_required(VERSION 3.28)
project(muna_crash LANGUAGES C CXX)

include(FetchContent)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CPPTRACE_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CPPTRACE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CPPTRACE_UNWIND_WITH_LIBUNWIND ON CACHE BOOL "" FORCE)
set(CPPTRACE_GET_SYMBOLS_WITH_LIBDWARF ON CACHE BOOL "" FORCE)
set(ZLIB_USE_STATIC_LIBS ON)
FetchContent_Declare(cpptrace
    GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
    GIT_TAG 3db8da80111171c219ab5839905771386bee06b3
)
FetchContent_MakeAvailable(cpptrace)

# Distro libunwind/liblzma static archives can contain non-PIC LTO objects.
# Build a PIC local unwinder, without compressed-debug readers (the helper's
# libdwarf handles symbols), so Rust's PIE executable stays self-contained.
include(ExternalProject)
set(UNWIND_PREFIX "${CMAKE_BINARY_DIR}/unwind")
ExternalProject_Add(muna_unwind
    GIT_REPOSITORY https://github.com/libunwind/libunwind.git
    GIT_TAG 9cc4d98b22ae57bc1d8c253988feb85d4298a634
    UPDATE_DISCONNECTED TRUE
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E chdir <SOURCE_DIR> autoreconf -i
        COMMAND <SOURCE_DIR>/configure --prefix=${UNWIND_PREFIX}
            --with-pic --disable-shared --enable-static --disable-tests
            --disable-coredump --disable-minidebuginfo --disable-zlibdebuginfo
    BUILD_COMMAND make -j2
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS "${UNWIND_PREFIX}/lib/libunwind.a"
)

add_library(muna_crash STATIC crash_handler.cpp)
add_dependencies(muna_crash muna_unwind)
target_compile_features(muna_crash PRIVATE cxx_std_20)
target_compile_options(muna_crash PRIVATE -g -fno-omit-frame-pointer -fasynchronous-unwind-tables)
target_link_libraries(muna_crash PRIVATE cpptrace::cpptrace)

# Keep deployment self-contained: no libunwind, libdwarf, or symbolizer package
# needs to be installed on the inference node.
find_package(ZLIB REQUIRED)
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/native-libraries.txt" CONTENT
"$<TARGET_FILE:muna_crash>
$<TARGET_FILE:cpptrace::cpptrace>
$<TARGET_FILE:dwarf>
$<TARGET_FILE:libzstd_static>
${UNWIND_PREFIX}/lib/libunwind.a
${ZLIB_LIBRARIES}
")
