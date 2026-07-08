cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(tree
	STABLE_ABI LTO NB_STATIC
	src/zifor/tree.cpp)
target_compile_definitions(tree PRIVATE VERSION_INFO="${PROJECT_VERSION}")
set_property(TARGET tree PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
set_property(TARGET tree PROPERTY CXX_VISIBILITY_PRESET hidden)

install(TARGETS tree DESTINATION zifor)
