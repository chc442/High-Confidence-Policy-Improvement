cmake_minimum_required(VERSION 3.15)
project(cs687_final_project)

set(CMAKE_CXX_STANDARD 14)

add_executable(cs687_final_project source/main.cpp source/dataSplit.cpp header/dataSplit.h header/stdafx.h source/safetyTest.cpp header/safetyTest.h source/utils.cpp header/utils.h source/candidateSelection.cpp header/candidateSelection.h)