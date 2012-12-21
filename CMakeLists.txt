CMAKE_MINIMUM_REQUIRED(VERSION 2.8.0)

project(mini_project3)
find_package(OpenCV REQUIRED)
add_executable(mini_project3 main.cpp)
target_link_libraries(mini_project3 ${OpenCV_LIBS})
