#TBB library
include_directories(${TBB_ROOT_DIR}/include)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)	
			
add_executable(FaceRecognition_Test FaceRecognition_Test.cpp 2DfaceRecognition.cpp lbp.cpp histogram.cpp)
target_link_libraries(FaceRecognition_Test LandmarkDetector)
target_link_libraries(FaceRecognition_Test FaceAnalyser)
target_link_libraries(FaceRecognition_Test dlib)

target_link_libraries(FaceRecognition_Test ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})

install (TARGETS FaceRecognition_Test DESTINATION bin)
