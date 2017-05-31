#TBB library
include_directories(${TBB_ROOT_DIR}/include)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)	
			
add_executable(FaceRecognition_Train FaceRecognition_Train.cpp 2DfaceRecognition.cpp lbp.cpp histogram.cpp)
target_link_libraries(FaceRecognition_Train LandmarkDetector)
target_link_libraries(FaceRecognition_Train FaceAnalyser)
target_link_libraries(FaceRecognition_Train dlib)

target_link_libraries(FaceRecognition_Train ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})

install (TARGETS FaceRecognition_Train DESTINATION bin)
