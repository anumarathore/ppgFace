#TBB library
include_directories(${TBB_ROOT_DIR}/include)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)	
			
add_executable(FaceDetectionImg FaceDetectionImg.cpp)
target_link_libraries(FaceDetectionImg LandmarkDetector)
target_link_libraries(FaceDetectionImg FaceAnalyser)
target_link_libraries(FaceDetectionImg dlib)

target_link_libraries(FaceDetectionImg ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})

install (TARGETS FaceDetectionImg DESTINATION bin)
