#TBB library
include_directories(${TBB_ROOT_DIR}/include)

# Local libraries
include_directories(${LandmarkDetector_SOURCE_DIR}/include)
	
include_directories(../../lib/local/LandmarkDetector/include)
include_directories(../../lib/local/FaceAnalyser/include)	
			
add_executable(FacialExpression FacialExpression.cpp PredictExpression.cpp lbp.cpp histogram.cpp)
target_link_libraries(FacialExpression LandmarkDetector)
target_link_libraries(FacialExpression FaceAnalyser)
target_link_libraries(FacialExpression dlib)

target_link_libraries(FacialExpression ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})

install (TARGETS FacialExpression DESTINATION bin)
