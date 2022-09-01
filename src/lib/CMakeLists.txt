include_directories(fmt/include)
include_directories(abseil-cpp)
include_directories(labm8)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(fmt)
add_subdirectory(abseil-cpp)
add_subdirectory(labm8)
add_subdirectory(programl)

find_package(Protobuf REQUIRED)

set (IncreLuxSourceCodes
	UBIAnalysis.h
	UBIAnalysis.cpp
	#Points-to CG
	CallGraph.h
	CallGraph.cc
	Common.h
	Config.h
	PtsSet.h
	Helper.h
	Helper.cpp
	StructAnalyzer.h
	StructAnalyzer.cpp
	Annotation.h
	Annotation.cpp
	NodeFactory.h
	NodeFactory.cpp
	QualifierAnalysis.h
	QualifierAnalysis.cpp
	QualiInference.cpp
	QualifierCheck.cpp
	IOQualifierAnalysis.cpp
	IOQualifierCheck.cpp
	OOBQualifierAnalysis.cpp
	OOBQualifierCheck.cpp
	ConstraintCollect.cpp
	AAMap.cpp
	FuncProcess.cpp
	FunctionSummary.h
	#json.hpp
	json11.hpp
	json11.cpp
)

# Build libraries.
add_library (IncreLuxObj OBJECT ${IncreLuxSourceCodes})
add_library (IncreLux SHARED $<TARGET_OBJECTS:IncreLuxObj>)
add_library (IncreLuxStatic STATIC $<TARGET_OBJECTS:IncreLuxObj>)
llvm_map_components_to_libnames(llvm_libs bitwriter core ipo irreader instcombine instrumentation target linker analysis scalaropts support)
# Build executable ubitect.
set (EXECUTABLE_OUTPUT_PATH ${INCRELUX_BINARY_DIR})
link_directories (${INCRELUX_BINARY_DIR}/lib)
#project (ubitect C CXX)
add_executable(increlux ${IncreLuxSourceCodes})
add_dependencies(increlux GRAPHBUILDER LLVMINTERNAL PROTO)
# include_directories("{CMAKE_CURRENT_SOURCE_DIR}/programl/labm8/cpp/abseil-cpp")
target_link_libraries(increlux 
  ${llvm_libs}
  LLVMINTERNAL
  ${Protobuf_LIBRARIES}
  absl::flat_hash_map
  GRAPHBUILDER
  Labm8Objects
  LLVMHELP
  PROTO
  LLVMAsmParser 
  LLVMSupport 
  LLVMCore 
  LLVMAnalysis
  LLVMIRReader
  LLVMTransformUtils
  IncreLuxStatic
  )

