# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/HwHiAiUser/gemm/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/HwHiAiUser/gemm/build/intermediates/host

# Include any dependencies generated for this target.
include CMakeFiles/execute_gemm_op.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/execute_gemm_op.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/execute_gemm_op.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/execute_gemm_op.dir/flags.make

CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.o: CMakeFiles/execute_gemm_op.dir/flags.make
CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.o: /home/HwHiAiUser/gemm/src/gemm_pure.cpp
CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.o: CMakeFiles/execute_gemm_op.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/HwHiAiUser/gemm/build/intermediates/host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.o -MF CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.o.d -o CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.o -c /home/HwHiAiUser/gemm/src/gemm_pure.cpp

CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/HwHiAiUser/gemm/src/gemm_pure.cpp > CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.i

CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/HwHiAiUser/gemm/src/gemm_pure.cpp -o CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.s

# Object files for target execute_gemm_op
execute_gemm_op_OBJECTS = \
"CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.o"

# External object files for target execute_gemm_op
execute_gemm_op_EXTERNAL_OBJECTS =

/home/HwHiAiUser/gemm/run/out/execute_gemm_op: CMakeFiles/execute_gemm_op.dir/gemm_pure.cpp.o
/home/HwHiAiUser/gemm/run/out/execute_gemm_op: CMakeFiles/execute_gemm_op.dir/build.make
/home/HwHiAiUser/gemm/run/out/execute_gemm_op: CMakeFiles/execute_gemm_op.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/HwHiAiUser/gemm/build/intermediates/host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/HwHiAiUser/gemm/run/out/execute_gemm_op"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/execute_gemm_op.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/execute_gemm_op.dir/build: /home/HwHiAiUser/gemm/run/out/execute_gemm_op
.PHONY : CMakeFiles/execute_gemm_op.dir/build

CMakeFiles/execute_gemm_op.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/execute_gemm_op.dir/cmake_clean.cmake
.PHONY : CMakeFiles/execute_gemm_op.dir/clean

CMakeFiles/execute_gemm_op.dir/depend:
	cd /home/HwHiAiUser/gemm/build/intermediates/host && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/HwHiAiUser/gemm/src /home/HwHiAiUser/gemm/src /home/HwHiAiUser/gemm/build/intermediates/host /home/HwHiAiUser/gemm/build/intermediates/host /home/HwHiAiUser/gemm/build/intermediates/host/CMakeFiles/execute_gemm_op.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/execute_gemm_op.dir/depend

