# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

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
CMAKE_COMMAND = /home/lyrapuff/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/212.5457.51/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/lyrapuff/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/212.5457.51/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point

# Include any dependencies generated for this target.
include src/CMakeFiles/vulkan_guide.dir/depend.make
# Include the progress variables for this target.
include src/CMakeFiles/vulkan_guide.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/vulkan_guide.dir/flags.make

src/CMakeFiles/vulkan_guide.dir/main.cpp.o: src/CMakeFiles/vulkan_guide.dir/flags.make
src/CMakeFiles/vulkan_guide.dir/main.cpp.o: src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/vulkan_guide.dir/main.cpp.o"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vulkan_guide.dir/main.cpp.o -c /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/main.cpp

src/CMakeFiles/vulkan_guide.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vulkan_guide.dir/main.cpp.i"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/main.cpp > CMakeFiles/vulkan_guide.dir/main.cpp.i

src/CMakeFiles/vulkan_guide.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vulkan_guide.dir/main.cpp.s"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/main.cpp -o CMakeFiles/vulkan_guide.dir/main.cpp.s

src/CMakeFiles/vulkan_guide.dir/vk_engine.cpp.o: src/CMakeFiles/vulkan_guide.dir/flags.make
src/CMakeFiles/vulkan_guide.dir/vk_engine.cpp.o: src/vk_engine.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/vulkan_guide.dir/vk_engine.cpp.o"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vulkan_guide.dir/vk_engine.cpp.o -c /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_engine.cpp

src/CMakeFiles/vulkan_guide.dir/vk_engine.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vulkan_guide.dir/vk_engine.cpp.i"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_engine.cpp > CMakeFiles/vulkan_guide.dir/vk_engine.cpp.i

src/CMakeFiles/vulkan_guide.dir/vk_engine.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vulkan_guide.dir/vk_engine.cpp.s"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_engine.cpp -o CMakeFiles/vulkan_guide.dir/vk_engine.cpp.s

src/CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.o: src/CMakeFiles/vulkan_guide.dir/flags.make
src/CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.o: src/vk_initializers.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.o"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.o -c /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_initializers.cpp

src/CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.i"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_initializers.cpp > CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.i

src/CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.s"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_initializers.cpp -o CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.s

src/CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.o: src/CMakeFiles/vulkan_guide.dir/flags.make
src/CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.o: src/vk_pipeline.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.o"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.o -c /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_pipeline.cpp

src/CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.i"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_pipeline.cpp > CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.i

src/CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.s"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_pipeline.cpp -o CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.s

src/CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.o: src/CMakeFiles/vulkan_guide.dir/flags.make
src/CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.o: src/vk_mesh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.o"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.o -c /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_mesh.cpp

src/CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.i"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_mesh.cpp > CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.i

src/CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.s"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/vk_mesh.cpp -o CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.s

# Object files for target vulkan_guide
vulkan_guide_OBJECTS = \
"CMakeFiles/vulkan_guide.dir/main.cpp.o" \
"CMakeFiles/vulkan_guide.dir/vk_engine.cpp.o" \
"CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.o" \
"CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.o" \
"CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.o"

# External object files for target vulkan_guide
vulkan_guide_EXTERNAL_OBJECTS =

bin/vulkan_guide: src/CMakeFiles/vulkan_guide.dir/main.cpp.o
bin/vulkan_guide: src/CMakeFiles/vulkan_guide.dir/vk_engine.cpp.o
bin/vulkan_guide: src/CMakeFiles/vulkan_guide.dir/vk_initializers.cpp.o
bin/vulkan_guide: src/CMakeFiles/vulkan_guide.dir/vk_pipeline.cpp.o
bin/vulkan_guide: src/CMakeFiles/vulkan_guide.dir/vk_mesh.cpp.o
bin/vulkan_guide: src/CMakeFiles/vulkan_guide.dir/build.make
bin/vulkan_guide: third_party/libvkbootstrap.a
bin/vulkan_guide: third_party/libtinyobjloader.a
bin/vulkan_guide: third_party/libimgui.a
bin/vulkan_guide: /usr/lib/x86_64-linux-gnu/libvulkan.so
bin/vulkan_guide: src/CMakeFiles/vulkan_guide.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable ../bin/vulkan_guide"
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vulkan_guide.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/vulkan_guide.dir/build: bin/vulkan_guide
.PHONY : src/CMakeFiles/vulkan_guide.dir/build

src/CMakeFiles/vulkan_guide.dir/clean:
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src && $(CMAKE_COMMAND) -P CMakeFiles/vulkan_guide.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/vulkan_guide.dir/clean

src/CMakeFiles/vulkan_guide.dir/depend:
	cd /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src /media/lyrapuff/Programming/cpp/vulkan-guide-starting-point/src/CMakeFiles/vulkan_guide.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/vulkan_guide.dir/depend
