# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.22

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01\build"

# Include any dependencies generated for this target.
include CMakeFiles/PP01.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/PP01.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/PP01.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PP01.dir/flags.make

CMakeFiles/PP01.dir/PP01.cpp.obj: CMakeFiles/PP01.dir/flags.make
CMakeFiles/PP01.dir/PP01.cpp.obj: ../PP01.cpp
CMakeFiles/PP01.dir/PP01.cpp.obj: CMakeFiles/PP01.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PP01.dir/PP01.cpp.obj"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\X86_64~2.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/PP01.dir/PP01.cpp.obj -MF CMakeFiles\PP01.dir\PP01.cpp.obj.d -o CMakeFiles\PP01.dir\PP01.cpp.obj -c "C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01\PP01.cpp"

CMakeFiles/PP01.dir/PP01.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PP01.dir/PP01.cpp.i"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\X86_64~2.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01\PP01.cpp" > CMakeFiles\PP01.dir\PP01.cpp.i

CMakeFiles/PP01.dir/PP01.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PP01.dir/PP01.cpp.s"
	C:\PROGRA~1\MINGW-~1\X86_64~1.0-P\mingw64\bin\X86_64~2.EXE $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01\PP01.cpp" -o CMakeFiles\PP01.dir\PP01.cpp.s

# Object files for target PP01
PP01_OBJECTS = \
"CMakeFiles/PP01.dir/PP01.cpp.obj"

# External object files for target PP01
PP01_EXTERNAL_OBJECTS =

PP01.exe: CMakeFiles/PP01.dir/PP01.cpp.obj
PP01.exe: CMakeFiles/PP01.dir/build.make
PP01.exe: CMakeFiles/PP01.dir/linklibs.rsp
PP01.exe: CMakeFiles/PP01.dir/objects1.rsp
PP01.exe: CMakeFiles/PP01.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable PP01.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\PP01.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PP01.dir/build: PP01.exe
.PHONY : CMakeFiles/PP01.dir/build

CMakeFiles/PP01.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\PP01.dir\cmake_clean.cmake
.PHONY : CMakeFiles/PP01.dir/clean

CMakeFiles/PP01.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01" "C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01" "C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01\build" "C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01\build" "C:\Users\danie\OneDrive\Dokumenty\Drugi stopień\Semestr 2\PSPR\PP01\build\CMakeFiles\PP01.dir\DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/PP01.dir/depend

