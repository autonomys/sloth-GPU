# Parallel OpenCL code for Sloth Encode Implementation 

## How to Build

### Linux/MacOS
0. Download this repo
1. Open your favorite terminal
2. `cd` into this directory (this repo/OpenCL)
3. Run this command for compiling the project: `g++ main.cpp -o PleaseWork -I./OpenCL/include -I. -L./OpenCL/lib -lOpenCL`
4. Run this command for running the executable: `./PleaseWork`

Explanation of the above command:
- `g++` is for the compiler, you can replace this with `clang++` too if you so please
- `main.cpp`, this is the file we are going to compile, if you want to run the throughput version, replace this with `throughput_main.cpp`
- `-o PleaseWork` specifies the output name of the executable. `PleaseWork` is required for pleasing the OpenCL gods.
- `-I./OpenCL/include` includes the necessary header files for OpenCL
- `-I.` since we are going to include some code in this directory, this argument is added, it might not be necessary, but it is a safe bet
- `-L./OpenCL/lib` specifies the directory of OpenCL library files
- `-lOpenCL` probably telling the compiler to use OpenCL.dll (for windows) or OpenCL.so (for Linux) from the path.

### Windows with Visual Studio (currently, an NVidia GPU and CUDA Toolkit is required)
0. Download the repo
1. Create an empty project in Visual Studio (tested with version 2017 and 2019)
2. Change build option to x64 from x86
3. Right-click on project name, and go to `Properties`
4. Under `C/C++`, click on `General`.
5. `Additional Include Directories`, in here, you should paste `path this repo\OpenCL\include`
6. Under `Linker`, click on `General`.
7. `Additional Library Directories`, in here, you should paste `path this repo\OpenCL\lib`
8. Under `Linker`, click on `Input`.
9. `Additional Dependencies`, in here, add the following string to the beginning of that mambo-jambo: `OpenCL.lib;`
10. include only the `main.cpp` (or `throughput_main.cpp`) and `hello.cl` (or `throughput_kernel.cl`) to your project. DO NOT INCLUDE `sloth256_189.c` file in your Visual Studio Project. This `sloth256_189.c` should only be present in the same directory with the other files in the file-system. Adding this to the project itself will cause errors.
11. Run the program (whether debug or release does not matter, but do not forget to apply steps 3-9 for every setting (debug or release))





