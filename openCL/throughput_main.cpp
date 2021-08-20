#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <fstream>
using namespace std;

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#pragma warning(disable:4996)

#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     abort();                                                                   \
   } while (0)

#define CL_CHECK_ERR(_expr)                                                     \
   ({                                                                           \
     cl_int _err = CL_INVALID_VALUE;                                            \
     typeof(_expr) _ret = _expr;                                                \
     if (_err != CL_SUCCESS) {                                                  \
       fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       abort();                                                                 \
     }                                                                          \
     _ret;                                                                      \
   })

/* convert the kernel file into a string */
int convertToString(const char* filename, std::string& s)
{
    size_t size;
    char* str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if (f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        if (!str)
        {
            f.close();
            return 0;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    cout << "Error: failed to open file\n:" << filename << endl;
    return 1;
}

#define n 1024 * 256

int main()
{
    // variable declarations
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem piece_mem = NULL, iv_mem = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int err;

    // Platform and Device Queries

    /* Get Platform and Device Info */
    if (clGetPlatformIDs(0, NULL, &ret_num_platforms) != CL_SUCCESS)
    {
        printf("Unable to get platform_id\n");
        return 1;
    }
    cl_platform_id* platform_ids = new cl_platform_id[ret_num_platforms];
    if (clGetPlatformIDs(ret_num_platforms, platform_ids, NULL) != CL_SUCCESS)
    {
        printf("Unable to get platform_id\n");
        return 1;
    }
    bool found = false;
    for (int i = 0; i < ret_num_platforms; i++)
        if (clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices) == CL_SUCCESS) {
            //printf("GPU fund\n");
            found = true;
            break;
        }
    if (!found) {
        printf("Unable to get device_id\n");
        return 1;
    }

    unsigned int i, j;                //iterator variables for loops

    cl_platform_id platforms[32];            //an array to hold the IDs of all the platforms, hopefully there won't be more than 32
    cl_uint num_platforms;                //this number will hold the number of platforms on this machine
    char vendor[1024];                //this string will hold a platforms vendor

    cl_device_id devices[32];            //this variable holds the number of devices for each platform, hopefully it won't be more than 32 per platform
    cl_uint num_devices;                //this number will hold the number of devices on this machine
    char deviceName[1024];                //this string will hold the devices name
    cl_uint numberOfCores;                //this variable holds the number of cores of on a device
    cl_long amountOfMemory;                //this variable holds the amount of memory on a device
    cl_uint clockFreq;                //this variable holds the clock frequency of a device
    cl_ulong maxAllocatableMem;            //this variable holds the maximum allocatable memory
    cl_ulong localMem;                //this variable holds local memory for a device
    cl_bool    available;                //this variable holds if the device is available
    char driver_ver[1024];

    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numberOfCores), &numberOfCores, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(amountOfMemory), &amountOfMemory, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFreq), &clockFreq, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAllocatableMem), &maxAllocatableMem, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_AVAILABLE, sizeof(available), &available, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(driver_ver), &driver_ver, NULL);

    //print out device information
    printf("\t\tName:\t\t\t\t%s\n", deviceName);
    printf("\t\tVendor:\t\t\t\t%s\n", vendor);
    printf("\t\tAvailable:\t\t\t%s\n", available ? "Yes" : "No");
    printf("\t\tCompute Units:\t\t\t%u\n", numberOfCores);
    printf("\t\tClock Frequency:\t\t%u mHz\n", clockFreq);
    printf("\t\tGlobal Memory:\t\t\t%0.00f mb\n", (double)amountOfMemory / 1048576);
    printf("\t\tMax Allocateable Memory:\t%0.00f mb\n", (double)maxAllocatableMem / 1048576);
    printf("\t\tLocal Memory:\t\t\t%u kb\n\n", (unsigned int)localMem);
    printf("\t\tDriver version:\t\t\t%s \n\n", driver_ver);


    // actual OpenCL application starts
    unsigned char* chunk = (unsigned char*)malloc(sizeof(unsigned char) * 4096 * n);
    //unsigned char chunk[4096];
    unsigned char iv[32];
    size_t layers = 1;

    memset(chunk, 5, 4096 * n);
    memset(iv, 3, 32);

    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

    /* Create Memory Buffer */
    piece_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4096 * sizeof(unsigned char) * n, chunk, &err);
    iv_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sizeof(unsigned char) * 32), iv, &err);

    /*
    FILE* fp;
    char fileName[] = "./hello.cl";
    char* source_str;
    size_t source_size;

    / Load the source code containing the kernel
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    */
    const char* filename = "throughput_kernel.cl";
    string sourceStr;
    cl_int status = convertToString(filename, sourceStr);
    printf("%d", status);
    const char* source = sourceStr.c_str();
    size_t sourceSize[] = { strlen(source) };
    program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

    /* Build Kernel Program */
    cl_int errNum = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return -1;
    }

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "throughput_kernel", &err);
    printf("\nKernel status: %i\n", err);

    /* Set OpenCL Kernel Parameters */
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&piece_mem));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&iv_mem));

    cl_event event;
    size_t global = n, local = 256;

    /* Execute OpenCL Kernel */
    CL_CHECK(clEnqueueNDRangeKernel(command_queue, kernel,
        1, NULL, &global, &local, 0, NULL, &event));

    clWaitForEvents(1, &event); // Wait for the event

    unsigned long start = 0;
    unsigned long end = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
        sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong), &end, NULL);

    // Compute the duration in nanoseconds
    unsigned long duration = end - start;

    // Don't forget to release the event
    clReleaseEvent(event);

    printf("Kernel time in milliseconds: %lu\n", duration / (1000 * 1000));

    /* Copy results from the memory buffer */
    CL_CHECK(clEnqueueReadBuffer(command_queue, piece_mem, CL_TRUE, 0,
        4096 * sizeof(char) * n, chunk, 0, NULL, NULL));

    /* Display Result */
    /*for (int i = 0; i < 128 * 32 * n; i++)
    {
        unsigned number = (unsigned)chunk[i];

        if (number == 0)
        {
            cout << "00";
        }
        else if (number < 16)
        {
            cout << "0";
            cout << hex << number;
        }
        else
        {
            cout << hex << number;
        }

        if (i % 32 == 31)
            cout << endl;
    }*/

    /* Finalization */
    CL_CHECK(clFlush(command_queue));
    CL_CHECK(clFinish(command_queue));
    CL_CHECK(clReleaseKernel(kernel));
    CL_CHECK(clReleaseProgram(program));
    CL_CHECK(clReleaseMemObject(piece_mem));
    CL_CHECK(clReleaseCommandQueue(command_queue));
    CL_CHECK(clReleaseContext(context));
    //CL_CHECK(clReleaseMemObject(iv_mem));
    //CL_CHECK(clReleaseMemObject(piece_mem));

    return 0;
}

