#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
using namespace std;

#include <CL/cl.h>

#pragma warning(disable:4996)

#define MAX_SOURCE_SIZE (0x100000)

int main()
{
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem piece_mem = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    unsigned char chunk[4096];
    unsigned char iv[32];
    size_t layers = 1;

    memset(chunk, 5, 4096);
    memset(iv, 3, 32);
    
    FILE* fp;
    char fileName[] = "./hello.cl";
    char* source_str;
    size_t source_size;

    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

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
            //printf("cipiyu fund\n");
            found = true;
            break;
        }
    if (!found) {
        printf("Unable to get device_id\n");
        return 1;
    }

    unsigned int i, j;                //iterator variables for loops

    cl_platform_id platforms[32];            //an array to hold the IDs of all the platforms, hopefuly there won't be more than 32
    cl_uint num_platforms;                //this number will hold the number of platforms on this machine
    char vendor[1024];                //this strirng will hold a platforms vendor

    cl_device_id devices[32];            //this variable holds the number of devices for each platform, hopefully it won't be more than 32 per platform
    cl_uint num_devices;                //this number will hold the number of devices on this machine
    char deviceName[1024];                //this string will hold the devices name
    cl_uint numberOfCores;                //this variable holds the number of cores of on a device
    cl_long amountOfMemory;                //this variable holds the amount of memory on a device
    cl_uint clockFreq;                //this variable holds the clock frequency of a device
    cl_ulong maxAlocatableMem;            //this variable holds the maximum allocatable memory
    cl_ulong localMem;                //this variable holds local memory for a device
    cl_bool    available;                //this variable holds if the device is available
    char driver_ver[1024];

    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numberOfCores), &numberOfCores, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(amountOfMemory), &amountOfMemory, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFreq), &clockFreq, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlocatableMem), &maxAlocatableMem, NULL);
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
    printf("\t\tMax Allocateable Memory:\t%0.00f mb\n", (double)maxAlocatableMem / 1048576);
    printf("\t\tLocal Memory:\t\t\t%u kb\n\n", (unsigned int)localMem);
    printf("\t\tDriver version:\t\t\t%s \n\n", driver_ver);


    /* Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Create Memory Buffer */
    piece_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4096 * sizeof(unsigned char), &chunk, &ret);
    auto iv_mem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (sizeof(unsigned char) * 32), &iv, &ret);

    /* Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char**)&source_str,
        (const size_t*)&source_size, &ret);

    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    /* Create OpenCL Kernel */
    kernel = clCreateKernel(program, "hello", &ret);

    /* Set OpenCL Kernel Parameters */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&piece_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&iv_mem);

    /* Execute OpenCL Kernel */
    ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);

    /* Copy results from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, piece_mem, CL_TRUE, 0,
        4096 * sizeof(char), chunk, 0, NULL, NULL);

    /* Display Result */
    for (int i = 0; i < 128 * 32; i++)
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
    }

    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(piece_mem);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(source_str);

    return 0;
}
