#include <stdio.h>
#include <stdlib.h>
#include <iostream>
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