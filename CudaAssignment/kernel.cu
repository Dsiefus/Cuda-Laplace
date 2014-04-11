#include "cuda_runtime.h"
#include <helper_cuda.h>
#include <helper_timer.h>
#include <stdio.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

#define PI 3.1415926535897932384626433832795

#define INTERNAL_ITERATIONS 5

//binary operation definition (absolute difference)
template <typename T>
struct abs_diff : public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()(const T& a, const T& b)
    {
        return std::fabs(b - a);
    }
};

void print_file(int n, float* h_a) {
	int i, j;
	FILE *file = fopen("output.txt", "w+");
	for (i = 0; i < n + 2; i++) 
		for (j = 0; j < n + 2; j++) 
			fprintf(file, "%d %d %lf\n", i, j, h_a[i*(n+2)+ j]);		
	fclose(file);
}

//indexing of shared memory. Threads have 2 more rows and cols (for "halo" nodes)
__device__ inline int getSharedIndex(int thrIdx, int thrIdy)
{
	return ((thrIdy+1) * (blockDim.x+2) + thrIdx +1);
}

//indexing of global memory corresponding to each thread
__device__ inline int getGlobalIndex()
{	
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	return col+1 + (row+1)*(blockDim.x * gridDim.x +2);
}

__global__ void JacobiStep(const float *oldMatrix, float *newMatrix)
{
	extern __shared__ float aux[];
	register int thx = threadIdx.x, thy = threadIdx.y;	
	aux[ getSharedIndex(thx, thy)] = oldMatrix[getGlobalIndex()];	

	int leftIndex = getSharedIndex(thx-1,thy), rightIndex = getSharedIndex(thx+1,thy);
	int topIndex = getSharedIndex(thx,thy-1), botIndex = getSharedIndex(thx,thy+1);	
	
	//left
	if (thx == 0) 	   
		aux[leftIndex] = oldMatrix[getGlobalIndex()-1];	
		
	//top
	if  (thy == 0) 
		 aux[topIndex] = oldMatrix[getGlobalIndex()-(blockDim.x * gridDim.x +2)];			

	//right
	if (thx == blockDim.x-1)
		aux[rightIndex] = oldMatrix[getGlobalIndex()+1];		

	//bot
	if (thy == blockDim.y - 1)
		 aux[botIndex]=oldMatrix[getGlobalIndex()+(blockDim.x * gridDim.x +2)];	
	
	for (int i = 0; i < INTERNAL_ITERATIONS; i++)
	{
		__syncthreads();
		float temp = 0.25*(aux[rightIndex]+aux[topIndex]+ aux[leftIndex]+aux[botIndex]);
		__syncthreads();
		aux[getSharedIndex(thx,thy)]=temp;
	}
	__syncthreads();	
	newMatrix[getGlobalIndex()] = 0.25*(aux[rightIndex]+aux[topIndex]+ aux[leftIndex]+aux[botIndex]);
}

//Returns the maximum absolute difference element-wise between the elements of a float vector on device memory,
//and a thrust device_vector, both with size matrixSize
float GetMaxDiff( float* a, thrust::device_vector<float> b, int matrixSize)
{
	float maxDifference;
	try{
	thrust::device_ptr<float> dev_ptra =  thrust::device_pointer_cast(a);	
    float init = 0;    
    thrust::maximum<float> binary_op1;
    abs_diff<float> binary_op2;
	maxDifference =  thrust::inner_product(dev_ptra,dev_ptra +  matrixSize,b.begin(), init, binary_op1, binary_op2); 
	}	 catch(thrust::system_error e)
	{    printf(e.what());
	}
   return maxDifference;
}

thrust::host_vector<float> GetAnalyticalMatrix(char* matrixSide, int matrixSize)
{
	char analyticalFilename[50];
	analyticalFilename[0]=0;
	strcpy(analyticalFilename,matrixSide);
	strcat(analyticalFilename,"_2.dat");
	FILE *matrixFile = fopen(analyticalFilename, "rb");	
	if (matrixFile == NULL)
	{
		printf("Error: Analytical solution file not found\n");
		exit( -1);
	}
	thrust::host_vector<float> analyticalHost(matrixSize);	
	//We read directly from the file to a thrust host vector, which is then copied to a device vector
	int n=fread(&analyticalHost[0],sizeof(float),matrixSize,matrixFile);
	fclose(matrixFile);	 
	return analyticalHost;
}

void PrintResults(char* size,int matrixSide, int final_its,float total_time, float program_time, int matrixSize, float accuracy, float max_abs_diff)
{
	char outputFileName[50];
	outputFileName[0]=0;
	strcpy(outputFileName,size);
	strcat(outputFileName,"_multi_times.txt");
	FILE* outfile = fopen(outputFileName,"a+");
	if (outfile == NULL)
	{
		printf("Error with output file\n");
		exit(-1);
	}

	printf("Time for matrixSide= %d, %d (x%d) maxIterations: %f ms. Total time: %f. Memory bandwith is %f GB/s. ",matrixSide,final_its, INTERNAL_ITERATIONS,total_time, program_time,((1e-6)*matrixSize)*2*INTERNAL_ITERATIONS*final_its*sizeof(float)/(total_time)); 
	printf("Accuracy desired: %f (obtained %f)\n",accuracy,max_abs_diff);
	fprintf(outfile,"Iterations: %d (x%d). Time: %f ms. Accuracy desired: %f (obtained %f). Memory bandwith: %f GB/s\n",final_its,INTERNAL_ITERATIONS, total_time,accuracy,max_abs_diff,((1e-6)*matrixSize)*2*INTERNAL_ITERATIONS*final_its*sizeof(float)/(total_time)); 
	fclose(outfile);	
}

int main(int argc, char* argv[])
{
	LARGE_INTEGER t_ini, t_fin, freq;
		QueryPerformanceCounter(&t_ini);

	if (argc != 3)
	{
		printf("Usage: %s <matrix_side> <desired_accuracy>\n", argv[0]);
		return 0;
	}
	const int matrixSide = atoi(argv[1]);
	if (matrixSide%16 != 0)
	{
		printf("Error: matrix side must divide 16\n");
		return -1;
	}	
	const float accuracy = atof(argv[2]);
	if (accuracy > 0.5 || accuracy < 0.001)
	{
		printf("Error: accuracy must be smaller than 0.5 and bigger than 0.001\n");
		return -1;
	}
	
	const int maxIterations=80000;
	const int matrixSize = (matrixSide+2)*(matrixSide+2);
	char evolutionFileName[50];
	evolutionFileName[0]=0;
	strcpy(evolutionFileName,argv[1]);
	strcat(evolutionFileName,"multi_iteration.txt");
	FILE* evolutionFile = fopen (evolutionFileName, "a+");
	if (evolutionFile == NULL)
	{
		printf("Error opening evolution file\n");
		return -1;
	}

	{ //block to encapuslate thrust
	fprintf(evolutionFile,"Internal iterations: %d\n",INTERNAL_ITERATIONS);
	//We can directly copy from a thrust host vector to a device vector
	thrust::device_vector<float> analyticalDev = GetAnalyticalMatrix(argv[1],matrixSize);		 
	
    float *oldMatrix = 0, *newMatrix = 0;
	checkCudaErrors( cudaMalloc((void**)&oldMatrix, matrixSize*sizeof(float)));	
	checkCudaErrors( cudaMalloc((void**)&newMatrix, matrixSize*sizeof(float)));	   
  
	float* hostAuxVector = 0;
	checkCudaErrors(cudaHostAlloc((void**) &hostAuxVector,matrixSize * sizeof(float), cudaHostAllocDefault));	
	for (int i = 0; i < matrixSize; i++)  
		hostAuxVector[i]=0.0f;
	//boundary conditions
	for (int i = 0; i < matrixSide+2; i++)  
		hostAuxVector[i*(matrixSide+2)] = sin(PI*i/(matrixSide+1))*sin(PI*i/(matrixSide+1));
  
    // Copy input vectors from host memory to GPU buffers.
    checkCudaErrors(cudaMemcpy(oldMatrix, hostAuxVector, matrixSize *sizeof(float), cudaMemcpyHostToDevice));		
	dim3 threadsPerBlock(16, 16);   
	dim3 numBlocks(matrixSide/16, matrixSide/16);
    
	cudaEvent_t start, stop;
	float time, total_time = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int final_its;
	float max_abs_diff=1.0;
	for (final_its = 0; max_abs_diff > accuracy && final_its < maxIterations; final_its++)
	{	
		cudaEventRecord(start, 0); 
		JacobiStep<<<numBlocks, threadsPerBlock, (threadsPerBlock.x+2)*(threadsPerBlock.y+2)*sizeof(float)>>>(oldMatrix,newMatrix);

		cudaEventRecord(stop, 0); // 0 - the default stream
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		total_time += time;
		if ((final_its+1) % 1000 == 0)
		{
			 max_abs_diff =GetMaxDiff(oldMatrix,analyticalDev,matrixSize);		
			 printf("%f\n",max_abs_diff);
			 fprintf(evolutionFile,"%f %f\n",total_time,max_abs_diff);
		}
		std::swap(oldMatrix, newMatrix);           
	}        

	cudaDeviceSynchronize();
	cudaEventDestroy(start);
	cudaEventDestroy(stop);   

	max_abs_diff =GetMaxDiff(oldMatrix,analyticalDev,matrixSize);

	QueryPerformanceCounter(&t_fin);\
	QueryPerformanceFrequency(&freq);\
	double program_time = (double)(t_fin.QuadPart - t_ini.QuadPart) / (double)freq.QuadPart;

	fprintf(evolutionFile,"----------------------------\n");
	fclose(evolutionFile);
	PrintResults(argv[1],matrixSide,  final_its, total_time,  program_time,  matrixSize,  accuracy,  max_abs_diff);

	checkCudaErrors(cudaFree(oldMatrix));
	checkCudaErrors(cudaFree(newMatrix));
	checkCudaErrors(cudaFreeHost(hostAuxVector));
  }
  checkCudaErrors( cudaDeviceReset());  
   
}
