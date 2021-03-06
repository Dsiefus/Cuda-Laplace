#include "cuda_runtime.h"
#include <helper_cuda.h>
#include <helper_timer.h>
#include <stdio.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

#define PI 3.1415926535897932384626433832795

template <typename T>
struct abs_diff : public thrust::binary_function<T,T,T>
{
    __host__ __device__
    T operator()(const T& a, const T& b)
    {
        return std::fabs(b - a);
    }
};



//indexing of shared memory. Threads have 2 more rows and cols (for halo nodes)
__device__ inline int getSharedIndex(int thrIdx, int thrIdy)
{
	return (thrIdy * blockDim.x + thrIdx);
}

//indexing of global memory corresponding to each thread
__device__ inline int getGlobalIndex()
{	
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	return col + row*blockDim.x * gridDim.x;
}

__global__ void JacobiStep(const float *oldMatrix, float *newMatrix)
{
	extern __shared__ float aux[];
	int thx = threadIdx.x, thy = threadIdx.y;	
	aux[ getSharedIndex(thx, thy)] = oldMatrix[getGlobalIndex()];	

	int leftIndex = getSharedIndex(thx-1,thy);
	int rightIndex = getSharedIndex(thx+1,thy);
	int topIndex = getSharedIndex(thx,thy-1);
	int botIndex = getSharedIndex(thx,thy+1);
	float left, right, top, bot;
	__syncthreads();
	
	if (thx == 0) 	   {
		 (blockIdx.x == 0) ?
			left = __sinf((PI*((thy + blockDim.y * blockIdx.y)+1))/
													(blockDim.y*gridDim.y+1))*
					__sinf((PI*((thy + blockDim.y * blockIdx.y)+1))/
													(blockDim.y*gridDim.y+1)) 			
		   : left = oldMatrix[getGlobalIndex()-1] ;
	}	
	else
		left = aux[leftIndex];
		
	
	if  (thy == 0) {
		 (blockIdx.y == 0) ?
			  top = 0.0
			: top = oldMatrix[getGlobalIndex()-blockDim.x * gridDim.x];		
	}
	else
		top = aux[topIndex];

	//right
	if (thx == blockDim.x-1){
		(blockIdx.x == gridDim.x-1) ?
			  right = 0.0f
			: right = oldMatrix[getGlobalIndex()+1];
	}
	else
		right = aux[rightIndex];

	//bot
	if (thy == blockDim.y - 1){
		(blockIdx.y == gridDim.y - 1) ?
			  bot = 0.0f
			: bot = oldMatrix[getGlobalIndex()+blockDim.x * gridDim.x];
	}
	else
		bot = aux[botIndex];

	float newValue =  0.25*(left+right+top+bot);	
	newMatrix[getGlobalIndex()] = newValue;
}

//Returns the maximum absolute difference between the elements of a float vector 
//on device memory, and a thrust device_vector, both with size matrixSize
float GetMaxDiff( float* a, thrust::device_vector<float> b, int matrixSize)
{
	thrust::device_ptr<float> dev_ptra =  thrust::device_pointer_cast(a);
	
    float init = 0;    
    thrust::maximum<float> binary_op1;
    abs_diff<float> binary_op2;
    return thrust::inner_product(dev_ptra,dev_ptra +  matrixSize,b.begin(), init,
   								binary_op1, binary_op2); 
}

thrust::host_vector<float> GetAnalyticalMatrix(char* matrixSide, int matrixSize)
{
	char analyticalFilename[50];
	analyticalFilename[0]=0;
	strcpy(analyticalFilename,matrixSide);
	strcat(analyticalFilename,"_1.dat");
	FILE *matrixFile = fopen(analyticalFilename, "rb");	
	if (matrixFile == NULL)
	{
		printf("Error: Analytical solution file not found\n");
		exit( -1);
	}
	thrust::host_vector<float> analyticalHost(matrixSize);	
	//We read directly from the file to a thrust host vector, 
	//which is then copied to a device vector
	int n=fread(&analyticalHost[0],sizeof(float),matrixSize,matrixFile);
	fclose(matrixFile);	 
	return analyticalHost;
}

void PrintResults(char* size,int matrixSide, int final_its,float total_time, 
		float program_time, int matrixSize, float accuracy, float max_abs_diff)
{
	char outputFileName[50];
	outputFileName[0]=0;
	strcpy(outputFileName,size);
	strcat(outputFileName,"_shared_no_bound_times.txt");
	FILE* outfile = fopen(outputFileName,"a+");
	if (outfile == NULL)
	{
		printf("Error with output file\n");
		exit(-1);
	}

	printf("Time for matrixSide= %d, %d Iterations. %f ms. Total time: %f.",
			matrixSide,final_its, total_time, program_time);
	printf("Memory bandwith is %f GB/s.", 
	   ((1e-6)*matrixSize)*2*final_its*sizeof(float)/(total_time)); 
	printf("Accuracy desired: %f (obtained %f)\n",accuracy,max_abs_diff);
	fprintf(outfile,"Iterations: %d. Time: %f ms. Accuracy desired: %f ",
	 		final_its, total_time,accuracy);
	fprintf(outfile,"(obtained %f). Memory bandwith: %f GB/s\n",
	  max_abs_diff,((1e-6)*matrixSize)*2*final_its*sizeof(float)/(total_time)); 

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
	if(accuracy > 0.5)
	{
		printf("Error: accuracy must be smaller than 0.5\n");
		return -1;
	}
	


	char evolutionFileName[50];
evolutionFileName[0]=0;
	strcpy(evolutionFileName,argv[1]);
	strcat(evolutionFileName,"_shared_no_bound.txt");
	FILE* evolutionFile = fopen (evolutionFileName, "a+");

			
	int maxIterations = 200000;	
	const int matrixSize = (matrixSide)*(matrixSide);
     
	
	{//block to encapuslate thrust
	thrust::device_vector<float> analyticalDev = 
										GetAnalyticalMatrix(argv[1],matrixSize);
	float *oldMatrix = 0, *newMatrix = 0;
	checkCudaErrors( cudaMalloc((void**)&oldMatrix, matrixSize*sizeof(float)));	
	checkCudaErrors( cudaMalloc((void**)&newMatrix, matrixSize*sizeof(float)));	

	float* hostAuxVector = 0;
	checkCudaErrors(cudaHostAlloc((void**) &hostAuxVector,
							matrixSize * sizeof(float), cudaHostAllocDefault));
	for (int i = 0; i < matrixSize; i++)  
	  hostAuxVector[i]=0.0f;
  
    // Copy input vectors from host memory to GPU buffers.
    checkCudaErrors(cudaMemcpy(oldMatrix, hostAuxVector, 
    						matrixSize *sizeof(float), cudaMemcpyHostToDevice));	
	dim3 threadsPerBlock(16, 16);   
	dim3 numBlocks(matrixSide/16, matrixSide/16);
    
	cudaEvent_t start, stop;
	float time, total_time = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int final_its;
	float max_abs_diff=1.0;
	for (final_its = 0; max_abs_diff > accuracy && final_its < maxIterations; 
																	final_its++)
	{	
		cudaEventRecord(start, 0); 
		JacobiStep<<<numBlocks, threadsPerBlock, (threadsPerBlock.x+2)*
					(threadsPerBlock.y+2)*sizeof(float)>>>(oldMatrix,newMatrix);

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
	double program_time = (double)(t_fin.QuadPart - t_ini.QuadPart) / 
						  (double)freq.QuadPart;

	fprintf(evolutionFile,"----------------------------\n");
	fclose(evolutionFile);
	PrintResults(argv[1],matrixSide,  final_its, total_time,  program_time,  
				 matrixSize,  accuracy,  max_abs_diff);	
	 
	checkCudaErrors(cudaFree(oldMatrix));
	checkCudaErrors(cudaFree(newMatrix));
}
checkCudaErrors( cudaDeviceReset());  
  
}
