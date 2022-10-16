#include "scan.cuh"

__global__ void hillis_steele(float *g_odata, float *g_idata, float *g_additional, int n, int isAdditionalBlock) {
    extern volatile __shared__  float temp[]; // allocated on invocation
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int pout = 0, pin = 1;

    // boundary check
    if (index < n) {
        temp[threadIdx.x] = g_idata[index]; // load input into shared memory for inclusive scan
        __syncthreads();

        for (int offset = 1; offset < blockDim.x; offset *= 2 ) {
            pout = 1 - pout; // swap double buffer indices
            pin  = 1 - pout;

            int index1 = pout * blockDim.x + threadIdx.x;
            int index2 = pin * blockDim.x + threadIdx.x;
            if (threadIdx.x >= offset) {
                temp[index1] = temp[index2] + temp[index2 - offset];
            }
    	    else {
                temp[index1] = temp[index2];
            }

            __syncthreads(); // need to sync before starting next iteration 
        }
    
        int index3 = pout * n + threadIdx.x;
        if (pout * blockDim.x + threadIdx.x < blockDim.x) {
            g_odata[index] = temp[index3];
        }
        if (isAdditionalBlock == 0 && threadIdx.x == blockDim.x - 1) {
            g_additional[blockIdx.x] = temp[index3];
        }
    }
}

__global__ void inclusive_add(float *g_odata, float *g_otemp, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n && blockIdx.x != 0) {
        g_odata[index] += g_otemp[blockIdx.x - 1];
    }
}

__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block) {
    int nBlocks = (n + threads_per_block - 1) / threads_per_block;
    float *g_idata, *g_odata, *g_itemp, *g_otemp, *g_temp;
    cudaMalloc(&g_idata, n * sizeof(float));
    cudaMalloc(&g_odata, n * sizeof(float));
    cudaMalloc(&g_itemp, nBlocks * sizeof(float));
    cudaMalloc(&g_otemp, nBlocks * sizeof(float));
    cudaMalloc(&g_temp, nBlocks * sizeof(float));

    // copy from host to device
    cudaMemcpy(g_idata, input, n * sizeof(float), cudaMemcpyHostToDevice);

    int size = 2 * threads_per_block * sizeof(float);
    hillis_steele<<<nBlocks, threads_per_block, size>>>(g_odata, g_idata, g_itemp, n, 0);
    hillis_steele<<<1, threads_per_block, size>>>(g_otemp, g_itemp, g_temp, nBlocks, 1);
    inclusive_add<<<nBlocks, threads_per_block>>>(g_odata, g_otemp, n);

    // copy from device to host
    cudaMemcpy(output, g_odata, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	// free malloc
	cudaFree(g_idata);
  	cudaFree(g_odata);
  	cudaFree(g_itemp);
    cudaFree(g_otemp);
    cudaFree(g_temp);
}
