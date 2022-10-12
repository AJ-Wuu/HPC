#include "matmul.cuh"

template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n) {
    extern __shared__ char sharedArray_temp[]; // need to be declared as a non-generic type to avoid error
    T *sharedArray = reinterpret_cast<T*>(sharedArray_temp); // pointer type casting
    T *sharedA = &sharedArray[0];
    T *sharedB = &sharedArray[blockDim.x * 2];

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int nBlocks = (n + blockDim.x - 1) / blockDim.x;
    int indexA = blockIdx.y * blockDim.x * n; // y == row, x == column
    int indexB = blockIdx.x * blockDim.x;
    int indexA_global, indexB_global, indexC_global, indexA_shared, indexB_shared;
    T c_temp = 0;
    for (int i=0; i<nBlocks; i++) {
        indexA_global = threadIdx.y * n + threadIdx.x + indexA;
        indexB_global = threadIdx.y * n + threadIdx.x + indexB;
        indexA_shared = threadIdx.x + threadIdx.y * blockDim.x;
        indexB_shared = indexA_shared;

        // take value from global memory to shared memory
        if (indexA_global < n * n) {
            sharedA[indexA_shared] = A[indexA_global];
        }
        else {
            sharedA[indexA_shared] = 0;
        }
        if (indexB_global < n * n) {
            sharedB[indexB_shared] = B[indexB_global];
        }
        else {
            sharedB[indexB_shared] = 0;
        }
        __syncthreads();

        // calculate result
        for (int j=0; j<blockDim.x; j++) {
            c_temp += sharedA[threadIdx.y * blockDim.x + j] * sharedB[threadIdx.x + blockDim.x * j];
        }
        __syncthreads();

        // prepare the index for next round
        indexA += blockDim.x;
        indexB += blockDim.x * n;
    }

    // verify valid thread
    if (index < n) {
        indexC_global = n * (threadIdx.y + blockDim.x * blockIdx.y) + (threadIdx.x + blockDim.x * blockIdx.x);
        C[indexC_global] = c_temp;
    }
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    int grid_dim = (n + block_dim - 1) / block_dim;
    matmul_kernel<int><<<dim3(grid_dim, grid_dim), dim3(block_dim, block_dim), block_dim * block_dim * 2 * sizeof(int)>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    int grid_dim = (n + block_dim - 1) / block_dim;
    matmul_kernel<float><<<dim3(grid_dim, grid_dim), dim3(block_dim, block_dim), block_dim * block_dim * 2 * sizeof(float)>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    int grid_dim = (n + block_dim - 1) / block_dim;
    matmul_kernel<double><<<dim3(grid_dim, grid_dim), dim3(block_dim, block_dim), block_dim * block_dim * 2 * sizeof(double)>>>(A, B, C, n);
    cudaDeviceSynchronize();
}
