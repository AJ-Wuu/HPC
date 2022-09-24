__global__ void vscale(const float *a, float *b, unsigned int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {  // validate the index
        b[index] *= a[index];
    }
}