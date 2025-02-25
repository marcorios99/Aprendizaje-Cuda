#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Tamaño de la matriz (NxN)
#define N 1024

// Kernel que realiza la multiplicación de matrices
__global__ void matrixMulKernel(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < width && col < width) {
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main(void) {
    int size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Inicializa las matrices con valores aleatorios
    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copia las matrices del host a la GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define la configuración de hilos y bloques
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    // Medición del tiempo de ejecución del kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Llama al kernel para realizar la multiplicación
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    
    // Sincroniza y mide el tiempo
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Tiempo de multiplicacion de matrices: %f ms\n", elapsedTime);

    // Copia el resultado de vuelta al host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Imprime un valor de ejemplo para verificar el resultado
    printf("Resultado de muestra: %f\n", h_C[0]);

    // Libera memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
