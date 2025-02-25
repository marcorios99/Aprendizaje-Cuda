#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024  // Tamaño de la matriz (NxN)

int main(void) {
    int i, j, k;
    // Reserva de memoria para las matrices A, B y C
    float *A = (float *)malloc(N * N * sizeof(float));
    float *B = (float *)malloc(N * N * sizeof(float));
    float *C = (float *)malloc(N * N * sizeof(float));
    
    if (!A || !B || !C) {
        printf("Error al asignar memoria.\n");
        return 1;
    }
    
    // Inicializa las matrices A y B con valores aleatorios y C en 0
    for (i = 0; i < N * N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
        C[i] = 0.0f;
    }
    
    // Medición del tiempo usando clock_gettime
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Multiplicación de matrices
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float sum = 0.0f;
            for (k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    // Calcula el tiempo transcurrido en segundos
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Tiempo de multiplicacion de matrices en CPU: %f segundos\n", elapsed);
    
    // Muestra un valor de la matriz resultante para verificar el cálculo
    printf("Resultado de muestra: %f\n", C[0]);
    
    // Libera la memoria asignada
    free(A);
    free(B);
    free(C);
    
    return 0;
}
