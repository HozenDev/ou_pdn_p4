#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "support.h"
#include "kernel.cu"

#define BILLION  1000000000.0
#define MAX_LINE_LENGTH 25000

#define BLUR_SIZE 2

int main (int argc, char *argv[])
{
    // Check console errors
    if( argc != 6)
    {
        printf("USE LIKE THIS: convolution_CUDA n_row n_col mat_input.csv mat_output_prob3.csv time_prob3_CUDA.csv\n");
        return EXIT_FAILURE;
    }

    // Get dims
    int n_row = strtol(argv[1], NULL, 10);
    int n_col = strtol(argv[2], NULL, 10);

    // Get files to read/write 
    FILE* inputFile1 = fopen(argv[3], "r");
    if (inputFile1 == NULL){
        printf("Could not open file %s",argv[2]);
        return EXIT_FAILURE;
    }
    FILE* outputFile = fopen(argv[4], "w");
    FILE* timeFile  = fopen(argv[5], "w");

    // Matrices to use
    int* filterMatrix_h = (int*)malloc(5 * 5 * sizeof(int));
    int* inputMatrix_h  = (int*) malloc(n_row * n_col * sizeof(int));
    int* outputMatrix_h = (int*) malloc(n_row * n_col * sizeof(int));

    // read the data from the file
    int row_count = 0;
    char line[MAX_LINE_LENGTH] = {0};
    while (fgets(line, MAX_LINE_LENGTH, inputFile1)) {
        if (line[strlen(line) - 1] != '\n') printf("\n");
        char *token;
        const char s[2] = ",";
        token = strtok(line, s);
        int i_col = 0;
        while (token != NULL) {
            inputMatrix_h[row_count*n_col + i_col] = strtol(token, NULL,10 );
            i_col++;
            token = strtok (NULL, s);
        }
        row_count++;
    }


    // Filling filter
	// 1 0 0 0 1 
	// 0 1 0 1 0 
	// 0 0 1 0 0 
	// 0 1 0 1 0 
	// 1 0 0 0 1 
    for(int i = 0; i< 5; i++)
        for(int j = 0; j< 5; j++)
            filterMatrix_h[i*5+j]=0;

    filterMatrix_h[0*5+0] = 1;
    filterMatrix_h[1*5+1] = 1;
    filterMatrix_h[2*5+2] = 1;
    filterMatrix_h[3*5+3] = 1;
    filterMatrix_h[4*5+4] = 1;
    
    filterMatrix_h[4*5+0] = 1;
    filterMatrix_h[3*5+1] = 1;
    filterMatrix_h[1*5+3] = 1;
    filterMatrix_h[0*5+4] = 1;

    fclose(inputFile1); 


    // --------------------------------------------------------------------------- //
    // ------ Algorithm Start ---------------------------------------------------- //

    Timer timer;
    int* input_matrix_d = NULL;
    int* output_matrix_d = NULL;
    int* filter_matrix_d = NULL;
    dim3 dimGrid(n_row, n_col, 1);
    dim3 dimBlock(5, 5, 1);

    cudaMalloc((void**)&input_matrix_d, n_row * n_col * sizeof(int));
    cudaMalloc((void**)&output_matrix_d, n_row * n_col * sizeof(int));
    cudaMalloc((void**)&filter_matrix_d, 5 * 5 * sizeof(int));
    
    // **************************************** //
    //         Cuda Copy Host to Device         //
    // **************************************** //

    startTime(&timer);
    cudaMemcpy(input_matrix_d, inputMatrix_h, n_row * n_col * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(filter_matrix_d, filterMatrix_h, 5*5 * sizeof(int), cudaMemcpyHostToDevice);
    stopTime(&timer);
    fprintf(stdout, "Copy H-D: %.20f\n", elapsedTime(timer));
    fprintf(timeFile, "%.20f\n", elapsedTime(timer));
    
    // ************************* //
    //        Cuda Kernel 1      //
    // ************************* //
    
    startTime(&timer);
    blur_kernel<<<dimGrid, dimBlock>>>(input_matrix_d, output_matrix_d, filter_matrix_d, n_row, n_col, 5);
    stopTime(&timer);
    fprintf(stdout, "Kernel 1: %.20f\n", elapsedTime(timer));
    fprintf(timeFile, "%.20f\n", elapsedTime(timer));

    // **************************************** //
    //         Cuda Copy Device To Host         //
    // **************************************** //

    startTime(&timer);
    cudaMemcpy(outputMatrix_h, output_matrix_d, n_row * n_col * sizeof(int), cudaMemcpyDeviceToHost);
    stopTime(&timer);
    fprintf(stdout, "Copy D-H: %.20f\n", elapsedTime(timer));
    fprintf(timeFile, "%.20f\n", elapsedTime(timer));
    
    // --------------------------------------------------------------------------- //
    // ------ Algorithm End ------------------------------------------------------ //

    // Save output matrix as csv file
    for (int i = 0; i<n_row; i++)
    {
        for (int j = 0; j<n_col; j++)
        {
            fprintf(outputFile, "%d", outputMatrix_h[i*n_col +j]);
            if (j != n_col -1)
                fprintf(outputFile, ",");
            else if ( i < n_row-1)
                fprintf(outputFile, "\n");
        }
    }

    // Cleanup
    fclose (outputFile);
    fclose (timeFile);

    free(inputMatrix_h);
    free(outputMatrix_h);
    free(filterMatrix_h);
    cudaFree(input_matrix_d);
    cudaFree(output_matrix_d);
    cudaFree(filter_matrix_d);

    return 0;
}
