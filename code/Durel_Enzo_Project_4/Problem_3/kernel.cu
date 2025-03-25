__global__
void blur_kernel(int* in, int* out, int* filter, int w, int h, int filter_size)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int blurSize = filter_size / 2;

    if (Col < w && Row < h)
    {
	int pixVal = 0;
	
	for (int blurRow = -blurSize; blurRow < blurSize+1; ++blurRow)
	{
	    for (int blurCol = -blurSize; blurCol < blurSize+1; ++blurCol)
	    {
		int curRow = Row + blurRow;
		int curCol = Col + blurCol;

		if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
		{
		    int filter_x = blurCol + blurSize;
		    int filter_y = blurRow + blurSize;
		    int filter_value = filter[filter_y * filter_size + filter_x];
		    pixVal += in[curRow * w + curCol] * filter_value;
		}
	    }
	}

	out[Row * w + Col] = pixVal;
    }
}
