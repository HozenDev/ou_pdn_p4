__global__
void blur_kernel(int* in, int* out, int w, int h, int blurSize)
{
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h)
    {
	int pixVal = 0;
	int pixels = 0;

	for (int blurRow = -blurSize; blurRow < blurSize+1; ++blurRow)
	{
	    for (int blurCol = -blurSize; blurCol < blurSize+1; ++blurCol)
	    {
		int curRow = Row + blurRow;
		int curCol = Col + blurCol;

		if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
		{
		    pixVal += in[curRow * w + curCol];
		    pixels++;
		}
	    }
	}

	out[Row * w + Col] = (int) (pixVal/pixels);
    }
}
