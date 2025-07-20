template <typename T>
__global__ void FillWithCoeffKernel(T* data, int ndim, int dim0_size, int dim1_size, int dim2_size, T startVal, T xCoef, T yCoef)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = gridDim.x * blockDim.x;

	size_t total_elements = (ndim == 2) ? (size_t)dim0_size * dim1_size : (size_t)dim0_size * dim1_size * dim2_size;

	for (size_t i = idx; i < total_elements; i += stride)
	{
		int coord_k = 0;
		int coord_i = 0;
		int coord_j = 0;

		if (ndim == 2) {
			coord_i = i / dim1_size;
			coord_j = i % dim1_size;
		}
		else {
			int plane_size = dim1_size * dim2_size;
			coord_k = i / plane_size;
			int remaining_idx = i % plane_size;
			coord_i = remaining_idx / dim2_size;
			coord_j = remaining_idx % dim2_size;
		}
		T val1 = ((startVal * coord_i + coord_j) / startVal / 10);
		T val2 = xCoef * sin(2 * 3.14 * val1);
		T val3 = yCoef * sin(4 * 3.14 * val1);
		data[i] = val2 + val3 + 1e-5;
	}   
}