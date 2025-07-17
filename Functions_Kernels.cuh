#include <cuda_runtime.h>
#include <math_constants.h>

template <typename T>
__global__ void PositionalEncodingKernel(T* data, int seq_len, int model_size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t total_elements = seq_len * model_size;

    for (size_t i = idx; i < total_elements; i += stride)
    {
        int pos = i / model_size;
        int dim_j = i % model_size;

        float div_term = powf(10000.0f, (float)(2 * (dim_j / 2)) / (float)model_size);

        float angle = (float)pos / div_term;

        T pe_value = (dim_j % 2 == 0) ? sinf(angle) : cosf(angle);

        data[i] = pe_value;
    }
}

// Burada tensor yerine bulunan indexin pointeri ile view dondurulebilir;
template <typename T>
__global__ void EmbeddingLookupKernel(T* output_data,
    const int* indices_data,
    const T* weight_data,
    int embedding_dim,
    size_t total_indices)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < total_indices; i += stride)
    {
        int word_id = indices_data[i];

        const T* src_ptr = weight_data + (word_id * embedding_dim);
        T* dest_ptr = output_data + (i * embedding_dim);

        for (int j = 0; j < embedding_dim; ++j)
        {
            dest_ptr[j] = src_ptr[j];
        }
    }
}

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

		T value = startVal + (coord_i * xCoef) + (coord_j * yCoef);
		data[i] = value;
	}
}