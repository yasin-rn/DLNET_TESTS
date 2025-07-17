
import torch

def fill_with_coeff(shape, start_val, x_coef, y_coef):
    """
    Fills a 2D tensor with values based on their indices, mimicking the
    C++ FillWithCoeff function.
    Value at (i, j) = start_val + (i * x_coef) + (j * y_coef)
    """
    rows, cols = shape
    # Create a column vector for the row-dependent part of the value
    i_vals = torch.arange(rows, dtype=torch.float32).view(-1, 1) * x_coef
    # Create a row vector for the column-dependent part of the value
    j_vals = torch.arange(cols, dtype=torch.float32) * y_coef
    # Use broadcasting to add the components together with the start value
    tensor = start_val + i_vals + j_vals
    return tensor

def main():
    """
    Main function to replicate the tensor chunking logic from kernel.cu.
    """
    # Mimic: Tensor<float> Input({ 16,8 });
    # Mimic: Functions::FillWithCoeff(Input, 0.1f, 0.01f, 0.001f);
    input_tensor_2d = fill_with_coeff((16, 8), 0.1, 0.01, 0.001)

    # Mimic: auto Input4D = Input.Reshape({ 2,2,4,8 });
    input_4d = input_tensor_2d.reshape(2, 2, 4, 8)

    # --- Chunk along dimension 0 ---
    # Mimic: auto chunks_dim0 = Input4D.Chunk(0, 2);
    chunks_dim0 = torch.chunk(input_4d, 2, dim=0)
    print("Original Tensor:\n", input_4d, "\n\n")
    print("--- Chunked along Dimension 0 ---")
    print("Chunk 1:\n", chunks_dim0[0], "\n")
    print("Chunk 2:\n", chunks_dim0[1], "\n\n")

    # --- Chunk along dimension 1 ---
    # Mimic: auto chunks_dim1 = Input4D.Chunk(1, 2);
    chunks_dim1 = torch.chunk(input_4d, 2, dim=1)
    print("--- Chunked along Dimension 1 ---")
    print("Chunk 1:\n", chunks_dim1[0], "\n")
    print("Chunk 2:\n", chunks_dim1[1], "\n\n")

    # --- Chunk along dimension 2 ---
    # Mimic: auto chunks_dim2 = Input4D.Chunk(2, 2);
    chunks_dim2 = torch.chunk(input_4d, 2, dim=2)
    print("--- Chunked along Dimension 2 ---")
    print("Chunk 1:\n", chunks_dim2[0], "\n")
    print("Chunk 2:\n", chunks_dim2[1], "\n\n")

    # --- Chunk along dimension 3 ---
    # Mimic: auto chunks_dim3 = Input4D.Chunk(3, 2);
    chunks_dim3 = torch.chunk(input_4d, 2, dim=3)
    print("--- Chunked along Dimension 3 ---")
    print("Chunk 1:\n", chunks_dim3[0], "\n")
    print("Chunk 2:\n", chunks_dim3[1], "\n\n")


if __name__ == "__main__":
    # Set print options for better readability
    torch.set_printoptions(precision=3, sci_mode=False)
    main()
