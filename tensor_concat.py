
import torch

def fill_with_coeff(shape, start_val, x_coef, y_coef):
    """
    Fills a 2D tensor with values based on their indices, mimicking the
    C++ FillWithCoeff function.
    Value at (i, j) = start_val + (i * x_coef) + (j * y_coef)
    """
    rows, cols = shape
    i_vals = torch.arange(rows, dtype=torch.float32).view(-1, 1) * x_coef
    j_vals = torch.arange(cols, dtype=torch.float32) * y_coef
    tensor = start_val + i_vals + j_vals
    return tensor

def main():
    """
    Main function to replicate the tensor concatenation logic from kernel.cu.
    """
    # Mimic: Tensor<float> TensorA({ 16,8 });
    # Mimic: Functions::FillWithCoeff(TensorA, 0.1f, 0.01f, 0.001f);
    tensor_a_2d = fill_with_coeff((16, 8), 0.1, 0.01, 0.001)

    # Mimic: Tensor<float> TensorB({ 16,8 });
    # Mimic: Functions::FillWithCoeff(TensorB, 0.2f, 0.02f, 0.002f);
    tensor_b_2d = fill_with_coeff((16, 8), 0.2, 0.02, 0.002)

    # Mimic: auto TensorA4D = TensorA.Reshape({ 2,2,4,8 });
    # Mimic: auto TensorB4D = TensorB.Reshape({ 2,2,4,8 });
    tensor_a_4d = tensor_a_2d.reshape(2, 2, 4, 8)
    tensor_b_4d = tensor_b_2d.reshape(2, 2, 4, 8)

    tensors_to_concat = [tensor_a_4d, tensor_b_4d]

    # --- Concatenate along different dimensions ---
    # Mimic: Tensor<float> Concat_Dim0 = Functions::Concat(Tensors, 0);
    concat_dim0 = torch.cat(tensors_to_concat, dim=0)
    # Mimic: Tensor<float> Concat_Dim1 = Functions::Concat(Tensors, 1);
    concat_dim1 = torch.cat(tensors_to_concat, dim=1)
    # Mimic: Tensor<float> Concat_Dim2 = Functions::Concat(Tensors, 2);
    concat_dim2 = torch.cat(tensors_to_concat, dim=2)
    # Mimic: Tensor<float> Concat_Dim3 = Functions::Concat(Tensors, 3);
    concat_dim3 = torch.cat(tensors_to_concat, dim=3)

    # --- Print results ---
    print("Tensor A:\n", tensor_a_4d, "\n\n")
    print("Tensor B:\n", tensor_b_4d, "\n\n")

    print("--- Concatenated along Dimension 0 ---\n", concat_dim0, "\n\n")
    print("--- Concatenated along Dimension 1 ---\n", concat_dim1, "\n\n")
    print("--- Concatenated along Dimension 2 ---\n", concat_dim2, "\n\n")
    print("--- Concatenated along Dimension 3 ---\n", concat_dim3, "\n\n")


if __name__ == "__main__":
    # Set print options for better readability
    torch.set_printoptions(precision=3, sci_mode=False)
    main()
