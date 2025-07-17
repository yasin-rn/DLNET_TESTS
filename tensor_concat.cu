
int main()
{
	Tensor<float> TensorA({ 16,8 });
	Functions::FillWithCoeff(TensorA, 0.1f, 0.01f, 0.001f);

	Tensor<float> TensorB({ 16,8 });
	Functions::FillWithCoeff(TensorB, 0.2f, 0.02f, 0.002f);

	auto TensorA4D = TensorA.Reshape({ 2,2,4,8 });
	auto TensorB4D = TensorB.Reshape({ 2,2,4,8 });

	std::vector<Tensor<float>> Tensors = { TensorA4D ,TensorB4D };

	Tensor<float> Concat_Dim0 = Functions::Concat(Tensors, 0);
	Tensor<float> Concat_Dim1 = Functions::Concat(Tensors, 1);
	Tensor<float> Concat_Dim2 = Functions::Concat(Tensors, 2);
	Tensor<float> Concat_Dim3 = Functions::Concat(Tensors, 3);

	std::cout << "TensorA :\n" << TensorA4D.ToString() << "\n\n";
	std::cout << "TensorB :\n" << TensorB4D.ToString() << "\n\n";

	std::cout << "Concat_Dim0(A,B) :\n" << Concat_Dim0.ToString() << "\n\n";
	std::cout << "Concat_Dim1(A,B) :\n" << Concat_Dim1.ToString() << "\n\n";
	std::cout << "Concat_Dim2(A,B) :\n" << Concat_Dim2.ToString() << "\n\n";
	std::cout << "Concat_Dim3(A,B) :\n" << Concat_Dim3.ToString() << "\n\n";


	return 0;
}