
int main()
{
	int BatchSize = 2;
	int SeqLen = 5;
	int InputSize = 8;
	int HiddenSize = 10;
	int NumLayers = 1;

	Tensor<float> Input2D({ BatchSize * SeqLen, InputSize });
	Tensor<float> W_ih({ HiddenSize,InputSize });
	Tensor<float> W_hh({ HiddenSize,HiddenSize });
	Tensor<float> Weights({ HiddenSize * HiddenSize + InputSize * HiddenSize });

	Functions::FillWithCoeff(Input2D, 0.5f, 0.4f, 0.3f);
	Functions::FillWithCoeff(W_ih, 0.1f, 0.2f, 0.3f);
	Functions::FillWithCoeff(W_hh, 0.4f, 0.5f, 0.6f);

	Tensor<float> Input = Input2D.Reshape({ BatchSize , SeqLen, InputSize });

	cudaMemcpy(Weights.GetDataPtr(), W_ih.GetDataPtr(), W_ih.GetMemSize(), cudaMemcpyDeviceToDevice);
	cudaMemcpy(Weights.GetDataPtr() + W_ih.GetTotalSize(), W_hh.GetDataPtr(), W_hh.GetMemSize(), cudaMemcpyDeviceToDevice);

	Recurrent<float> recurrent(BatchSize, InputSize, HiddenSize, NumLayers, CUDNN_RNN_TANH);
	recurrent.SetWeights(Weights);
	Tensor<float> Output = recurrent.Forward(Input);

	std::cout << Output.ToString();
	return 0;
}
