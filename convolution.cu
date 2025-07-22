int main()
{
	int BatchSize = 1;
	int InChannels = 3;
	int OutChannels = 2;
	int KernelSize = 4;
	int InH = 8;
	int InW = 8;

	Tensor<float> Image2D({ BatchSize * InChannels * InH,InW });
	Tensor<float> Weights2D({ OutChannels * InChannels * KernelSize,KernelSize });
	Functions::FillWithCoeff(Image2D, 0.5f, 0.4f, 0.3f);
	Functions::FillWithCoeff(Weights2D, 0.1f, 0.2f, 0.3f);

	Tensor<float> Image = Image2D.Reshape({ BatchSize,InChannels,InH,InW });
	Tensor<float> Weights = Weights2D.Reshape({ OutChannels , InChannels , KernelSize,KernelSize });

	std::cout << "Image:\n" << Image.ToString() << "\n\n";
	std::cout << "Weights:\n" << Weights.ToString() << "\n\n";

	Convolution<float> ConvolutinLayer(InChannels, OutChannels, KernelSize, 1, 2);
	ConvolutinLayer.SetWeights(Weights);

	const auto& Output = ConvolutinLayer.Forward(Image);
	std::cout << "Output:\n" << Output.ToString() << "\n\n";


	return 0;
}