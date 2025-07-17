
int main()
{

	int InFeatures = 8;
	int OutFeatures = 4;
	int BatchSize = 8;


	Tensor<float> Inputs({ BatchSize,InFeatures });
	Functions::FillWithCoeff(Inputs, 0.1f, 0.01f, 0.001f);


	Tensor<float> Weights({ OutFeatures,InFeatures });
	Functions::FillWithCoeff(Weights, 0.2f, 0.02f, 0.002f);

	Tensor<float> Bias({ 1,OutFeatures });
	Functions::FillWithCoeff(Bias, 0.3f, 0.03f, 0.003f);



	std::cout << "Inputs :\n" << Inputs.ToString() << "\n\n";
	std::cout << "Weights :\n" << Weights.ToString() << "\n\n";
	std::cout << "Bias :\n" << Bias.ToString() << "\n\n";



	Linear<float> LinearF(InFeatures, OutFeatures, true);
	LinearF.SetWeights(Weights, Bias);

	auto Output = LinearF.Forward(Inputs);

	std::cout << "Output :\n" << Output.ToString() << "\n\n";

	return 0;
}