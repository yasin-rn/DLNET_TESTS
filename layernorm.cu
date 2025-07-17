int main()
{

	int ModelSize = 8;
	int SeqLen = 5;
	int BatchSize = 2;
	int BeamSize = 2;


	Tensor<float> Inputs({ BatchSize * BeamSize * SeqLen,ModelSize });
	Functions::FillWithCoeff(Inputs, 0.1f, 0.01f, 0.001f);

	Tensor<float> Inputs4D = Inputs.Reshape({ BatchSize,BeamSize,SeqLen,ModelSize });

	std::cout << "Inputs :\n" << Inputs4D.ToString() << "\n\n";

	LayerNorm<float> LayerNorm(ModelSize);

	auto Output = LayerNorm.Forward(Inputs4D);

	std::cout << "Output :\n" << Output.ToString() << "\n\n";

	return 0;
}