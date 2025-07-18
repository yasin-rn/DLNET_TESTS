
int main()
{

	int ModelSize = 8;
	int SeqLen = 5;
	int BatchSize = 2;
	int BeamSize = 2;
	int NumHead = 2;

	Tensor<float> Inputs({ BatchSize * BeamSize * SeqLen,ModelSize });
	Functions::FillWithCoeff(Inputs, 0.1f, 0.01f, 0.001f);

	Tensor<float> Inputs4D = Inputs.Reshape({ BatchSize,BeamSize,SeqLen,ModelSize });


	Tensor<float> QW({ ModelSize,ModelSize });
	Tensor<float> KW({ ModelSize,ModelSize });
	Tensor<float> VW({ ModelSize,ModelSize });
	Tensor<float> OW({ ModelSize,ModelSize });

	Functions::FillWithCoeff(QW, 0.2f, 0.02f, 0.002f);
	Functions::FillWithCoeff(KW, 0.3f, 0.03f, 0.003f);
	Functions::FillWithCoeff(VW, 0.4f, 0.04f, 0.004f);
	Functions::FillWithCoeff(OW, 0.5f, 0.05f, 0.005f);

	std::vector<Tensor<float>> Weights = { QW,KW,VW,OW };

	auto MHA_Weights = Functions::Concat(Weights, 0);

	MultiheadAttention<float> MHA(SeqLen, ModelSize, NumHead, false, true);
	MHA.SetWeights(MHA_Weights);


	auto Output = MHA.Forward(Inputs4D);

	std::cout << "Inputs :\n" << Inputs4D.ToString() << "\n\n";
	std::cout << "MHA_Weights :\n" << MHA_Weights.ToString() << "\n\n";

	std::cout << "Output :\n" << Output.ToString() << "\n\n";

	return 0;
}