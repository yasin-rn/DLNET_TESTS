
int main()
{
Tensor<float> Input({ 16,8 });
Functions::FillWithCoeff(Input, 0.1f, 0.01f, 0.001f);
auto Input4D = Input.Reshape({ 2,2,4,8 });


auto chunks_dim0 = Input4D.Chunk(0, 2);

std::cout << "Tensor:\n" << Input4D.ToString() << "\n\n";

std::cout << "Chunked Dim:0 Part:1\n" << chunks_dim0[0].ToString() << "\n\n";
std::cout << "Chunked Dim:0 Part:2\n" << chunks_dim0[1].ToString() << "\n\n";


auto chunks_dim1 = Input4D.Chunk(1, 2);

std::cout << "Tensor:\n" << Input4D.ToString() << "\n\n";

std::cout << "Chunked Dim:1 Part:1\n" << chunks_dim1[0].ToString() << "\n\n";
std::cout << "Chunked Dim:1 Part:2\n" << chunks_dim1[1].ToString() << "\n\n";


auto chunks_dim2 = Input4D.Chunk(2, 2);

std::cout << "Tensor:\n" << Input4D.ToString() << "\n\n";

std::cout << "Chunked Dim:2 Part:1\n" << chunks_dim2[0].ToString() << "\n\n";
std::cout << "Chunked Dim:2 Part:2\n" << chunks_dim2[1].ToString() << "\n\n";



auto chunks_dim3 = Input4D.Chunk(3, 2);

std::cout << "Tensor:\n" << Input4D.ToString() << "\n\n";

std::cout << "Chunked Dim:3 Part:1\n" << chunks_dim3[0].ToString() << "\n\n";
std::cout << "Chunked Dim:3 Part:2\n" << chunks_dim3[1].ToString() << "\n\n";


return 0;
}
