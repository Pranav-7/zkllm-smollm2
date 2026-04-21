#include "zksoftmax.cuh"
#include "zkfc.cuh"
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "rescaling.cuh"
#include <string>
#include <vector>
using namespace std;

int main(int argc, char *argv[])
{
    // argv: [0]=binary [1]=mode [2]=input [3]=seq_len [4]=embed_dim
    //       [5]=workdir [6]=layer_prefix [7]=output [8]=n_heads (NEW)
    if (argc < 8) {
        cerr << "Usage: " << argv[0]
             << " <linear|attn> <input> <seq_len> <embed_dim>"
                " <workdir> <layer_prefix> <o> [n_heads]" << endl;
        return 1;
    }
    string mode             = argv[1];
    string input_file_name  = argv[2];
    uint   seq_len          = std::stoi(argv[3]);
    uint   embed_dim        = std::stoi(argv[4]);
    string workdir          = argv[5];
    string layer_prefix     = argv[6];
    string output_file_name = argv[7];
    uint   n_heads          = (argc >= 9) ? std::stoi(argv[8]) : 32;

    if (mode == "linear") {
        auto X   = FrTensor::from_int_bin(input_file_name);
        auto Q_w = FrTensor::from_int_bin(workdir + "/" + layer_prefix + "-self_attn.q_proj.weight-int.bin");
        auto K_w = FrTensor::from_int_bin(workdir + "/" + layer_prefix + "-self_attn.k_proj.weight-int.bin");
        auto V_w = FrTensor::from_int_bin(workdir + "/" + layer_prefix + "-self_attn.v_proj.weight-int.bin");

        auto Q = FrTensor::matmul(X, Q_w, seq_len, embed_dim, embed_dim);
        auto K = FrTensor::matmul(X, K_w, seq_len, embed_dim, embed_dim);
        auto V = FrTensor::matmul(X, V_w, seq_len, embed_dim, embed_dim);

        Q.save_int("temp_Q.bin");
        K.save_int("temp_K.bin");
        V.save_int("temp_V.bin");
        cout << "Linear projection done. Q,K,V saved." << endl;
        return 0;
    }
    else if (mode == "attn") {
        auto Q_full = FrTensor::from_int_bin("temp_Q.bin");
        auto K_full = FrTensor::from_int_bin("temp_K.bin");
        auto V_full = FrTensor::from_int_bin("temp_V.bin");

        uint d         = embed_dim / n_heads;          // head_dim
        uint head_size = seq_len * d;
        FrTensor full_out(seq_len * embed_dim);

        for (uint h = 0; h < n_heads; h++) {
            FrTensor Q(head_size, Q_full.gpu_data + h * head_size);
            FrTensor K(head_size, K_full.gpu_data + h * head_size);
            FrTensor V(head_size, V_full.gpu_data + h * head_size);

            auto X_scores = FrTensor::matmul(Q, K.transpose(seq_len, d), seq_len, d, seq_len);

            // zkSoftmax tuned for small per-head attention: bits {1<<8, 1<<14, 1<<14}
            // K=1, L=0, range 1UL<<32, theta {1<<14, 1<<18}
            zkSoftmax softmax({1<<8, 1<<14, 1<<14}, 1, 0, 1UL<<32, {1<<14, 1<<18}, seq_len, seq_len, d, 1);
            Rescaling rs1(1<<20), rs2(1<<20);

            FrTensor shift(seq_len), X_shifted(seq_len * seq_len);
            vector<FrTensor> X_segments, Y_segments, m_segments;
            FrTensor Y = softmax.compute(X_scores, shift, X_shifted, X_segments, Y_segments, m_segments);

            auto head_out = FrTensor::matmul(Y, V, seq_len, seq_len, d);
            auto head_out_scaled = rs1(rs2(head_out));

            cudaMemcpy(full_out.gpu_data + h * head_size,
                       head_out_scaled.gpu_data,
                       head_size * sizeof(Fr_t),
                       cudaMemcpyDeviceToDevice);
        }

        full_out.save_int(output_file_name);
        cout << "Attention proof complete. Full output saved to "
             << output_file_name << endl;
        return 0;
    }

    cerr << "Unknown mode: " << mode << endl;
    return 1;
}
