#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  // Persistent HBM blocks. They are grown by concatenation directly in HBM
  // every round (cost 25 * size, and it never stalls the calculation queue
  // because both inputs are already in HBM). They are copied into SRAM only
  // when a matmul actually needs them. Releasing the operands immediately after
  // each matmul keeps the peak SRAM occupancy at the theoretical minimum, which
  // is exactly what the scoring formula rewards.
  Matrix *K_block = keys[0];   // [n, d], in HBM
  Matrix *V_block = values[0]; // [n, d], in HBM

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t n = i + 1;

    // 1) Grow the persistent HBM blocks with the new key/value rows.
    if (i > 0) {
      Matrix *nk = matrix_memory_allocator.Allocate("K_block");
      gpu_sim.Concat(K_block, keys[i], nk, 0, Position::kInGpuHbm);
      K_block = nk;
      Matrix *nv = matrix_memory_allocator.Allocate("V_block");
      gpu_sim.Concat(V_block, values[i], nv, 0, Position::kInGpuHbm);
      V_block = nv;
    }

    // 2) S = Q * K^T, shape [n, n].
    //    Qw: copy the query block (HBM) into a fresh matrix, then move it to
    //    SRAM. current_query is used only here, so we do not keep it resident.
    Matrix *Qw = matrix_memory_allocator.Allocate("Qw");
    gpu_sim.Copy(current_query, Qw, Position::kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(Qw);

    //    Kt: copy K_block (HBM) into a fresh matrix, move to SRAM, then
    //    transpose IN PLACE (the simulator's Transpose has no result operand).
    Matrix *Kt = matrix_memory_allocator.Allocate("Kt");
    gpu_sim.Copy(K_block, Kt, Position::kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(Kt);
    gpu_sim.Transpose(Kt, Position::kInSharedMemory); // Kt = K_block^T in place

    Matrix *S = matrix_memory_allocator.Allocate("S");
    gpu_sim.MatMul(Qw, Kt, S);
    gpu_sim.ReleaseMatrix(Qw);
    gpu_sim.ReleaseMatrix(Kt);

    // 3) Softmax along each row of S (plain softmax matches the reference).
    Matrix *P = nullptr;
    for (size_t r = 0; r < n; ++r) {
      Matrix *Sr = matrix_memory_allocator.Allocate("Sr");
      gpu_sim.GetRow(S, r, Sr, Position::kInSharedMemory);
      Matrix *Er = matrix_memory_allocator.Allocate("Er");
      gpu_sim.MatExp(Sr, Er);
      Matrix *sumE = matrix_memory_allocator.Allocate("sumE");
      gpu_sim.Sum(Er, sumE);
      Matrix *Pr = matrix_memory_allocator.Allocate("Pr");
      gpu_sim.MatDiv(Er, sumE, Pr);
      gpu_sim.ReleaseMatrix(Sr);
      gpu_sim.ReleaseMatrix(Er);
      gpu_sim.ReleaseMatrix(sumE);
      if (P == nullptr) {
        P = Pr;
      } else {
        Matrix *np = matrix_memory_allocator.Allocate("P");
        gpu_sim.Concat(P, Pr, np, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(Pr);
        P = np;
      }
    }
    gpu_sim.ReleaseMatrix(S);

    // 4) O = P * V_block, shape [n, d].
    Matrix *Vs = matrix_memory_allocator.Allocate("Vs");
    gpu_sim.Copy(V_block, Vs, Position::kInGpuHbm);
    gpu_sim.MoveMatrixToSharedMem(Vs);
    Matrix *O = matrix_memory_allocator.Allocate("O");
    gpu_sim.MatMul(P, Vs, O);
    gpu_sim.ReleaseMatrix(P);
    gpu_sim.ReleaseMatrix(Vs);
    gpu_sim.MoveMatrixToGpuHbm(O);

    // 5) Run the simulator and commit the answer (must be in HBM).
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*O);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
