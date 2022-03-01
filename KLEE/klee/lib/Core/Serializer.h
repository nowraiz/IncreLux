//
// Created by pseudo on 2/21/22.
//

#include "klee/ExecutionState.h"
#include "klee/Internal/Module/KModule.h"
#include <fstream>
#include <string>
//
// Created by pseudo on 10/27/21.
//

#ifndef KLEE_SERIALIZER_H
#define KLEE_SERIALIZER_H

struct PathTuple {
  std::vector<llvm::Instruction *> instructions;
  bool feasible;

  PathTuple(const std::vector<llvm::Instruction *> instructions, bool feasible) {
    this->instructions = instructions;
    this->feasible = feasible;
  }
};

class Serializer {

  std::vector<PathTuple> trainingData;
  std::ofstream output;

public:
  Serializer() {
      output.open("training.data");
  }

  // check if the instruction is part of the given bitcode (HACK)
  bool isValidInstruction(llvm::Instruction* ins);

  // add a training tuple

  void addTrainingTuple(const std::vector<llvm::Instruction*> instructions,
                        bool feasible, bool valid);
  // dump the instruction trace
  void dumpInstructions(const std::vector<llvm::Instruction*> path,
                        std::ofstream &output);

  // dumps the training data into a file
  void dumpTrainingData(const std::string &filename);

  // get the canonical name for the instruction
  std::string getCanonicalName(const llvm::Instruction *instruction);

  ~Serializer() {
      output.close();
  }
};





#endif //KLEE_SERIALIZER_H
