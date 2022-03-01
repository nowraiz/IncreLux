//
// Created by pseudo on 2/21/22.
//

#include "Serializer.h"
#include "llvm/IR/Function.h"
#include <iostream>

bool Serializer::isValidInstruction(llvm::Instruction *ins) {
    return ins->getParent()->getName().str() != "";
}

void Serializer::addTrainingTuple(const std::vector<llvm::Instruction *> instructions,
                                  bool feasible, bool valid) {
    if (!valid) return;
//    trainingData.push_back(PathTuple(instructions, feasible));
    // instant dump
    output << feasible << "\n";
    dumpInstructions(instructions, output);
}

void Serializer::dumpInstructions(const std::vector<llvm::Instruction*> path,
                                  std::ofstream &output) {
    output << path.size() << "\n";
    for (llvm::Instruction *i : path) {
        output << getCanonicalName(i) << ";;";
    }
    output << "\n";
}

std::string Serializer::getCanonicalName(const llvm::Instruction *instruction) {
    /*
     * Returns the canonical name for the given instruction to uniquely identify
     * it outside of the llvm ecosystem. This could be done better by doing it
     * once for every instruction instead of searching for the indexes for every
     * instruction
     */
    const llvm::BasicBlock *basicBlock = instruction->getParent();
    const llvm::Function *function = basicBlock->getParent();
    std::string funcName = function->getName().str();
    int insIdx = -1;
    int i = 0;

    // find the index of instruction within basic block
    for (auto it = basicBlock->begin(); it != basicBlock->end(); it++, i++) {
        const llvm::Instruction *ins = &*it;
        if (ins == instruction) {
            insIdx = i;
            break;
        }
    }

    std::stringstream canonicalName;
    canonicalName << funcName << basicBlock->getName().str() << "-" << insIdx;
    return canonicalName.str();
}

void Serializer::dumpTrainingData(const std::string &filename) {

    return;
    std::cout << "Dumping Data" << std::endl;
    std::ofstream output;
    output.open("training.data");

    for (auto tuple : trainingData) {

        output << tuple.feasible << "\n";
        dumpInstructions(tuple.instructions, output);
    }
    output.close();
}

