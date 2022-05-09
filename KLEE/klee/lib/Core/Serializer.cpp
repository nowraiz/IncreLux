//
// Created by pseudo on 2/21/22.
//

#include "Serializer.h"
#include "llvm/IR/Function.h"
#include <iostream>

#define MAX_TUPLE 640
bool Serializer::isValidInstruction(llvm::Instruction *ins) {
    // assuming for now an instruction is valid if it contains a canonical name metadata node
    if (ins->getMetadata("canonical.name") != NULL)
        return true;
    return false;
}

void Serializer::addTrainingTuple(const std::vector<llvm::Instruction *> instructions,
                                  bool feasible, bool valid) {
    if (!valid) return;
    if (!dumpPaths) return;
    tuple_count++;
//    trainingData.push_back(PathTuple(instructions, feasible));
    // instant dump
    output << feasible << "\n";
    dumpInstructions(instructions, output);
}

void Serializer::dumpInstructions(const std::vector<llvm::Instruction*> path,
                                  std::ofstream &output) {
    output << path.size() << "\n";
    for (llvm::Instruction *i : path) {
        output << extractCanonicalName(i) << ";;";
    }
    output << "\n";
}

std::string Serializer::extractCanonicalName(const llvm::Instruction* instruction) {
    /*
     * Returns the canonical name using the index present in the metadata of the given instruction (also used as
     * the instruction id)
     */

    llvm::MDNode* N = instruction->getMetadata("canonical.name");
    std::string idx = dyn_cast<llvm::MDString>(N->getOperand(0))->getString().str();
    return getCanonicalName(instruction, idx);
}

std::string Serializer::getCanonicalName(const llvm::Instruction *instruction, std::string idx) {
    /*
     * Returns the canonical name for the given instruction to uniquely identify
     * it outside of the llvm ecosystem. This could be done better by doing it
     * once for every instruction instead of searching for the indexes for every
     * instruction
     */
    const llvm::BasicBlock *basicBlock = instruction->getParent();

    std::stringstream canonicalName;
    canonicalName << basicBlock->getName().str() << "-" << idx;
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

