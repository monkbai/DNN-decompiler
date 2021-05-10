/*
 * Copyright 2002-2019 Intel Corporation.
 * 
 * This software is provided to you as Sample Source Code as defined in the accompanying
 * End User License Agreement for the Intel(R) Software Development Products ("Agreement")
 * section 1.L.
 * 
 * This software and the related documents are provided as is, with no express or implied
 * warranties, other than those that are expressly stated in the License.
 */

/*
 *  This file contains an ISA-portable PIN tool for tracing memory accesses.
 */

#include <stdio.h>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include "pin.H"

static std::unordered_map<ADDRINT, std::string> str_of_ins_at;
FILE * trace;

uint64_t start_addr = 0;
uint64_t end_addr = 0;

int end_flag = 0; // when end_flag > 0, stop logging

int after_call_flag = 0; // is current instruction after a call  

/* ===================================================================== */
// Command line switches
/* ===================================================================== */
KNOB<std::string> KnobOutputFile(KNOB_MODE_WRITEONCE,  "pintool",
    "o", "", "specify file name for MyPinTool output");

KNOB<uint64_t>   KnobStartAddr(KNOB_MODE_WRITEONCE,  "pintool",
    "start", "0x422860", "specify file start address of instrumentation");
    
KNOB<uint64_t>   KnobEndAddr(KNOB_MODE_WRITEONCE,  "pintool",
    "end", "0x424b6a", "count instructions, basic blocks and threads in the application");


/* ===================================================================== */
// Utilities
/* ===================================================================== */

VOID AfterCall(VOID * ip, ADDRINT rax){
    if (end_flag == 0){
        fprintf(trace,"RAX:\t%p\n", (void *)rax);
    }
}

VOID SetEndFlag(VOID * ip){
    end_flag = 1;
}

VOID RecordLea(VOID * ip, VOID * addr)
{
    if (end_flag == 0){
        std::string ins_str = str_of_ins_at[(ADDRINT)ip];
        fprintf(trace,"%p:\t%s\n", ip, ins_str.c_str());
        fprintf(trace,"R:\t%p:\n", addr);
        fprintf(trace,"M:\n");
    }
}

VOID RecordInst(VOID * ip)
{
    if (end_flag == 0){
        std::string ins_str = str_of_ins_at[(ADDRINT)ip];
        fprintf(trace,"%p:\t%s\n", ip, ins_str.c_str());
        //fprintf(trace,"N:\t%p:\t%d\n", (void *)0xDEADBEEF, 7); // not real memory op
        fprintf(trace,"N:\n");
        fprintf(trace,"M:\n");
    }
}

// Print a memory read record
VOID RecordMemRead(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
    if (end_flag == 0 ){
        std::string ins_str = str_of_ins_at[(ADDRINT)ip];
        fprintf(trace,"%p:\t%s\n", ip, ins_str.c_str());
        fprintf(trace,"R:\t%p:\t%lu\n", mem_addr, mem_size);
        if (mem_size == 4){
            fprintf(trace,"M:\t0x%x\n", *(uint32_t *)(mem_addr));
        }
        else if (mem_size == 8){
            fprintf(trace,"M:\t0x%lx\n", *(uint64_t *)(mem_addr));
        }
        else if (mem_size == 16){
            fprintf(trace,"M:\t%p\n", *(void **)(mem_addr)); // TODO: we do not care about 16 bytes value
        }
        else{
            fprintf(trace,"M:\t%p\n", *(void **)(mem_addr));
        }
        //fprintf(trace,"%p: R %p\n", ip, addr);
    }
}

// Print a memory write record
VOID RecordMemWrite(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
    if (end_flag == 0){
        std::string ins_str = str_of_ins_at[(ADDRINT)ip];
        fprintf(trace,"%p:\t%s\n", ip, ins_str.c_str());
        fprintf(trace,"W:\t%p:\t%lu\n", mem_addr, mem_size);
        fprintf(trace,"M:\n");
        //fprintf(trace,"%p: W %p\n", ip, addr);
    }
}

// Is called for every instruction and instruments reads and writes
VOID Instruction(INS ins, VOID *v)
{
    ADDRINT ins_addr = INS_Address(ins);
    if (ins_addr < start_addr || ins_addr > end_addr){//if (ins_addr < 0x4213e0 || ins_addr > 0x42186f){//if (ins_addr < 0x422860 || ins_addr > 0x424b6a){
        return;
    }
    str_of_ins_at[INS_Address(ins)] = INS_Disassemble(ins);
    std::string ins_asm = INS_Disassemble(ins);
    /*
    if (!(ins_asm.find("xmm")!=ins_asm.npos || ins_asm.find("ymm")!=ins_asm.npos)){
        return;
    }
    */
    
    if (ins_addr == end_addr){
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)SetEndFlag,
            IARG_INST_PTR,
            IARG_END);
        return;
    }
    
    if (after_call_flag != 0){
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)AfterCall,
            IARG_INST_PTR,
            IARG_REG_VALUE, LEVEL_BASE::REG_RAX,
            IARG_END);
    }
    
    if (ins_asm.find("call")!=ins_asm.npos){
        after_call_flag = 1;
    }
    else{
        after_call_flag = 0;
    }
    
    // Instruments memory accesses using a predicated call, i.e.
    // the instrumentation is called iff the instruction will actually be executed.
    //
    // On the IA-32 and Intel(R) 64 architectures conditional moves and REP 
    // prefixed instructions appear as predicated instructions in Pin.
    UINT32 memOperands = INS_MemoryOperandCount(ins);
    
    if (INS_IsLea(ins)){
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordLea,
            IARG_INST_PTR,
            IARG_EXPLICIT_MEMORY_EA,
            IARG_END);
    }
    else if (memOperands == 0){
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordInst,
            IARG_INST_PTR,
            IARG_END);
        return;
    }

    // Iterate over each memory operand of the instruction.
    for (UINT32 memOp = 0; memOp < memOperands; memOp++)
    {
        if (INS_MemoryOperandIsRead(ins, memOp))
        {
            USIZE mem_size = INS_MemoryReadSize(ins);
            INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE, (AFUNPTR)RecordMemRead,
                IARG_INST_PTR,
                IARG_MEMORYOP_EA, memOp,
                IARG_UINT64, mem_size,
                IARG_END);
        }
        // Note that in some architectures a single memory operand can be 
        // both read and written (for instance incl (%eax) on IA-32)
        // In that case we instrument it once for read and once for write.
        if (INS_MemoryOperandIsWritten(ins, memOp))
        {
            USIZE mem_size = INS_MemoryWriteSize(ins);
            INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE, (AFUNPTR)RecordMemWrite,
                IARG_INST_PTR,
                IARG_MEMORYOP_EA, memOp,
                IARG_UINT64, mem_size,
                IARG_END);
        }
    }
}

VOID Fini(INT32 code, VOID *v)
{
    fprintf(trace, "#eof\n");
    fclose(trace);
}

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */
   
INT32 Usage()
{
    PIN_ERROR( "This Pintool prints a trace of memory addresses\n" 
              + KNOB_BASE::StringKnobSummary() + "\n");
    return -1;
}

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(int argc, char *argv[])
{
    if (PIN_Init(argc, argv)) return Usage();

    std::string fileName = KnobOutputFile.Value();
    trace = fopen(fileName.c_str(), "w");
    //trace = fopen("pinatrace.out", "w");
    start_addr = KnobStartAddr.Value();
    end_addr = KnobEndAddr.Value();

    // debug
    //printf("output: %s, start: %p, end: %p\n", fileName.c_str(), (void *)start_addr, (void *)end_addr);
    //return 0;

    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns
    PIN_StartProgram();
    
    return 0;
}
