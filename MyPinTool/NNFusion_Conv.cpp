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
#include <list>
#include "pin.H"

static std::unordered_map<ADDRINT, std::string> str_of_ins_at;

FILE * trace;

// just want a hash table
static std::unordered_map<uint64_t, uint64_t> addrs_list;
/* ===================================================================== */
// Command line switches
/* ===================================================================== */
KNOB<std::string> KnobOutputFile(KNOB_MODE_WRITEONCE,  "pintool",
    "o", "", "specify file name for MyPinTool output");

KNOB<std::string>   KnobAddrsFile(KNOB_MODE_WRITEONCE,  "pintool",
    "addrs_file", "0x422860", "file path");
    

/* ===================================================================== */
// Utilities
/* ===================================================================== */

VOID RecordInst(VOID * ip, ADDRINT rsp_value, ADDRINT rsi_value, ADDRINT rdx_value, ADDRINT rcx_value, ADDRINT r8_value, ADDRINT r9_value)
{
    fprintf(trace,"MlasConvPrepare: %p\n", ip);
    //then print parameters of MlasConvPrepare
    // rsi --> dimensions
    fprintf(trace, "dimensions: %ld\n", rsi_value);
    // rdx --> batch count
    fprintf(trace, "batch count: %ld\n", rdx_value);
    // rcx --> group count
    fprintf(trace, "group count: %ld\n", rcx_value);
    // r8 --> input channels
    fprintf(trace, "dimensions: %ld\n", r8_value);
    // r9 --> input shape
    //printf("debug r9: %p\n", (void *)r9_value);
    fprintf(trace, "input shape: %ld, %ld\n", *((uint64_t *)r9_value), *((uint64_t *)r9_value + 1));
    // rsp+8 --> kernel shape
    ADDRINT ptr = *((uint64_t *)(rsp_value + 8));
    //printf("debug ptr: %p\n", (void *)ptr);
    fprintf(trace, "kernel shape: %ld, %ld\n", *((uint64_t *)ptr), *((uint64_t *)ptr + 1));
    // rsp+0x10 --> dialation shape
    ptr = *((uint64_t *)(rsp_value + 0x10));
    //printf("debug ptr: %p\n", (void *)ptr);
    fprintf(trace, "dialation shape: %ld, %ld\n", *((uint64_t *)ptr), *((uint64_t *)ptr + 1));
    // rsp+0x18 --> padding
    ptr = *((uint64_t *)(rsp_value + 0x18));
    //printf("debug ptr: %p\n", (void *)ptr);
    fprintf(trace, "padding: %ld, %ld\n", *((uint64_t *)ptr), *((uint64_t *)ptr + 1));
    // rsp+0x20 --> stride shape
    ptr = *((uint64_t *)(rsp_value + 0x20));
    //printf("debug ptr: %p\n", (void *)ptr);
    fprintf(trace, "stride shape: %ld, %ld\n", *((uint64_t *)ptr), *((uint64_t *)ptr + 1));
    // rsp+0x28 --> output shape
    ptr = *((uint64_t *)(rsp_value + 0x28));
    //printf("debug ptr: %p\n", (void *)ptr);
    fprintf(trace, "output shape: %ld, %ld\n", *((uint64_t *)ptr), *((uint64_t *)ptr + 1));
    // rsp+0x30 --> filter count / output channels
    ptr = rsp_value + 0x30;
    fprintf(trace, "filter shape/output channels: %ld\n", *((uint64_t *)ptr));
    fprintf(trace, "\n");
}

// Print a memory read record
VOID RecordMemRead(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
    fprintf(trace,"%p\n", ip);
    //std::string ins_str = str_of_ins_at[(ADDRINT)ip];
    //fprintf(trace,"%p:\t%s\n", ip, ins_str.c_str());
    //fprintf(trace,"R:\t%p:\t%lu\n", mem_addr, mem_size);
    //fprintf(trace,"%p: R %p\n", ip, addr);
}

// Print a memory write record
VOID RecordMemWrite(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
    fprintf(trace,"%p\n", ip);
    //std::string ins_str = str_of_ins_at[(ADDRINT)ip];
    //fprintf(trace,"%p:\t%s\n", ip, ins_str.c_str());
    //fprintf(trace,"W:\t%p:\t%lu\n", mem_addr, mem_size);
    //fprintf(trace,"%p: W %p\n", ip, addr);
}

// Is called for every instruction and instruments reads and writes
VOID Instruction(INS ins, VOID *v)
{
    ADDRINT ins_addr = INS_Address(ins);
    
    if (addrs_list.find(ins_addr) == addrs_list.end()){
        return;
    }
    
    str_of_ins_at[INS_Address(ins)] = INS_Disassemble(ins);
    std::string ins_asm = INS_Disassemble(ins);
    /*
    if (!(ins_asm.find("xmm")!=ins_asm.npos || ins_asm.find("ymm")!=ins_asm.npos)){
        return;
    }
    */
    // Instruments memory accesses using a predicated call, i.e.
    // the instrumentation is called iff the instruction will actually be executed.
    //
    // On the IA-32 and Intel(R) 64 architectures conditional moves and REP 
    // prefixed instructions appear as predicated instructions in Pin.
    UINT32 memOperands = INS_MemoryOperandCount(ins);
    
    
    INS_InsertPredicatedCall(
        ins, IPOINT_BEFORE, (AFUNPTR)RecordInst,
        IARG_INST_PTR,
        IARG_REG_VALUE, LEVEL_BASE::REG_RSP,
        IARG_REG_VALUE, LEVEL_BASE::REG_RSI,
        IARG_REG_VALUE, LEVEL_BASE::REG_RDX,
        IARG_REG_VALUE, LEVEL_BASE::REG_RCX,
        IARG_REG_VALUE, LEVEL_BASE::REG_R8,
        IARG_REG_VALUE, LEVEL_BASE::REG_R9,
        IARG_END);
    return;

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

int ReadAddrList(){
    std::string addrs_file = KnobAddrsFile.Value();
    FILE *fp = fopen(addrs_file.c_str(),"r");
    //int count = 0;
    while(!feof(fp)){
        uint64_t current_addr;
        fscanf(fp, "%lx\n", &current_addr);
        addrs_list[current_addr] = current_addr;
        //printf("insert 0x%lx\n", current_addr); // debug
        //count += 1;
        //printf("%d\n", count);
    }
    return 0;
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
    
    ReadAddrList();    
    
    // debug
    //printf("output: %s, start: %p, end: %p\n", fileName.c_str(), (void *)start_addr, (void *)end_addr);
    
    /*
    std::unordered_map<uint64_t, uint64_t>::iterator iter;
    iter = addrs_list.begin();
    int count = 0;
    while(iter != addrs_list.end()) {
        printf("0x%lx\n", iter->second);
        iter++;
        count += 1;
        printf("%d\n", count);
    }
    return 0;
    */
    
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns
    PIN_StartProgram();
    
    return 0;
}
