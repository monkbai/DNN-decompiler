#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iostream>
#include <fstream>
#include "pin.H"
/*
struct mem_obj{
    uint64_t start;
    uint64_t end;
};
*/

static std::tr1::unordered_set<VOID *> visable_addrs;

static std::unordered_map<ADDRINT, std::string> str_of_ins_at;
FILE * trace;

void * max_write_mem = 0;
uint64_t size_max = 0;
void * min_write_mem = (void *)0x7f0000000000;
uint64_t size_min = 0;

void * stack_border = (void *)0x7f0000000000;

uint64_t start_addr = 0;
uint64_t end_addr = 0;

uint64_t early_stop = 0;

int end_flag = 0; // when end_flag > 0, stop logging

/* ===================================================================== */
// Command line switches
/* ===================================================================== */
KNOB<std::string> KnobOutputFile(KNOB_MODE_WRITEONCE,  "pintool",
    "o", "tmp.log", "specify file name for MyPinTool output");

KNOB<uint64_t>   KnobStartAddr(KNOB_MODE_WRITEONCE,  "pintool",
    "start", "0x422860", "specify file start address of instrumentation");
    
KNOB<uint64_t>   KnobEndAddr(KNOB_MODE_WRITEONCE,  "pintool",
    "end", "0x424b6a", "count instructions, basic blocks and threads in the application");

KNOB<uint64_t>   KnobEarlyStop(KNOB_MODE_WRITEONCE,  "pintool",
    "early_stop", "0", "");

/* ===================================================================== */
// Utilities
/* ===================================================================== */

VOID SetEndFlag(VOID * ip){
    end_flag = 1;
    printf("stop at %p\n", ip);
}

VOID RecordInst(VOID * ip)
{
    // do nothing
    //std::string ins_str = str_of_ins_at[(ADDRINT)ip];
    //fprintf(trace,"%p:\t%s\n", ip, ins_str.c_str());
    //fprintf(trace,"N:\t%p:\t%d\n", (void *)0xDEADBEEF, 7); // not real memory op
}

// Print a memory read record
VOID RecordMemRead(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
    if (end_flag != 0){
        return;
    }
    // do nothing
    //std::string ins_str = str_of_ins_at[(ADDRINT)ip];
    //fprintf(trace,"%p:\t%s\n", ip, ins_str.c_str());
    //fprintf(trace,"R:\t%p:\t%lu\n", mem_addr, mem_size);
    
}

// Print a memory write record
VOID RecordMemWrite(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
    if (end_flag != 0){
        return;
    }
    // do nothing
    //std::string ins_str = str_of_ins_at[(ADDRINT)ip];
    //fprintf(trace,"%p:\t%s\n", ip, ins_str.c_str());
    //fprintf(trace,"W:\t%p:\t%lu\n", mem_addr, mem_size);
    if (mem_addr < stack_border){
        //std::string mem_str = std::to_string(mem_addr);
        std::tr1::unordered_set<VOID *>::iterator iter = visable_addrs.find(mem_addr);
        if (iter != visable_addrs.end()) {
            return;
        }
        //fprintf(trace, "%p:%p,%lu\n", ip, mem_addr, mem_size);
        fprintf(trace, "%p,%lu\n", mem_addr, mem_size);
        visable_addrs.insert(mem_addr);
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
    
    if (ins_addr == end_addr or ins_addr == early_stop){
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)SetEndFlag,
            IARG_INST_PTR,
            IARG_END);
        return;
    }
    
    // Instruments memory accesses using a predicated call, i.e.
    // the instrumentation is called iff the instruction will actually be executed.
    //
    // On the IA-32 and Intel(R) 64 architectures conditional moves and REP 
    // prefixed instructions appear as predicated instructions in Pin.
    UINT32 memOperands = INS_MemoryOperandCount(ins);
    
    if (memOperands == 0){
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)RecordInst,
            IARG_INST_PTR,
            IARG_END);
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
    //printf("min_write_mem: %p, %lu", min_write_mem, size_min);
    //printf("max_write_mem: %p, %lu", max_write_mem, size_max);
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
    early_stop = KnobEarlyStop.Value();

    // debug
    printf("output: %s, start: %p, end: %p, early_stop: %p\n", fileName.c_str(), (void *)start_addr, (void *)end_addr, (void *)early_stop);
    //return 0;

    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns
    PIN_StartProgram();
    
    return 0;
}
