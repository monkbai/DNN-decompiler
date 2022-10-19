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


void * stack_border = (void *)0x7f0000000000;

uint64_t dump_addr = 0;
uint64_t length = 0;
uint64_t dump_point = 0;

/* ===================================================================== */
// Command line switches
/* ===================================================================== */
KNOB<std::string> KnobOutputFile(KNOB_MODE_WRITEONCE,  "pintool",
    "o", "tmp.log", "specify file name for MyPinTool output");

KNOB<uint64_t>   KnobDumpAddr(KNOB_MODE_WRITEONCE,  "pintool",
    "dump_addr", "", "no description");
    
KNOB<uint64_t>   KnobDumpLen(KNOB_MODE_WRITEONCE,  "pintool",
    "length", "", "no description");

KNOB<uint64_t>   KnobDumpPoint(KNOB_MODE_WRITEONCE,  "pintool",
    "dump_point", "", "no description");


/* ===================================================================== */
// Utilities
/* ===================================================================== */

VOID PrintDword(const int * addr, int length){
    int i = 0;
    fprintf(trace, "Start from 0x%.16lx\n", (uint64_t)addr);
    for (; i < length; i++){
        fprintf(trace, "0x%.8x\n", *(addr+i));
    }
    fprintf(trace, "end\n");
}

VOID Dump(VOID * ip){
	int length = 1;
    PrintDword((const int *)dump_addr, (int)length);
    
}

VOID RecordInst(VOID * ip)
{
}

// Print a memory read record
VOID RecordMemRead(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
}

// Print a memory write record
VOID RecordMemWrite(VOID * ip, VOID * mem_addr, USIZE mem_size)
{
}

// Is called for every instruction and instruments reads and writes
VOID Instruction(INS ins, VOID *v)
{
    ADDRINT ins_addr = INS_Address(ins);
    
    //if (ins_addr < start_addr || ins_addr > end_addr){//if (ins_addr < 0x4213e0 || ins_addr > 0x42186f){//if (ins_addr < 0x422860 || ins_addr > 0x424b6a){
    //    return;
    //}
    str_of_ins_at[INS_Address(ins)] = INS_Disassemble(ins);
    std::string ins_asm = INS_Disassemble(ins);
    
    if (ins_addr == dump_point){
        printf("instrument at %p\n", (VOID *)ins_addr);
        INS_InsertPredicatedCall(
            ins, IPOINT_BEFORE, (AFUNPTR)Dump,
            IARG_INST_PTR,
            IARG_END);
        return;
    }
    return;
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
    trace = fopen(fileName.c_str(), "a");
    
    dump_addr = KnobDumpAddr.Value();
    length = KnobDumpLen.Value();
    dump_point = KnobDumpPoint.Value();

    // debug
    printf("output: %s, dump_addr: %p, dump_length: %lu, dump_point: %p\n", fileName.c_str(), (void *)dump_addr, length, (void *)dump_point);
    //return 0;

    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);

    // Never returns
    PIN_StartProgram();
    
    return 0;
}
