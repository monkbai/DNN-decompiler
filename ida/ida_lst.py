import idaapi
import idautils
import idc

idaapi.auto_wait()
idc.gen_file(idc.OFILE_LST , './tmp.lst', 0, idc.BADADDR, 0)
idaapi.qexit(0)

# import ida_pro 
# ida_pro.qexit()
