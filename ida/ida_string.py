import idautils
import idaapi

idaapi.auto_wait()
sc = idautils.Strings()

with open('strings.txt', 'w') as f:
    for s in sc:
        print("%x: len=%d type=%d -> '%s'" % (s.ea, s.length, s.strtype, str(s)))
        f.write(str(s)+'\n')
    f.close()

idaapi.qexit(0)