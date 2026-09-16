"""ELF64 analysis: DT_NEEDED deps, undefined symbols (imports), defined exports."""
import struct
import sys


class Elf:
    def __init__(self, path):
        self.data = open(path, "rb").read()
        assert self.data[:4] == b"\x7fELF" and self.data[4] == 2, "ELF64 LE expected"
        self.dynamic = self.parse_dynamic()
        self.strtab = self.load_vaddr(self.dynamic.get(5, [0])[0])
        self.needed = [self.str_at(v) for v in self.dynamic.get(1, [])]

    def str_at(self, off):
        end = self.strtab.find(b"\x00", off)
        return self.strtab[off:end].decode("utf-8", "replace")

    def parse_dynamic(self):
        # locate PT_DYNAMIC
        e_phoff = struct.unpack_from("<Q", self.data, 0x20)[0]
        e_phentsize = struct.unpack_from("<H", self.data, 0x36)[0]
        e_phnum = struct.unpack_from("<H", self.data, 0x38)[0]
        dyn_va = None
        self.loads = []
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            p_type = struct.unpack_from("<I", self.data, off)[0]
            if p_type == 1:  # PT_LOAD
                p_offset = struct.unpack_from("<Q", self.data, off + 8)[0]
                p_vaddr = struct.unpack_from("<Q", self.data, off + 0x10)[0]
                p_filesz = struct.unpack_from("<Q", self.data, off + 0x20)[0]
                self.loads.append((p_offset, p_vaddr, p_filesz))
            if p_type == 2:  # PT_DYNAMIC
                dyn_va = struct.unpack_from("<Q", self.data, off + 0x10)[0]
        if dyn_va is None:
            return {}
        file_off = self.va_to_off(dyn_va)
        out = {}
        while True:
            tag = struct.unpack_from("<q", self.data, file_off)[0]
            val = struct.unpack_from("<Q", self.data, file_off + 8)[0]
            if tag == 0:
                break
            out.setdefault(tag, []).append(val)
            file_off += 16
        return out

    def va_to_off(self, va):
        for p_offset, p_vaddr, p_filesz in self.loads:
            if p_vaddr <= va < p_vaddr + p_filesz:
                return p_offset + (va - p_vaddr)
        raise ValueError("va not in any LOAD segment")

    def load_vaddr(self, va):
        if va == 0:
            return b""
        off = self.va_to_off(va)
        return self.data[off:]

    def symtabs(self):
        # .dynsym section
        e_shoff = struct.unpack_from("<Q", self.data, 0x28)[0]
        e_shentsize = struct.unpack_from("<H", self.data, 0x3A)[0]
        e_shnum = struct.unpack_from("<H", self.data, 0x3C)[0]
        e_shstrndx = struct.unpack_from("<H", self.data, 0x3E)[0]
        secs = []
        for i in range(e_shnum):
            off = e_shoff + i * e_shentsize
            secs.append(
                (
                    struct.unpack_from("<I", self.data, off)[0],
                    struct.unpack_from("<I", self.data, off + 4)[0],
                    struct.unpack_from("<Q", self.data, off + 0x18)[0],
                    struct.unpack_from("<Q", self.data, off + 0x20)[0],
                )
            )
        shstr = secs[e_shstrndx]
        shstrtab = self.data[shstr[2]:shstr[2] + shstr[3]]

        def sname(idx):
            end = shstrtab.find(b"\x00", idx)
            return shstrtab[idx:end].decode()

        dynsym = dynstr = None
        for sec in secs:
            nm = sname(sec[0])
            if nm == ".dynsym":
                dynsym = sec
            elif nm == ".dynstr":
                dynstr = sec
        if not dynsym or not dynstr:
            return [], []
        strtab = self.data[dynstr[2]:dynstr[2] + dynstr[3]]
        imports, exports = [], []
        base = dynsym[2]
        for off in range(0, dynsym[3], 24):
            st_name = struct.unpack_from("<I", self.data, base + off)[0]
            st_info = self.data[base + off + 4]
            st_shndx = struct.unpack_from("<H", self.data, base + off + 6)[0]
            end = strtab.find(b"\x00", st_name)
            name = strtab[st_name:end].decode("utf-8", "replace")
            if not name:
                continue
            if st_shndx == 0:
                imports.append(name)
            elif st_info >> 4 == 1:
                exports.append(name)
        return imports, exports


def inspect(path):
    e = Elf(path)
    print("NEEDED:", e.needed)
    if 14 in e.dynamic:
        print("SONAME:", e.str_at(e.dynamic[14][0]))
    imports, exports = e.symtabs()
    ort_imp = [s for s in imports if "Ort" in s or "onnxruntime" in s]
    ort_exp = [s for s in exports if "Ort" in s]
    print("imports matching Ort/onnxruntime:", len(ort_imp))
    for s in ort_imp[:30]:
        print("   I", s)
    print("exports matching Ort:", len(ort_exp))
    for s in ort_exp[:30]:
        print("   E", s)
    print("total imports:", len(imports), "total exports:", len(exports))


if __name__ == "__main__":
    inspect(sys.argv[1])
