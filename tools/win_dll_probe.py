"""Explain a WinError 1114 from the native runtime.

Windows reports "A dynamic link library (DLL) initialization routine failed"
whenever anything inside DllMain raises, and the loader swallows the exception
itself, so `flyweight doctor` can only repeat the number. This installs a
vectored exception handler ahead of the loader's own, loads the library, and
prints the exception code and the module plus offset it came from:

    0xC000001D  illegal instruction -- code compiled for an instruction set
                this CPU lacks ran during initialization
    0xC0000005  access violation
    0xE06D7363  a C++ exception thrown by a static initializer

Then it loads the image with DONT_RESOLVE_DLL_REFERENCES, which maps the file
without running any initializer, to show whether the file itself is sound.

When the MSVC toolchain the build uses is installed, it finishes by
disassembling the code around the faulting address with dumpbin, so the
instruction that faulted -- and therefore the instruction set it needs -- is
visible without a debugger.

Usage, from any prompt with the venv active:

    python tools\\win_dll_probe.py
    python tools\\win_dll_probe.py C:\\path\\to\\flyweight_v2.dll
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

DONT_RESOLVE_DLL_REFERENCES = 0x1
GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS = 0x4
GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT = 0x2
EXCEPTION_CONTINUE_SEARCH = 0
# Debugger and thread-naming notifications that are not failures.
IGNORED_CODES = {0x40010006, 0x4001000A, 0x406D1388}
NAMES = {
    0xC000001D: "illegal instruction (CPU lacks an instruction set the code was built for)",
    0xC0000005: "access violation",
    0xC0000096: "privileged instruction",
    0xC00000FD: "stack overflow",
    0xE06D7363: "C++ exception thrown during initialization",
}


class ExceptionRecord(ctypes.Structure):
    _fields_ = [
        ("ExceptionCode", wintypes.DWORD),
        ("ExceptionFlags", wintypes.DWORD),
        ("ExceptionRecord", ctypes.c_void_p),
        ("ExceptionAddress", ctypes.c_void_p),
        ("NumberParameters", wintypes.DWORD),
        ("ExceptionInformation", ctypes.c_size_t * 15),
    ]


class ExceptionPointers(ctypes.Structure):
    _fields_ = [
        ("ExceptionRecord", ctypes.POINTER(ExceptionRecord)),
        ("ContextRecord", ctypes.c_void_p),
    ]


def main() -> int:
    if sys.platform != "win32":
        print("this probe is for Windows")
        return 2
    if len(sys.argv) > 1:
        library = Path(sys.argv[1])
    else:
        import flyweight  # noqa: PLC0415 - locate the installed copy

        library = Path(flyweight.__file__).with_name("_native") / "flyweight_v2.dll"
    print(f"library: {library}")
    print(f"python:  {platform.python_version()} {platform.architecture()[0]}")
    print(f"cpu:     {platform.processor()}")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LoadLibraryExW.restype = ctypes.c_void_p
    kernel32.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
    kernel32.GetModuleHandleExW.restype = wintypes.BOOL
    kernel32.GetModuleHandleExW.argtypes = [
        wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    kernel32.GetModuleFileNameW.argtypes = [
        ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.AddVectoredExceptionHandler.restype = ctypes.c_void_p

    handler_type = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.POINTER(ExceptionPointers))
    # RVA of each fault inside the library, taken while it is still mapped:
    # the base can differ between loads, so the absolute address is useless
    # once the loader has unmapped the failed image.
    faults: list[int] = []

    def handler(pointers: ctypes.POINTER(ExceptionPointers)) -> int:  # type: ignore[valid-type]
        record = pointers.contents.ExceptionRecord.contents
        code = record.ExceptionCode
        if code in IGNORED_CODES:
            return EXCEPTION_CONTINUE_SEARCH
        address = record.ExceptionAddress or 0
        module = ctypes.c_void_p()
        where = "unknown module"
        if kernel32.GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
            | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            ctypes.c_void_p(address), ctypes.byref(module),
        ):
            buffer = ctypes.create_unicode_buffer(1024)
            kernel32.GetModuleFileNameW(module, buffer, 1024)
            where = f"{buffer.value}+0x{address - (module.value or 0):X}"
            if Path(buffer.value).resolve() == library.resolve():
                faults.append(address - (module.value or 0))
        meaning = NAMES.get(code, "")
        print(f"exception 0x{code:08X} {meaning}\n  at 0x{address:X} = {where}", flush=True)
        return EXCEPTION_CONTINUE_SEARCH

    keep_alive = handler_type(handler)
    kernel32.AddVectoredExceptionHandler(1, keep_alive)

    handle = kernel32.LoadLibraryExW(str(library), None, 0)
    error = ctypes.get_last_error()
    if handle:
        print("full load: ok (initializers ran)")
        kernel32.FreeLibrary(handle)
    else:
        print(f"full load: failed, WinError {error}")

    mapped = kernel32.LoadLibraryExW(str(library), None, DONT_RESOLVE_DLL_REFERENCES)
    if mapped:
        print("map only:  ok (the file is a valid x64 image; the failure is in initialization)")
        kernel32.FreeLibrary(mapped)
    else:
        print(f"map only:  failed, WinError {ctypes.get_last_error()}")
    if faults:
        disassemble_around(library, faults[0])
    return 0 if handle else 1


def disassemble_around(library: Path, rva: int) -> None:
    """Print the instructions around `rva` using the build's own dumpbin."""
    print(f"\nfaulting RVA: 0x{rva:X}")
    try:
        from flyweight.native_build import _build_environment  # noqa: PLC0415

        environment = _build_environment()
    except Exception as error:  # noqa: BLE001 - diagnostics only
        print(f"(no MSVC toolchain for dumpbin: {error})")
        return
    # CreateProcess resolves a bare name against this process's PATH, not the
    # child's, so the vcvars PATH has to be searched by hand.
    dumpbin = shutil.which("dumpbin", path=environment.get("PATH"))
    if dumpbin is None:
        print("(dumpbin not found on the MSVC toolchain PATH)")
        return
    try:
        headers = subprocess.run(
            [dumpbin, "/nologo", "/headers", str(library)], env=environment,
            capture_output=True, text=True, check=True, errors="replace").stdout
        listing = subprocess.run(
            [dumpbin, "/nologo", "/disasm:nobytes", str(library)], env=environment,
            capture_output=True, text=True, check=True, errors="replace").stdout
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"(dumpbin unavailable: {error})")
        return
    match = re.search(r"([0-9A-F]+) image base", headers)
    image_base = int(match.group(1), 16) if match else 0x180000000
    target = image_base + rva
    lines = listing.splitlines()
    hit = None
    for index, line in enumerate(lines):
        head = line.strip().split(":", 1)[0]
        if len(head) == 16 and all(c in "0123456789ABCDEF" for c in head):
            if int(head, 16) >= target:
                hit = index
                break
    if hit is None:
        print("(address not found in the disassembly)")
        return
    # The nearest preceding label, if the DLL exports it, names the function.
    label = next((lines[i] for i in range(hit, -1, -1)
                  if lines[i] and not lines[i][0].isspace() and lines[i].endswith(":")),
                 None)
    if label:
        print(f"in {label}")
    print("instructions before and at the fault (the last one raised):")
    for line in lines[max(0, hit - 24):hit + 1]:
        print("  " + line.rstrip())
    faulting = lines[hit].split(None, 1)[-1].strip().lower()
    if faulting.startswith(("ud2", "int")):
        print("=> a deliberate trap, not an instruction-set problem")
    elif ("zmm" in faulting or re.search(r"\bk[0-7]\b", faulting) or "{" in faulting
          or re.match(r"v(movdq[ua](8|16|32|64)|pternlog|p(and|or|xor|andn)[dq]|"
                      r"broadcast[if]32x|extract[if](32|64)x|insert[if](32|64)x)", faulting)):
        print("=> an AVX-512 instruction ran during DLL initialization")
    elif faulting.startswith("v"):
        print("=> a VEX-encoded (AVX/AVX2/FMA) instruction ran during DLL initialization;"
              " if the operands are xmm/ymm it may still be an EVEX (AVX-512VL) form")


if __name__ == "__main__":
    sys.exit(main())
