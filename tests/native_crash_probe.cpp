/*
*   Muna
*   Copyright © 2026 NatML Inc. All rights reserved.
*/

#include <cstdint>

extern "C" __attribute__((noinline)) void native_crash_probe() {
    volatile uintptr_t address = 0;
    *reinterpret_cast<volatile uint32_t*>(address) = 42;
}
