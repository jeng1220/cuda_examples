#pragma once
void add1_hub(void* dst, void* src0, void* src1, size_t count);
void add2_hub(void* dst, void* src0, void* src1, size_t count);
void add8_hub(void* dst, void* src0, void* src1, size_t count);
void add_thrust(void* dst, void* src0, void* src1, size_t count);
void add_perf(size_t count, int itr);