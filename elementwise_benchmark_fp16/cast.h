#pragma once
void cast1_hub(void* dst, void* src0, void* src1, size_t count);
void cast4_hub(void* dst, void* src0, void* src1, size_t count);
void cast8_hub(void* dst, void* src0, void* src1, size_t count);
void cast_thrust(void* dst, void* src0, void* src1, size_t count);
void cast_perf(size_t count, int itr);