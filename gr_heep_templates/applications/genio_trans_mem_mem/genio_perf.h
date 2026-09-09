#ifndef GENIO_PERF_H_
#define GENIO_PERF_H_

#include <stdint.h>
#include <stdio.h>

static inline void genio_perf_init(void) {
    __asm__ volatile("csrci mcountinhibit, 1");
}

static inline uint64_t genio_read_cycles(void) {
#if __riscv_xlen == 32
    uint32_t high_before;
    uint32_t low;
    uint32_t high_after;
    do {
        __asm__ volatile("csrr %0, mcycleh" : "=r"(high_before));
        __asm__ volatile("csrr %0, mcycle" : "=r"(low));
        __asm__ volatile("csrr %0, mcycleh" : "=r"(high_after));
    } while (high_before != high_after);
    return ((uint64_t)high_after << 32) | low;
#else
    uint64_t cycles;
    __asm__ volatile("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
#endif
}

#define GENIO_PERF_BEGIN(name) \
    const uint64_t genio_perf_start_##name = genio_read_cycles()

#define GENIO_PERF_END(name)                                               \
    do {                                                                   \
        const uint64_t genio_perf_end = genio_read_cycles();               \
        printf(                                                            \
            "GENIO_PERF:%s:%lu\n",                                       \
            #name,                                                         \
            (unsigned long)(genio_perf_end - genio_perf_start_##name)      \
        );                                                                 \
    } while (0)

#endif
