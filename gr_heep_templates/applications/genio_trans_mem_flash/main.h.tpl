#ifndef GENIO_TRANS_MEM_FLASH_MAIN_H_
#define GENIO_TRANS_MEM_FLASH_MAIN_H_

#include <stdint.h>

#include "genio_app_config.h"

/* The configurable input starts in SRAM and is streamed into SAFA by DMA. */
static const uint32_t image_input[GENIO_INPUT_WORDS > 0 ? GENIO_INPUT_WORDS : 1]
    __attribute__((aligned(16))) = {
#if GENIO_INPUT_WORDS > 0
    @IMAGE_WORDS@
#endif
};

/* SAFA output is staged in SRAM before the sector-preserving SPI write. */
static uint32_t image_output[GENIO_OUTPUT_WORDS > 0 ? GENIO_OUTPUT_WORDS : 1]
    __attribute__((aligned(16)));

#endif
