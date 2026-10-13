#ifndef GENIO_SPI_FLASH_MAIN_H_
#define GENIO_SPI_FLASH_MAIN_H_

#include <stdint.h>

#include "genio_app_config.h"

/*
 * The on-chip linker uses matching SRAM and flash offsets. Flash-load builds
 * need a flash-only VMA so heep_get_flash_address_offset() can recover the
 * physical 24-bit SPI address without keeping a second SRAM copy.
 */
#if defined(ON_CHIP)
#define GENIO_FLASH_ONLY
#else
#define GENIO_FLASH_ONLY __attribute__((section(".xheep_data_flash_only")))
#endif

static const uint32_t image_input[GENIO_INPUT_WORDS > 0 ? GENIO_INPUT_WORDS : 1]
    GENIO_FLASH_ONLY __attribute__((aligned(16))) = {
#if GENIO_INPUT_WORDS > 0
    @IMAGE_WORDS@
#endif
};

#undef GENIO_FLASH_ONLY

/* SAFA output is staged in SRAM before the sector-preserving flash write. */
static uint32_t image_output[GENIO_OUTPUT_WORDS > 0 ? GENIO_OUTPUT_WORDS : 1]
    __attribute__((aligned(16)));

#endif
