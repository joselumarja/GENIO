#ifndef MAIN_H_
#define MAIN_H_

#include <stdint.h>

#include "genio_app_config.h"

static const uint32_t image_input[GENIO_INPUT_WORDS > 0 ? GENIO_INPUT_WORDS : 1]
    __attribute__((aligned(16))) = {
#if GENIO_INPUT_WORDS > 0
    @IMAGE_WORDS@
#endif
};

static uint32_t image_output[GENIO_OUTPUT_WORDS > 0 ? GENIO_OUTPUT_WORDS : 1]
    __attribute__((aligned(16)));

#endif
