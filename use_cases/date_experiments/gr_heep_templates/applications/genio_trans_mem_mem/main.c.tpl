#include <stdint.h>
#include <stdio.h>

#include "dma.h"
#include "mmio.h"

#include "genio_app_config.h"
#include "genio_perf.h"
#include "gr_heep.h"
#include "main.h"
#include "safa.h"

static dma_trans_t accelerator_transaction;
static dma_target_t accelerator_source;
static dma_target_t accelerator_destination;
static safa_t safa;

static int configure_safa(void) {
    if (GENIO_INPUT_WORDS == 0u || GENIO_OUTPUT_WORDS == 0u) {
        printf("Invalid image dimensions\n");
        return -1;
    }

    safa_result_t result = safa_init(
        &safa,
        mmio_region_from_addr((uintptr_t)SAFA_PERIPH_START_ADDRESS)
    );
    if (result != SAFA_RESULT_OK) {
        printf("SAFA init failed: %d\n", (int)result);
        return -1;
    }

    const safa_config_t config = {
        .input_words = GENIO_INPUT_WORDS,
        .output_words = GENIO_OUTPUT_WORDS,
        .auto_start = true,
        .irq_enable_mask = 0u,
    };
    result = safa_configure(&safa, &config);
    if (result != SAFA_RESULT_OK) {
        printf("SAFA configuration failed: %d\n", (int)result);
        return -1;
    }
    return 0;
}

static int configure_accelerator_dma(void) {
    accelerator_source.ptr = (uint8_t *)image_input;
    accelerator_source.inc_d1_du = 1;
    accelerator_source.trig = DMA_TRIG_MEMORY;
    accelerator_source.type = DMA_DATA_TYPE_WORD;

    accelerator_destination.ptr = (uint8_t *)image_output;
    accelerator_destination.inc_d1_du = 1;
    accelerator_destination.trig = DMA_TRIG_MEMORY;
    accelerator_destination.type = DMA_DATA_TYPE_WORD;

    accelerator_transaction.src = &accelerator_source;
    accelerator_transaction.dst = &accelerator_destination;
    accelerator_transaction.mode = DMA_TRANS_MODE_SINGLE;
    accelerator_transaction.hw_fifo_en = 1;
    accelerator_transaction.channel = GENIO_DMA_CHANNEL;
    accelerator_transaction.dim = DMA_DIM_CONF_1D;
    accelerator_transaction.size_d1_du = GENIO_INPUT_WORDS;
    accelerator_transaction.end = DMA_TRANS_END_POLLING;

    if (dma_validate_transaction(
            &accelerator_transaction,
            DMA_ENABLE_REALIGN,
            DMA_PERFORM_CHECKS_INTEGRITY
        ) != DMA_CONFIG_OK) {
        printf("Accelerator DMA validation failed\n");
        return -1;
    }

    return 0;
}

static int wait_for_accelerator(void) {
    const uint64_t start = genio_read_cycles();

    while (!dma_is_ready(GENIO_DMA_CHANNEL)) {
        safa_status_t status;
        if (safa_get_status(&safa, &status) != SAFA_RESULT_OK) {
            return -1;
        }
        if (status.error || status.aborted) {
            printf("SAFA stopped, errors=0x%08lx\n",
                   (unsigned long)safa_get_errors(&safa));
            (void)safa_abort(&safa);
            return -1;
        }

        if (GENIO_TIMEOUT_CYCLES != 0u &&
            genio_read_cycles() - start > (uint64_t)GENIO_TIMEOUT_CYCLES) {
            printf("SAFA timeout\n");
            (void)safa_abort(&safa);
            return -1;
        }
    }

    safa_result_t result = safa_wait_done(&safa, SAFA_WAIT_FOREVER);
    if (result != SAFA_RESULT_OK) {
        printf("SAFA completion failed: %d, errors=0x%08lx\n",
               (int)result,
               (unsigned long)safa_get_errors(&safa));
        return -1;
    }
    return 0;
}

static int output_is_complete(void) {
    safa_counters_t counters;
    if (safa_get_counters(&safa, &counters) != SAFA_RESULT_OK) {
        return 0;
    }
    return counters.input_accepted == GENIO_INPUT_WORDS &&
           counters.input_consumed == GENIO_INPUT_WORDS &&
           counters.output_generated == GENIO_OUTPUT_WORDS &&
           counters.output_popped == GENIO_OUTPUT_WORDS;
}

static void print_metrics(void) {
    safa_counters_t counters;
    if (safa_get_counters(&safa, &counters) != SAFA_RESULT_OK) {
        return;
    }

    printf("GENIO_METRIC:safa_active_cycles:%lu\n", (unsigned long)counters.active_cycles);
    printf("GENIO_METRIC:safa_input_fifo_empty_cycles:%lu\n", (unsigned long)counters.input_fifo_empty_cycles);
    printf("GENIO_METRIC:safa_input_fifo_full_cycles:%lu\n", (unsigned long)counters.input_fifo_full_cycles);
    printf("GENIO_METRIC:safa_output_fifo_empty_cycles:%lu\n", (unsigned long)counters.output_fifo_empty_cycles);
    printf("GENIO_METRIC:safa_output_fifo_full_cycles:%lu\n", (unsigned long)counters.output_fifo_full_cycles);
    printf("GENIO_METRIC:safa_input_stall_cycles:%lu\n", (unsigned long)counters.input_stall_cycles);
    printf("GENIO_METRIC:safa_output_stall_cycles:%lu\n", (unsigned long)counters.output_stall_cycles);
    printf("GENIO_METRIC:safa_dma_push_stall_cycles:%lu\n", (unsigned long)counters.dma_push_stall_cycles);
    printf("GENIO_METRIC:safa_dma_pop_stall_cycles:%lu\n", (unsigned long)counters.dma_pop_stall_cycles);
    printf("GENIO_METRIC:safa_input_words:%lu\n", (unsigned long)counters.input_accepted);
    printf("GENIO_METRIC:safa_output_words:%lu\n", (unsigned long)counters.output_popped);
}

int main(void) {
    int status = 1;

    genio_perf_init();
    dma_init(NULL);

    if (configure_safa() != 0 ||
        configure_accelerator_dma() != 0) {
        printf("GENIO_STATUS:%d\n", status);
        return status;
    }

    dma_load_transaction(&accelerator_transaction);

    GENIO_PERF_BEGIN(application);
    dma_launch(&accelerator_transaction);
    if (wait_for_accelerator() == 0 && output_is_complete()) {
        status = 0;
    }
    GENIO_PERF_END(application);

    print_metrics();
    (void)safa_clear_done(&safa);
    printf("GENIO_STATUS:%d\n", status);
    return status;
}
