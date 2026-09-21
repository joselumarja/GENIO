#include <stdint.h>
#include <stdio.h>

#include "dma.h"
#include "mmio.h"

#include "genio_app_config.h"
#include "genio_perf.h"
#include "gr_heep.h"
#include "main.h"
#include "safa.h"
#include "traffic_generator.h"

#define TRAFFIC_WORDS 8192u
#define TRAFFIC_BYTES (TRAFFIC_WORDS * sizeof(uint32_t))
#define TRAFFIC_MASK  (TRAFFIC_BYTES - sizeof(uint32_t))

static dma_trans_t accelerator_transaction;
static dma_target_t accelerator_source;
static dma_target_t accelerator_destination;
static safa_t safa;
static traffic_generator_t traffic_generator;

static volatile uint32_t traffic_buffer[TRAFFIC_WORDS] __attribute__((aligned(TRAFFIC_BYTES)));
//uint32_t ram_start = xheep_memory_regions[0].start;
//uint32_t ram_end = xheep_memory_regions[MEMORY_BANKS - 1u].end;

static traffic_generator_config_t base_config(void)
{
    traffic_generator_config_t config = {
        .base_address = (uintptr_t)traffic_buffer,
        .address_mask = TRAFFIC_MASK, /* 0xFC */
        //.base_address = ram_start;
        //.address_mask = (ram_end - ram_start) - sizeof(uint32_t);
        .stride = sizeof(uint32_t),
        .seed = 0x12345678u,
        .write_data = 0xA5A5A5A5u,
        .byte_enable = 0xFu,

        .injection_rate = UINT32_MAX,
        .period = 1u,
        .burst_length = 1u,
        .idle_length = 0u,

        .duration_limit = 0u,
        .duration_mode = TRAFFIC_GENERATOR_DURATION_INFINITE,

        .address_mode = TRAFFIC_GENERATOR_ADDRESS_FIXED,
        .temporal_mode = TRAFFIC_GENERATOR_TEMPORAL_SATURATED,
        .rw_mode = TRAFFIC_GENERATOR_RW_READ,
        .write_data_mode = TRAFFIC_GENERATOR_WRITE_DATA_FIXED,

        .irq_enable_mask = TRAFFIC_GENERATOR_IRQ_ALL_MASK,
    };

    return config;
}

static int traffic_generator_initialize(void)
{
    return traffic_generator_init(
        &traffic_generator,
        mmio_region_from_addr(
            OBI_TRAFFIC_GENERATOR_PERIPH_START_ADDRESS))
        != TRAFFIC_GENERATOR_RESULT_OK;
}

static int start_irregular_traffic(void)
{
    traffic_generator_config_t config = base_config();

    //config.address_mode = TRAFFIC_GENERATOR_ADDRESS_GAUSSIAN;
    config.address_mode = TRAFFIC_GENERATOR_ADDRESS_UNIFORM;
    config.seed = 0xC001D00Du;

    config.temporal_mode = TRAFFIC_GENERATOR_TEMPORAL_BERNOULLI;

    //config.injection_rate = 0x40000000u; /* Aproximadamente 25 %. */
    config.injection_rate = 0x80000000u; /* Aproximadamente 50 %. */
    //config.injection_rate = 0xC0000000u; /* Aproximadamente 75 %. */
    //config.injection_rate = UINT32_MAX;  /* Siempre que el bus lo permita. */

    config.rw_mode = TRAFFIC_GENERATOR_RW_READ;

    config.duration_mode = TRAFFIC_GENERATOR_DURATION_INFINITE;
    config.duration_limit = 0u;

    traffic_generator_result_t result =
        traffic_generator_configure(&traffic_generator, &config);

    if (result != TRAFFIC_GENERATOR_RESULT_OK) {
        return 1;
    }

    result = traffic_generator_start(&traffic_generator);
    return result != TRAFFIC_GENERATOR_RESULT_OK;
}

static int stop_traffic(void)
{
    traffic_generator_result_t result =
        traffic_generator_stop(&traffic_generator);

    if (result != TRAFFIC_GENERATOR_RESULT_OK) {
        return 1;
    }

    result = traffic_generator_wait_done(
        &traffic_generator,
        TRAFFIC_GENERATOR_WAIT_FOREVER);

    if (result != TRAFFIC_GENERATOR_RESULT_OK) {
        return 1;
    }

    traffic_generator_counters_t counters;
    traffic_generator_get_counters(&traffic_generator, &counters);

    printf("cycles=%lu requests=%lu completed=%lu reads=%lu writes=%lu\n",
           (unsigned long)counters.elapsed_cycles,
           (unsigned long)counters.requests,
           (unsigned long)counters.completed,
           (unsigned long)counters.reads,
           (unsigned long)counters.writes);

    return 0;
}

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

    if (traffic_generator_initialize() != 0) {
        printf("Traffic generator configuration error\n");
        return 1;
    }

    if (start_irregular_traffic() != 0) {
        printf("Traffic generator initialization error\n");
        return 1;
    }

    dma_load_transaction(&accelerator_transaction);

    GENIO_PERF_BEGIN(application);
    dma_launch(&accelerator_transaction);
    if (wait_for_accelerator() == 0 && output_is_complete()) {
        status = 0;
    }
    GENIO_PERF_END(application);

    stop_traffic();
    print_metrics();
    (void)safa_clear_done(&safa);
    printf("GENIO_STATUS:%d\n", status);

    return status;
}
