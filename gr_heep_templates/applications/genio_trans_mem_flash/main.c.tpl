#include <stdint.h>
#include <stdio.h>

#include "dma.h"
#include "mmio.h"
#include "w25q128jw.h"

#include "genio_app_config.h"
#include "genio_perf.h"
#include "gr_heep.h"
#include "main.h"
#include "safa.h"

#define GENIO_INPUT_BYTES (GENIO_INPUT_WORDS * sizeof(uint32_t))
#define GENIO_OUTPUT_BYTES (GENIO_OUTPUT_WORDS * sizeof(uint32_t))
#define GENIO_FLASH_SECTOR_BYTES 4096u
#define GENIO_DMA_MAX_WORDS 65535u

#define TRAFFIC_WORDS 1024u
#define TRAFFIC_DMA_CHANNEL 1u

static dma_trans_t accelerator_transaction;
static dma_target_t accelerator_source;
static dma_target_t accelerator_destination;
static dma_trans_t traffic_transaction;
static dma_target_t traffic_source_target;
static dma_target_t traffic_destination_target;
static safa_t safa;
static safa_counters_t captured_counters;
static int captured_counters_valid;

static uint32_t traffic_source[TRAFFIC_WORDS]
    __attribute__((aligned(16)));
static uint32_t traffic_destination[TRAFFIC_WORDS]
    __attribute__((aligned(16)));

static int validate_configuration(void) {
    if (GENIO_INPUT_WORDS == 0u || GENIO_OUTPUT_WORDS == 0u) {
        printf("Invalid image dimensions\n");
        return -1;
    }
    if (GENIO_INPUT_WORDS > GENIO_DMA_MAX_WORDS) {
        printf("Input exceeds the DMA 1D transfer limit\n");
        return -1;
    }
    if (GENIO_OUTPUT_WORDS > GENIO_INPUT_WORDS) {
        printf("SAFA output cannot exceed the DMA input transfer size\n");
        return -1;
    }
    if (GENIO_DMA_CHANNEL != 0u) {
        printf("SAFA must use DMA HW FIFO channel 0\n");
        return -1;
    }

    const uint64_t output_start = (uint64_t)GENIO_FLASH_OUTPUT_OFFSET;
    const uint64_t output_end = output_start + GENIO_OUTPUT_BYTES;
    const uint64_t last_sector_start =
        ((output_end - 1u) / GENIO_FLASH_SECTOR_BYTES) *
        GENIO_FLASH_SECTOR_BYTES;

    if ((GENIO_FLASH_OUTPUT_OFFSET % GENIO_FLASH_SECTOR_BYTES) != 0u) {
        printf("Flash output offset must be sector aligned\n");
        return -1;
    }
    if (output_start > (uint64_t)MAX_FLASH_ADDR ||
        output_end > (uint64_t)MAX_FLASH_ADDR ||
        last_sector_start + GENIO_FLASH_SECTOR_BYTES >
            (uint64_t)MAX_FLASH_ADDR) {
        printf("Flash output range exceeds W25Q128JW capacity\n");
        return -1;
    }
    return 0;
}

static int configure_flash(void) {
    if (w25q128jw_init(spi_flash) != FLASH_OK) {
        printf("SPI flash initialization failed\n");
        return -1;
    }
    return 0;
}

static int configure_safa(void) {
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

    /*
     * SAFA's output must first be drained to SRAM. SPI page programming needs
     * command and erase sequencing that cannot be represented by one DMA target.
     */
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

static int configure_traffic_dma(void) {
    traffic_source_target.ptr = (uint8_t *)traffic_source;
    traffic_source_target.inc_d1_du = 1;
    traffic_source_target.trig = DMA_TRIG_MEMORY;
    traffic_source_target.type = DMA_DATA_TYPE_WORD;

    traffic_destination_target.ptr = (uint8_t *)traffic_destination;
    traffic_destination_target.inc_d1_du = 1;
    traffic_destination_target.trig = DMA_TRIG_MEMORY;
    traffic_destination_target.type = DMA_DATA_TYPE_WORD;

    traffic_transaction.src = &traffic_source_target;
    traffic_transaction.dst = &traffic_destination_target;
    traffic_transaction.mode = DMA_TRANS_MODE_SINGLE;
    traffic_transaction.hw_fifo_en = 0;
    traffic_transaction.channel = TRAFFIC_DMA_CHANNEL;
    traffic_transaction.dim = DMA_DIM_CONF_1D;
    traffic_transaction.size_d1_du = TRAFFIC_WORDS;
    traffic_transaction.end = DMA_TRANS_END_POLLING;

    if (dma_validate_transaction(
            &traffic_transaction,
            DMA_ENABLE_REALIGN,
            DMA_PERFORM_CHECKS_INTEGRITY
        ) != DMA_CONFIG_OK) {
        printf("Traffic DMA validation failed\n");
        return -1;
    }
    return 0;
}

static void initialize_traffic(void) {
    uint32_t state = 0x6d2b79f5u;
    for (uint32_t i = 0; i < TRAFFIC_WORDS; ++i) {
        state = state * 1664525u + 1013904223u;
        traffic_source[i] = state;
        traffic_destination[i] = 0u;
    }
}

static void capture_safa_counters(void) {
    if (safa_get_counters(&safa, &captured_counters) == SAFA_RESULT_OK) {
        captured_counters_valid = 1;
    }
}

static int wait_for_accelerator(void) {
    const uint64_t start = genio_read_cycles();
    int dma_completed = 0;

    while (1) {
        safa_status_t status;
        if (safa_get_status(&safa, &status) != SAFA_RESULT_OK) {
            return -1;
        }
        if (status.error || status.aborted) {
            printf("SAFA stopped, errors=0x%08lx\n",
                   (unsigned long)safa_get_errors(&safa));
            capture_safa_counters();
            return -1;
        }
        if (!dma_completed && dma_is_ready(GENIO_DMA_CHANNEL)) {
            dma_completed = 1;
        }
        if (dma_completed && status.done) {
            return 0;
        }
        if (GENIO_TIMEOUT_CYCLES != 0u &&
            genio_read_cycles() - start > (uint64_t)GENIO_TIMEOUT_CYCLES) {
            printf("SAFA timeout\n");
            /*
             * Do not reset SAFA while DMA channel 0 may still be pushing data.
             * Resetting would clear metrics and auto-start another partial run.
             */
            capture_safa_counters();
            return -1;
        }
    }
}

static int load_dma_transaction(dma_trans_t *transaction, const char *name) {
    dma_config_flags_t result = dma_load_transaction(transaction);
    if (result != DMA_CONFIG_OK) {
        printf("%s DMA load failed: 0x%04x\n", name, (unsigned int)result);
        return -1;
    }
    return 0;
}

static int launch_dma_transaction(dma_trans_t *transaction, const char *name) {
    dma_config_flags_t result = dma_launch(transaction);
    if (result != DMA_CONFIG_OK) {
        printf("%s DMA launch failed: 0x%04x\n", name, (unsigned int)result);
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

static int store_output_in_flash(void) {
    /*
     * Avoid a second DMA transaction after HW FIFO mode. The BSP preserves
     * bytes surrounding the output range while erasing and rewriting sectors.
     */
    w25q_error_codes_t result = w25q128jw_erase_and_write_standard(
        GENIO_FLASH_OUTPUT_OFFSET,
        image_output,
        (uint32_t)GENIO_OUTPUT_BYTES
    );
    if (result != FLASH_OK) {
        printf("SPI flash output write failed: %d\n", (int)result);
        return -1;
    }
    return 0;
}

static uint32_t buffer_checksum(const uint32_t *buffer, uint32_t words) {
    uint32_t checksum = 0u;
    for (uint32_t i = 0; i < words; ++i) {
        checksum = (checksum << 5) | (checksum >> 27);
        checksum ^= buffer[i];
    }
    return checksum;
}

static void print_metrics(void) {
    safa_counters_t counters = captured_counters;
    if (!captured_counters_valid) {
        if (safa_get_counters(&safa, &counters) != SAFA_RESULT_OK) {
            return;
        }
    }

    printf("GENIO_METRIC:safa_active_cycles:%lu\n", (unsigned long)counters.active_cycles);
    printf("GENIO_METRIC:safa_input_stall_cycles:%lu\n", (unsigned long)counters.input_stall_cycles);
    printf("GENIO_METRIC:safa_output_stall_cycles:%lu\n", (unsigned long)counters.output_stall_cycles);
    printf("GENIO_METRIC:safa_dma_push_stall_cycles:%lu\n", (unsigned long)counters.dma_push_stall_cycles);
    printf("GENIO_METRIC:safa_dma_pop_stall_cycles:%lu\n", (unsigned long)counters.dma_pop_stall_cycles);
    printf("GENIO_METRIC:safa_input_words:%lu\n", (unsigned long)counters.input_accepted);
    printf("GENIO_METRIC:safa_output_words:%lu\n", (unsigned long)counters.output_popped);
    printf("GENIO_METRIC:flash_output_offset:%lu\n", (unsigned long)GENIO_FLASH_OUTPUT_OFFSET);
    printf("GENIO_METRIC:flash_output_bytes:%lu\n", (unsigned long)GENIO_OUTPUT_BYTES);
    printf("GENIO_METRIC:output_checksum:%lu\n",
           (unsigned long)buffer_checksum(image_output, GENIO_OUTPUT_WORDS));
}

int main(void) {
    int status = 1;
    int traffic_started = 0;

    genio_perf_init();
    dma_init(NULL);
    initialize_traffic();

    if (validate_configuration() != 0 ||
        configure_flash() != 0 ||
        configure_safa() != 0 ||
        configure_traffic_dma() != 0) {
        printf("GENIO_STATUS:%d\n", status);
        return status;
    }

    GENIO_PERF_BEGIN(application);
    if (configure_accelerator_dma() == 0) {
        if (load_dma_transaction(&accelerator_transaction, "Accelerator") == 0 &&
            load_dma_transaction(&traffic_transaction, "Traffic") == 0 &&
            launch_dma_transaction(&accelerator_transaction, "Accelerator") == 0) {
            if (launch_dma_transaction(&traffic_transaction, "Traffic") == 0) {
                traffic_started = 1;
            }

            if (wait_for_accelerator() == 0 &&
                output_is_complete() &&
                traffic_started) {
                while (!dma_is_ready(TRAFFIC_DMA_CHANNEL)) {
                }
                traffic_started = 0;

                GENIO_PERF_BEGIN(flash_write);
                if (store_output_in_flash() == 0) {
                    status = 0;
                }
                GENIO_PERF_END(flash_write);
            }
        }
    }
    GENIO_PERF_END(application);

    if (traffic_started) {
        while (!dma_is_ready(TRAFFIC_DMA_CHANNEL)) {
        }
    }
    print_metrics();
    (void)safa_clear_done(&safa);
    printf("GENIO_METRIC:traffic_words:%lu\n", (unsigned long)TRAFFIC_WORDS);
    printf("GENIO_METRIC:traffic_checksum:%lu\n",
           (unsigned long)buffer_checksum(traffic_destination, TRAFFIC_WORDS));
    printf("GENIO_STATUS:%d\n", status);
    return status;
}
