import re

from xheep import XHeep
from cpu.cpu import CPU
from bus_type import BusType
from memory_ss.memory_ss import MemorySS
from memory_ss.linker_section import LinkerSection
from peripherals.base_peripherals import (
    SOC_ctrl,
    Bootrom,
    SPI_flash,
    SPI_memio,
    DMA,
    Power_manager,
    RV_timer_ao,
    Fast_intr_ctrl,
    Ext_peripheral,
    Pad_control,
    GPIO_ao,
    W25Q128JW_Controller,
)
from peripherals.user_peripherals import (
    RV_plic,
    SPI_host,
    GPIO,
    I2C,
    RV_timer,
    SPI2,
    PDM2PCM,
    I2S,
    UART,
)
from peripherals.base_peripherals_domain import BasePeripheralDomain
from peripherals.user_peripherals_domain import UserPeripheralDomain


def config():
    system = XHeep(BusType.@BUS_TYPE@)
    system.set_cpu(CPU("@CPU@"))

    memory_ss = MemorySS()
    memory_ss.add_ram_banks(@RAM_BANKS@)
    if @INTERLEAVED_BANK_COUNT@:
        memory_ss.add_ram_banks_il(
            @INTERLEAVED_BANK_COUNT@,
            @INTERLEAVED_BANK_SIZE@,
            "data_interleaved",
        )
    memory_ss.add_linker_section(
        LinkerSection.by_size("code", 0, @CODE_SECTION_END@)
    )
    memory_ss.add_linker_section(
        LinkerSection("data", @DATA_SECTION_START@, None)
    )
    system.set_memory_ss(memory_ss)

    base_peripheral_domain = BasePeripheralDomain()
    user_peripheral_domain = UserPeripheralDomain()

    base_peripheral_domain.add_peripheral(SOC_ctrl(0x00000000))
    base_peripheral_domain.add_peripheral(Bootrom(0x00010000))
    base_peripheral_domain.add_peripheral(SPI_flash(0x00020000, 0x00008000))
    base_peripheral_domain.add_peripheral(SPI_memio(0x00028000, 0x00000008))
    base_peripheral_domain.add_peripheral(
        W25Q128JW_Controller(0x00029000, 0x00007000)
    )
    base_peripheral_domain.add_peripheral(
        DMA(
            address=0x00030000,
            length=0x00010000,
            ch_length=0x100,
            num_channels=@DMA_NUM_CHANNELS@,
            num_master_ports=@DMA_NUM_MASTER_PORTS@,
            num_channels_per_master_port=@DMA_NUM_CHANNELS_PER_MASTER_PORT@,
            fifo_depth=@DMA_FIFO_DEPTH@,
            addr_mode="@DMA_ADDR_MODE@",
            subaddr_mode="@DMA_SUBADDR_MODE@",
            hw_fifo_mode="yes",
            zero_padding="@DMA_ZERO_PADDING@",
        )
    )
    base_peripheral_domain.add_peripheral(Power_manager(0x00040000))
    base_peripheral_domain.add_peripheral(RV_timer_ao(0x00050000))
    base_peripheral_domain.add_peripheral(Fast_intr_ctrl(0x00060000))
    base_peripheral_domain.add_peripheral(Ext_peripheral(0x00070000))
    base_peripheral_domain.add_peripheral(Pad_control(0x00080000))
    base_peripheral_domain.add_peripheral(GPIO_ao(0x00090000))

    user_peripheral_domain.add_peripheral(RV_plic(0x00000000))
    user_peripheral_domain.add_peripheral(SPI_host(0x00010000))
    user_peripheral_domain.add_peripheral(GPIO(0x00020000))
    user_peripheral_domain.add_peripheral(I2C(0x00030000))
    user_peripheral_domain.add_peripheral(RV_timer(0x00040000))
    user_peripheral_domain.add_peripheral(SPI2(0x00050000))
    user_peripheral_domain.add_peripheral(PDM2PCM(0x00060000, cic_only=True))
    user_peripheral_domain.add_peripheral(I2S(0x00070000))
    user_peripheral_domain.add_peripheral(UART(0x00080000))

    system.add_peripheral_domain(base_peripheral_domain)
    system.add_peripheral_domain(user_peripheral_domain)
    system.add_extension("gr-heep", gr_heep_config())

    return system


def gr_heep_config():
    ext_xbar_nmasters = 0
    ext_xbar_slaves = {}
    ext_periph = {
        "SAFA": {
            "offset": @SAFA_OFFSET@,
            "length": @SAFA_LENGTH@,
        }
    }
    ao_spc_num = 1
    external_interrupts = 1
    hw_fifo_channels = @HW_FIFO_CHANNELS@

    slaves = []
    if len(ext_xbar_slaves) > 0:
        idx = 0
        for a_slave, slave_config in ext_xbar_slaves.items():
            slaves.append(
                {
                    "name": CamelCase(a_slave),
                    "SCREAMING_NAME": SCREAMING_SNAKE_CASE(a_slave),
                    "idx": idx,
                    "offset": slave_config["offset"],
                    "size": slave_config["length"],
                    "end_address": slave_config["offset"]
                    + slave_config["length"],
                }
            )
            idx += 1

    peripherals = []
    if len(ext_periph) > 0:
        idx = 0
        for a_peripheral, peripheral_config in ext_periph.items():
            peripherals.append(
                {
                    "name": CamelCase(a_peripheral),
                    "SCREAMING_NAME": SCREAMING_SNAKE_CASE(a_peripheral),
                    "idx": idx,
                    "offset": peripheral_config["offset"],
                    "size": peripheral_config["length"],
                    "end_address": peripheral_config["offset"]
                    + peripheral_config["length"],
                }
            )
            idx += 1

    return {
        "xbar_nmasters": ext_xbar_nmasters,
        "xbar_nslaves": len(ext_xbar_slaves),
        "periph_nslaves": len(ext_periph),
        "ao_spc_num": ao_spc_num,
        "slaves": slaves,
        "peripherals": peripherals,
        "ext_interrupts": external_interrupts,
        "hw_fifo_channels": hw_fifo_channels,
    }


def CamelCase(input_string):
    words = re.split(r"[^a-zA-Z0-9]+", input_string)
    return words[0].capitalize() + "".join(
        word.capitalize() for word in words[1:]
    )


def SCREAMING_SNAKE_CASE(input_string):
    words = re.sub(r"([a-z])([A-Z])", r"\1_\2", input_string)
    words = re.sub(r"[^a-zA-Z0-9]+", "_", words)
    return words.upper().strip("_")
