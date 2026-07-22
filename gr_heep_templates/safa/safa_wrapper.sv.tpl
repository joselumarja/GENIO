module safa_wrapper #(
    parameter int FIFO_DEPTH = 16
)(
    input  logic clk_i,
    input  logic rst_ni,

    input  fifo_pkg::fifo_req_t  hw_fifo_req_i,
    output fifo_pkg::fifo_resp_t hw_fifo_rsp_o,
    output logic                 hw_fifo_done_o
);

    import fifo_pkg::*;

    localparam int DATA_WIDTH = 32;
    localparam int ADDR_WIDTH = $clog2(DATA_WIDTH);
    localparam int ALMOST_FULL_THRESHOLD = FIFO_DEPTH - 4;

    logic start_i;

    logic [DATA_WIDTH-1:0] BUS_IN_dout;
    logic                  BUS_IN_empty_n;
    logic                  BUS_IN_read;

    logic [DATA_WIDTH-1:0] BUS_OUT_din;
    logic                  BUS_OUT_full_n;
    logic                  BUS_OUT_write;

    logic ap_rst;
    logic ap_start;
    logic ap_done;
    logic ap_ready;
    logic ap_idle;

    assign ap_rst = ~rst_ni;

    logic [DATA_WIDTH-1:0] in_fifo_dout;
    logic [ADDR_WIDTH-1:0] in_fifo_size;
    logic                  in_fifo_full;
    logic                  in_fifo_empty;
    logic                  in_fifo_almost_full;
    logic                  in_fifo_wr_en;
    logic                  in_fifo_rd_en;

    logic start_condition;
    logic accelerator_done_seen;

    assign start_condition = hw_fifo_req_i.push && !in_fifo_full && ap_idle;
    assign in_fifo_wr_en = hw_fifo_req_i.push && !in_fifo_full;
    assign in_fifo_rd_en = BUS_IN_read && !in_fifo_empty;
    assign in_fifo_almost_full = in_fifo_size > ALMOST_FULL_THRESHOLD;

    assign BUS_IN_dout = in_fifo_dout;
    assign BUS_IN_empty_n = !in_fifo_empty;

    sync_fifo_with_size_signal #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(FIFO_DEPTH)
    ) i_input_fifo (
        .clk   (clk_i),
        .rst   (!rst_ni),
        .wr_en (in_fifo_wr_en),
        .rd_en (in_fifo_rd_en),
        .din   (hw_fifo_req_i.data),
        .dout  (in_fifo_dout),
        .size  (in_fifo_size),
        .full  (in_fifo_full),
        .empty (in_fifo_empty)
    );

    logic [DATA_WIDTH-1:0] out_fifo_dout;
    logic [ADDR_WIDTH-1:0] out_fifo_size;
    logic                  out_fifo_full;
    logic                  out_fifo_empty;
    logic                  out_fifo_wr_en;
    logic                  out_fifo_rd_en;

    assign out_fifo_wr_en = BUS_OUT_write && !out_fifo_full;
    assign out_fifo_rd_en = hw_fifo_req_i.pop && !out_fifo_empty;
    assign BUS_OUT_full_n = !out_fifo_full;

    sync_fifo_with_size_signal #(
        .DATA_WIDTH(DATA_WIDTH),
        .DEPTH(FIFO_DEPTH)
    ) i_output_fifo (
        .clk   (clk_i),
        .rst   (!rst_ni),
        .wr_en (out_fifo_wr_en),
        .rd_en (out_fifo_rd_en),
        .din   (BUS_OUT_din),
        .dout  (out_fifo_dout),
        .size  (out_fifo_size),
        .full  (out_fifo_full),
        .empty (out_fifo_empty)
    );

    assign hw_fifo_rsp_o.data = out_fifo_dout;
    assign hw_fifo_rsp_o.full = in_fifo_full;
    assign hw_fifo_rsp_o.alm_full = in_fifo_almost_full;
    assign hw_fifo_rsp_o.empty = out_fifo_empty;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            start_i <= 1'b0;
            ap_start <= 1'b0;
            accelerator_done_seen <= 1'b0;
            hw_fifo_done_o <= 1'b0;
        end else begin
            start_i <= 1'b0;
            ap_start <= 1'b0;

            if (start_condition) begin
                start_i <= 1'b1;
                ap_start <= 1'b1;
                accelerator_done_seen <= 1'b0;
                hw_fifo_done_o <= 1'b0;
            end

            if (ap_done) begin
                accelerator_done_seen <= 1'b1;
            end

            if ((accelerator_done_seen || ap_done) && out_fifo_empty) begin
                hw_fifo_done_o <= 1'b1;
                accelerator_done_seen <= 1'b0;
            end
        end
    end

    @HLS_TOP_MODULE@ i_hls_top (
        .BUS_IN_dout    (BUS_IN_dout),
        .BUS_IN_empty_n (BUS_IN_empty_n),
        .BUS_IN_read    (BUS_IN_read),
        .BUS_OUT_din    (BUS_OUT_din),
        .BUS_OUT_full_n (BUS_OUT_full_n),
        .BUS_OUT_write  (BUS_OUT_write),
        .ap_clk         (clk_i),
        .ap_rst         (ap_rst),
        .ap_start       (ap_start),
        .ap_done        (ap_done),
        .ap_ready       (ap_ready),
        .ap_idle        (ap_idle)
    );

endmodule
