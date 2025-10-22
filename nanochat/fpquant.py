from fp_quant import FPQuantDtype, FPQuantConfig, replace_quantize_with_fp_quant_linear


def add_qat(model, store_master_weights):
    model = replace_quantize_with_fp_quant_linear(
        model,
        fp_quant_linear_config=FPQuantConfig(
            forward_dtype=FPQuantDtype.MXFP4,
            forward_method="abs_max",
            hadamard_group_size=128,
            backward_dtype=FPQuantDtype.MXFP8,
            store_master_weights=store_master_weights,
            modules_to_not_convert=["lm_head"],
        ),
    )
    return model
