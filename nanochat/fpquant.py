from fp_quant import FPQuantDtype, FPQuantConfig, replace_quantize_with_fp_quant_linear


def add_qat(model, store_master_weights):
    model = replace_quantize_with_fp_quant_linear(
        model,
        fp_quant_linear_config=FPQuantConfig(
            forward_dtype=FPQuantDtype.MXFP4,
            forward_method="quest",
            backward_dtype=FPQuantDtype.BF16,
            store_master_weights=store_master_weights,
            modules_to_not_convert=["lm_head"],
        ),
    )
    return model
