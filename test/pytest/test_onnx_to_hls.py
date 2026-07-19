from onnx import TensorProto, helper

from hls4ml.converters.onnx_to_hls import get_input_shape


def test_get_input_shape_skips_empty_optional_inputs():
    input_tensor = helper.make_tensor_value_info('global_in', TensorProto.FLOAT, [1, 1, 25, 25])
    scales_tensor = helper.make_tensor_value_info('resize_scales', TensorProto.FLOAT, [4])
    output_tensor = helper.make_tensor_value_info('global_out', TensorProto.FLOAT, [1, 1, 50, 50])

    resize_node = helper.make_node(
        'Resize',
        name='resize',
        inputs=['global_in', '', 'resize_scales'],
        outputs=['global_out'],
        mode='nearest',
    )

    graph = helper.make_graph(
        nodes=[resize_node],
        name='ResizeWithoutRoiGraph',
        inputs=[input_tensor],
        outputs=[output_tensor],
        value_info=[scales_tensor],
    )

    assert get_input_shape(graph, resize_node) == [[1, 1, 25, 25], [4]]
