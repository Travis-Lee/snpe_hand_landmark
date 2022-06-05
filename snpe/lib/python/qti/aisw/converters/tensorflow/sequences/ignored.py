# =============================================================================
#
#  Copyright (c) 2018-2020 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


real_div_sequence = GraphSequence([
    ConverterSequenceNode('root', ['RealDiv']),
    NonConsumableConverterSequenceNode('a', ['?']),
    NonConsumableConverterSequenceNode('b', ['?'])
])
real_div_sequence.set_inputs('root', ['a', 'b'])
real_div_sequence.set_outputs(['root'])

identity_sequence = GraphSequence([
    ConverterSequenceNode('root', ['Identity']),
    NonConsumableConverterSequenceNode('any', ['?']),
])
identity_sequence.set_inputs('root', ['any'])
identity_sequence.set_outputs(['root'])

placeholder_with_default_sequence = GraphSequence([
    ConverterSequenceNode('root', ['PlaceholderWithDefault']),
    NonConsumableConverterSequenceNode('any', ['?']),
])
placeholder_with_default_sequence.set_inputs('root', ['any'])
placeholder_with_default_sequence.set_outputs(['root'])

ignored_sequence_1 = GraphSequence([
    ConverterSequenceNode('root', ['Pack']),
    ConverterSequenceNode('a', ['Add']),
    ConverterSequenceNode('b', ['Add']),
    ConverterSequenceNode('c', ['Mul']),
    ConverterSequenceNode('d', ['Mul']),
    ConverterSequenceNode('e', ['?']),
    ConverterSequenceNode('f', ['?']),
    ConverterSequenceNode('g', ['?']),
    ConverterSequenceNode('h', ['?']),
    ConverterSequenceNode('i', ['?']),
    ConverterSequenceNode('j', ['?']),
    ConverterSequenceNode('k', ['?']),
    ConverterSequenceNode('l', ['?'])
])
ignored_sequence_1.set_inputs('root', ['a', 'b', 'e', 'f'])
ignored_sequence_1.set_inputs('a', ['c', 'g'])
ignored_sequence_1.set_inputs('b', ['d', 'h'])
ignored_sequence_1.set_inputs('c', ['i', 'j'])
ignored_sequence_1.set_inputs('d', ['k', 'l'])
ignored_sequence_1.set_outputs(['root'])

ignored_sequence_2 = GraphSequence([
    ConverterSequenceNode('root', ['Pack']),
    ConverterSequenceNode('a', ['Mul']),
    ConverterSequenceNode('b', ['Mul']),
    ConverterSequenceNode('e', ['?']),
    ConverterSequenceNode('f', ['?'])
])
ignored_sequence_2.set_inputs('root', ['a', 'b', 'e', 'f'])
ignored_sequence_2.set_outputs(['root'])

cast_sequence = GraphSequence([
    ConverterSequenceNode('root', ['Cast']),
    NonConsumableConverterSequenceNode('any', ['?'])
])
cast_sequence.set_inputs('root', ['any'])
cast_sequence.set_outputs(['root'])

dropout_cell_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('is_training/read', ['?']),
    ConverterSequenceNode('Dropout/cond/Switch', ['Switch']),
    NonConsumableConverterSequenceNode('Dropout/cond/switch_t', ['Identity']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/random_uniform/min', ['Const']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/random_uniform/max', ['Const']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/Shape', ['Const']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform/sub', ['Sub']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform/RandomUniform', ['RandomUniform']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform/mul', ['Mul']),
    ConverterSequenceNode('Dropout/cond/dropout/random_uniform', ['Add']),
    NonConsumableConverterSequenceNode('Dropout/cond/dropout/keep_prob', ['Const']),
    NonConsumableConverterSequenceNode('Dropout/cond/pred_id', ['Identity']),
    ConverterSequenceNode('Dropout/cond/dropout/add', ['Add']),
    ConverterSequenceNode('Dropout/cond/dropout/div/Switch', ['Switch']),
    ConverterSequenceNode('Dropout/cond/dropout/Floor', ['Floor']),
    ConverterSequenceNode('Dropout/cond/dropout/div', ['RealDiv']),
    ConverterSequenceNode('Dropout/cond/dropout/mul', ['Mul']),
    ConverterSequenceNode('Dropout/cond/Switch_1', ['Switch']),
    ConverterSequenceNode('Dropout/cond/Merge', ['Merge']),
    NonConsumableConverterSequenceNode('stub_20', ['?']),
    NonConsumableConverterSequenceNode('stub_25', ['?']),
])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/add',
                              ['Dropout/cond/dropout/keep_prob', 'Dropout/cond/dropout/random_uniform'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/Floor', ['Dropout/cond/dropout/add'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform/mul',
                              ['Dropout/cond/dropout/random_uniform/RandomUniform',
                               'Dropout/cond/dropout/random_uniform/sub'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/div',
                              ['Dropout/cond/dropout/div/Switch', 'Dropout/cond/dropout/keep_prob'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform', ['Dropout/cond/dropout/random_uniform/mul',
                                                                      'Dropout/cond/dropout/random_uniform/min'])
dropout_cell_sequence.set_inputs('Dropout/cond/Switch', ['stub_20', 'is_training/read'])
dropout_cell_sequence.set_inputs('Dropout/cond/pred_id', ['is_training/read'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform/RandomUniform',
                              ['Dropout/cond/dropout/Shape'])
dropout_cell_sequence.set_inputs('Dropout/cond/Merge', ['Dropout/cond/Switch_1', 'Dropout/cond/dropout/mul'])
dropout_cell_sequence.set_inputs('Dropout/cond/switch_t', ['Dropout/cond/Switch'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/mul',
                              ['Dropout/cond/dropout/div', 'Dropout/cond/dropout/Floor'])
dropout_cell_sequence.set_inputs('Dropout/cond/Switch_1', ['stub_25', 'Dropout/cond/pred_id'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/random_uniform/sub',
                              ['Dropout/cond/dropout/random_uniform/max',
                               'Dropout/cond/dropout/random_uniform/min'])
dropout_cell_sequence.set_inputs('Dropout/cond/dropout/div/Switch', ['stub_25', 'Dropout/cond/pred_id'])
dropout_cell_sequence.set_outputs(['Dropout/cond/Merge'])

# ignore these patterns that generate const as shape feeding into reshape in FRVSR,
# since reshape doesn't need shape input to resolve.
pack_4_strided_slice_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('input', ['Shape']),
    ConverterSequenceNode('strided_slice1', ['StridedSlice']),
    ConverterSequenceNode('stride1', ['Const']),
    ConverterSequenceNode('begin1', ['Const']),
    ConverterSequenceNode('end1', ['Const']),
    NonConsumableConverterSequenceNode('strided_slice2', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride2', ['Const']),
    NonConsumableConverterSequenceNode('begin2', ['Const']),
    NonConsumableConverterSequenceNode('end2', ['Const']),
    NonConsumableConverterSequenceNode('strided_slice3', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride3', ['Const']),
    NonConsumableConverterSequenceNode('begin3', ['Const']),
    NonConsumableConverterSequenceNode('end3', ['Const']),
    NonConsumableConverterSequenceNode('strided_slice4', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride4', ['Const']),
    NonConsumableConverterSequenceNode('begin4', ['Const']),
    NonConsumableConverterSequenceNode('end4', ['Const']),
    ConverterSequenceNode('root', ['Pack']),
])
pack_4_strided_slice_sequence.set_inputs('strided_slice1', ['input', 'begin1', 'end1', 'stride1'])
pack_4_strided_slice_sequence.set_inputs('strided_slice2', ['input', 'begin2', 'end2', 'stride2'])
pack_4_strided_slice_sequence.set_inputs('strided_slice3', ['input', 'begin3', 'end3', 'stride3'])
pack_4_strided_slice_sequence.set_inputs('strided_slice4', ['input', 'begin4', 'end4', 'stride4'])
pack_4_strided_slice_sequence.set_inputs('root', ['strided_slice1', 'strided_slice2', 'strided_slice3', 'strided_slice4'])
pack_4_strided_slice_sequence.set_outputs(['root'])

pack_const_mul_strided_slice_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('input', ['Shape']),
    NonConsumableConverterSequenceNode('strided_slice1', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride1', ['Const']),
    NonConsumableConverterSequenceNode('begin1', ['Const']),
    NonConsumableConverterSequenceNode('end1', ['Const']),
    NonConsumableConverterSequenceNode('strided_slice2', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride2', ['Const']),
    NonConsumableConverterSequenceNode('begin2', ['Const']),
    NonConsumableConverterSequenceNode('end2', ['Const']),
    NonConsumableConverterSequenceNode('strided_slice3', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride3', ['Const']),
    NonConsumableConverterSequenceNode('begin3', ['Const']),
    NonConsumableConverterSequenceNode('end3', ['Const']),
    ConverterSequenceNode('mul1', ['Mul']),
    ConverterSequenceNode('const1', ['Const']),
    ConverterSequenceNode('root', ['Pack']),
])
pack_const_mul_strided_slice_sequence.set_inputs('strided_slice1', ['input', 'begin1', 'end1', 'stride1'])
pack_const_mul_strided_slice_sequence.set_inputs('strided_slice2', ['input', 'begin2', 'end2', 'stride2'])
pack_const_mul_strided_slice_sequence.set_inputs('strided_slice3', ['input', 'begin3', 'end3', 'stride3'])
pack_const_mul_strided_slice_sequence.set_inputs('mul1', ['strided_slice1', 'strided_slice3'])
pack_const_mul_strided_slice_sequence.set_inputs('root', ['const1', 'mul1', 'strided_slice2'])
pack_const_mul_strided_slice_sequence.set_outputs(['root'])

pack_3_strided_slice_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('input', ['Shape']),
    NonConsumableConverterSequenceNode('input1', ['Shape']),
    NonConsumableConverterSequenceNode('strided_slice1', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride1', ['Const']),
    NonConsumableConverterSequenceNode('begin1', ['Const']),
    NonConsumableConverterSequenceNode('end1', ['Const']),
    NonConsumableConverterSequenceNode('strided_slice2', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride2', ['Const']),
    NonConsumableConverterSequenceNode('begin2', ['Const']),
    NonConsumableConverterSequenceNode('end2', ['Const']),
    NonConsumableConverterSequenceNode('strided_slice3', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride3', ['Const']),
    NonConsumableConverterSequenceNode('begin3', ['Const']),
    NonConsumableConverterSequenceNode('end3', ['Const']),
    ConverterSequenceNode('root', ['Pack']),
])
pack_3_strided_slice_sequence.set_inputs('strided_slice1', ['input', 'begin1', 'end1', 'stride1'])
pack_3_strided_slice_sequence.set_inputs('strided_slice2', ['input', 'begin2', 'end2', 'stride2'])
pack_3_strided_slice_sequence.set_inputs('strided_slice3', ['input1', 'begin3', 'end3', 'stride3'])
pack_3_strided_slice_sequence.set_inputs('root', ['strided_slice1', 'strided_slice2', 'strided_slice3'])
pack_3_strided_slice_sequence.set_outputs(['root'])

pack_strided_slice_mul_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('strided_slice_1', ['StridedSlice']),
    NonConsumableConverterSequenceNode('strided_slice', ['StridedSlice']),
    NonConsumableConverterSequenceNode('input', ['Shape']),
    NonConsumableConverterSequenceNode('strided_slice_2', ['StridedSlice']),
    ConverterSequenceNode('mul', ['Mul']),
    NonConsumableConverterSequenceNode('strided_slice_3', ['StridedSlice']),
    ConverterSequenceNode('mul_1', ['Mul']),
    ConverterSequenceNode('root', ['Pack']),
    NonConsumableConverterSequenceNode('stub_8', ['?']),
    NonConsumableConverterSequenceNode('stub_9', ['?']),
    NonConsumableConverterSequenceNode('stub_10', ['?']),
    NonConsumableConverterSequenceNode('stub_11', ['?']),
    NonConsumableConverterSequenceNode('stub_12', ['?']),
    NonConsumableConverterSequenceNode('stub_13', ['?']),
    NonConsumableConverterSequenceNode('stub_15', ['?']),
    NonConsumableConverterSequenceNode('stub_16', ['?']),
    NonConsumableConverterSequenceNode('stub_17', ['?']),
    NonConsumableConverterSequenceNode('stub_18', ['?']),
    NonConsumableConverterSequenceNode('stub_19', ['?']),
    NonConsumableConverterSequenceNode('stub_20', ['?']),
])
pack_strided_slice_mul_sequence.set_inputs('strided_slice', ['input', 'stub_11', 'stub_12', 'stub_13'])
pack_strided_slice_mul_sequence.set_inputs('strided_slice_1', ['input', 'stub_8', 'stub_9', 'stub_10'])
pack_strided_slice_mul_sequence.set_inputs('mul', ['strided_slice', 'strided_slice_1'])
pack_strided_slice_mul_sequence.set_inputs('strided_slice_2', ['input', 'stub_15', 'stub_16', 'stub_17'])
pack_strided_slice_mul_sequence.set_inputs('mul_1', ['mul', 'strided_slice_2'])
pack_strided_slice_mul_sequence.set_inputs('strided_slice_3', ['input', 'stub_18', 'stub_19', 'stub_20'])
pack_strided_slice_mul_sequence.set_inputs('root', ['mul_1', 'strided_slice_3'])
pack_strided_slice_mul_sequence.set_outputs(['root'])

# ignore this and below pattern that generate const which could never be used.
shape_strided_slice_pack_sequence = GraphSequence([
    NonConsumableConverterSequenceNode('input', ['?']),
    NonConsumableConverterSequenceNode('shape', ['Shape']),
    NonConsumableConverterSequenceNode('strided_slice1', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride1', ['Const']),
    NonConsumableConverterSequenceNode('begin1', ['Const']),
    NonConsumableConverterSequenceNode('end1', ['Const']),
    ConverterSequenceNode('const_1', ['Const']),
    ConverterSequenceNode('const_2', ['Const']),
    ConverterSequenceNode('root', ['Pack']),
])
shape_strided_slice_pack_sequence.set_inputs('shape', ['input'])
shape_strided_slice_pack_sequence.set_inputs('strided_slice1', ['shape', 'begin1', 'end1', 'stride1'])
shape_strided_slice_pack_sequence.set_inputs('root', ['strided_slice1', 'const_1', 'const_2'])
shape_strided_slice_pack_sequence.set_outputs(['root'])

shape_strided_slice_pack_sequence1 = GraphSequence([
    NonConsumableConverterSequenceNode('input', ['?']),
    NonConsumableConverterSequenceNode('shape', ['Shape']),
    NonConsumableConverterSequenceNode('strided_slice1', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride1', ['Const']),
    NonConsumableConverterSequenceNode('begin1', ['Const']),
    NonConsumableConverterSequenceNode('end1', ['Const']),
    ConverterSequenceNode('const_1', ['Const']),
    ConverterSequenceNode('const_2', ['Const']),
    ConverterSequenceNode('const_3', ['Const']),
    ConverterSequenceNode('root', ['Pack']),
])
shape_strided_slice_pack_sequence1.set_inputs('shape', ['input'])
shape_strided_slice_pack_sequence1.set_inputs('strided_slice1', ['shape', 'begin1', 'end1', 'stride1'])
shape_strided_slice_pack_sequence1.set_inputs('root', ['strided_slice1', 'const_1', 'const_2', 'const_3'])
shape_strided_slice_pack_sequence1.set_outputs(['root'])

shape_strided_slice_pack_sequence2 = GraphSequence([
    NonConsumableConverterSequenceNode('input', ['?']),
    NonConsumableConverterSequenceNode('shape', ['Shape']),
    NonConsumableConverterSequenceNode('strided_slice1', ['StridedSlice']),
    NonConsumableConverterSequenceNode('stride1', ['Const']),
    NonConsumableConverterSequenceNode('begin1', ['Const']),
    NonConsumableConverterSequenceNode('end1', ['Const']),
    ConverterSequenceNode('const_1', ['Const']),
    ConverterSequenceNode('root', ['Pack']),
])
shape_strided_slice_pack_sequence2.set_inputs('shape', ['input'])
shape_strided_slice_pack_sequence2.set_inputs('strided_slice1', ['shape', 'begin1', 'end1', 'stride1'])
shape_strided_slice_pack_sequence2.set_inputs('root', ['strided_slice1', 'const_1'])
shape_strided_slice_pack_sequence2.set_outputs(['root'])

# ignore this from cnns_mobilenet_ssd_april18
unused_pattern_feeding_into_multi_class_nms_sequence = GraphSequence([
    ConverterSequenceNode('Postprocessor/ToFloat', ['Cast']),
    ConverterSequenceNode('Postprocessor/ToFloat_1', ['Cast']),
    ConverterSequenceNode('Postprocessor/ToFloat_2', ['Cast']),
    ConverterSequenceNode('Postprocessor/ToFloat_1/x', ['Const']),
    ConverterSequenceNode('Postprocessor/ToFloat_2/x', ['Const']),
    ConverterSequenceNode('Postprocessor/unstack', ['Unpack']),
    ConverterSequenceNode('Postprocessor/zeros_like', ['ZerosLike']),
    ConverterSequenceNode('Postprocessor/zeros_like_1', ['ZerosLike']),
    ConverterSequenceNode('Postprocessor/div', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/div_1', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/stack_1', ['Pack']),
])
unused_pattern_feeding_into_multi_class_nms_sequence.set_inputs('Postprocessor/unstack', ['Postprocessor/ToFloat'])
unused_pattern_feeding_into_multi_class_nms_sequence.set_inputs('Postprocessor/ToFloat_1', ['Postprocessor/ToFloat_1/x'])
unused_pattern_feeding_into_multi_class_nms_sequence.set_inputs('Postprocessor/div', ['Postprocessor/unstack', 'Postprocessor/ToFloat_1'])
unused_pattern_feeding_into_multi_class_nms_sequence.set_inputs('Postprocessor/ToFloat_2', ['Postprocessor/ToFloat_2/x'])
unused_pattern_feeding_into_multi_class_nms_sequence.set_inputs('Postprocessor/div_1', ['Postprocessor/unstack', 'Postprocessor/ToFloat_2'])
unused_pattern_feeding_into_multi_class_nms_sequence.set_inputs('Postprocessor/zeros_like', ['Postprocessor/unstack'])
unused_pattern_feeding_into_multi_class_nms_sequence.set_inputs('Postprocessor/zeros_like_1', ['Postprocessor/unstack'])
unused_pattern_feeding_into_multi_class_nms_sequence.set_inputs('Postprocessor/stack_1', ['Postprocessor/zeros_like', 'Postprocessor/zeros_like_1', 'Postprocessor/div', 'Postprocessor/div_1'])
unused_pattern_feeding_into_multi_class_nms_sequence.set_outputs(['Postprocessor/stack_1'])

# ignore this from cnns_mobilenet_v1_fpn_ssd
unused_pattern_feeding_into_multi_class_nms_sequence1 = GraphSequence([
    ConverterSequenceNode('Postprocessor/ToFloat', ['Cast']),
    ConverterSequenceNode('stub_1', ['Const']),
    ConverterSequenceNode('stub_2', ['Const']),
    ConverterSequenceNode('Postprocessor/unstack', ['Unpack']),
    ConverterSequenceNode('Postprocessor/zeros_like', ['ZerosLike']),
    ConverterSequenceNode('Postprocessor/zeros_like_1', ['ZerosLike']),
    ConverterSequenceNode('Postprocessor/div', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/div_1', ['RealDiv']),
    ConverterSequenceNode('Postprocessor/stack_1', ['Pack']),
])
unused_pattern_feeding_into_multi_class_nms_sequence1.set_inputs('Postprocessor/unstack', ['Postprocessor/ToFloat'])
unused_pattern_feeding_into_multi_class_nms_sequence1.set_inputs('Postprocessor/div', ['Postprocessor/unstack', 'stub_1'])
unused_pattern_feeding_into_multi_class_nms_sequence1.set_inputs('Postprocessor/div_1', ['Postprocessor/unstack', 'stub_2'])
unused_pattern_feeding_into_multi_class_nms_sequence1.set_inputs('Postprocessor/zeros_like', ['Postprocessor/unstack'])
unused_pattern_feeding_into_multi_class_nms_sequence1.set_inputs('Postprocessor/zeros_like_1', ['Postprocessor/unstack'])
unused_pattern_feeding_into_multi_class_nms_sequence1.set_inputs('Postprocessor/stack_1', ['Postprocessor/zeros_like', 'Postprocessor/zeros_like_1', 'Postprocessor/div', 'Postprocessor/div_1'])
unused_pattern_feeding_into_multi_class_nms_sequence1.set_outputs(['Postprocessor/stack_1'])