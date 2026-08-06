#!/usr/bin/env python3
"""Prepare a QuickSRNet RGBA upscaler model for the Hexagon NPU.

What is wrong with the models as exported
-----------------------------------------
They come out of the training pipeline declaring

    image          uint8  [1, 4, height,    width   ]
    upscaled_image uint8  [1, 4, height_xN, width_xN]

and they carry the alpha channel in a side branch of *unquantized* uint8 ops:

    image -> Slice[3:4] -> Tile -> DepthToSpace -> Concat -> upscaled_image

The QNN HTP backend rejects every op in that branch. Those tensors have no
quantization encodings, so QNN.backendValidateOpConfig() fails with error 3110
for the Tile, the DepthToSpace and the Concat; ORT's layout transformer has by
then already rewritten the DepthToSpace into its internal NHWC domain, which has
no CPU kernel either, so session creation fails outright with

    Node '/DepthToSpace_token_90' OpType:DepthToSpace with domain:
    com.ms.internal.nhwc was inserted using the NHWC format as requested by
    QNNExecutionProvider, but was not selected by that EP.

This is not caused by the layout rewrite below -- the unconverted model fails
the same way. The alpha branch simply cannot run on the NPU.

What this script does
---------------------
1. Folds alpha into the quantized network. The two convolutions feeding
   DepthToSpace grow from 3*s^2 to 4*s^2 output channels; the added plane has
   zero weights and a bias that saturates the output Clip, so DepthToSpace
   lands a constant 1.0 in channel 3 and the existing Mul(255)/Round/Cast turns
   it into alpha 255. The alpha side branch is then deleted.

   RGB is untouched: an all-zero filter contributes nothing, so the colour
   channels come out bit-identical (--verify checks exactly this). Alpha stops
   being the input's alpha nearest-upscaled and becomes a constant -- which is
   what the renderer wants, and what the old unpack shader wrote.

2. Moves the declared layout to NHWC by wrapping the graph I/O in a pair of
   Transposes, so

    image          uint8  [1, height,    width,    4]
    upscaled_image uint8  [1, height_xN, width_xN, 4]

   which is byte-identical to a tightly packed R8G8B8A8_UNORM image. That is
   what lets upscaler.c move frames in and out with plain
   vkCmdCopyImageToBuffer / vkCmdCopyBufferToImage and no shader.

   The output-side Transpose folds away entirely against the NHWC graph the QNN
   EP builds. The input-side one does not -- the channel Slice that feeds the
   first convolutions blocks it -- so ORT runs it on the CPU. It is one uint8
   transpose of the render-resolution frame per inference. Removing it needs the
   first convolutions to take 4 input channels, which is a change for the export
   pipeline rather than something to patch in here.

Usage
-----
    python scripts/onnx-nhwc-io.py IN.onnx OUT.onnx [--verify]

--verify runs the original and the converted model on the CPU EP and asserts the
RGB channels match exactly and alpha is a constant 255. It needs onnxruntime and
numpy; the conversion itself only needs onnx and numpy.
"""

import argparse
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper as nh

INPUT_NAME = "image"
OUTPUT_NAME = "upscaled_image"
CHANNELS = 4


# --------------------------------------------------------------------------
# small graph helpers
# --------------------------------------------------------------------------
def dims(value_info):
    return [d.dim_param if d.dim_param else d.dim_value
            for d in value_info.type.tensor_type.shape.dim]


def find(seq, name):
    return next((x for x in seq if x.name == name), None)


def get_init(graph, name):
    return next((i for i in graph.initializer if i.name == name), None)


def set_init(graph, name, array):
    old = get_init(graph, name)
    if old is not None:
        graph.initializer.remove(old)
    graph.initializer.append(nh.from_array(array, name))


def rename_consumers(graph, old, new):
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new


def node_by_name(graph, name):
    return next(n for n in graph.node if n.name == name)


def producer_of(graph, tensor):
    return next(n for n in graph.node if n.output and n.output[0] == tensor)


def const_value(graph, name):
    """Value of an initializer or of a Constant node's output."""
    init = get_init(graph, name)
    if init is not None:
        return nh.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            return nh.to_array(node.attribute[0].t)
    return None


# --------------------------------------------------------------------------
# 1. fold alpha into the quantized network
# --------------------------------------------------------------------------
def drop_alpha_branch(graph):
    """Delete Slice(alpha) -> Tile -> DepthToSpace -> Concat."""
    concat = next(n for n in graph.node
                  if n.op_type == "Concat" and OUTPUT_NAME in n.output)
    rgb_tensor = concat.input[0]

    produced_by = {o: n for n in graph.node for o in n.output}
    dead, stack = {concat.name}, [concat.input[1]]
    while stack:
        node = produced_by.get(stack.pop())
        if node is None or node.name in dead:
            continue
        dead.add(node.name)
        stack += [i for i in node.input if i != INPUT_NAME and i in produced_by]

    for node in [n for n in graph.node if n.name in dead]:
        graph.node.remove(node)

    # What the Concat used to take as its RGB input is now the graph output.
    for node in graph.node:
        for i, out in enumerate(node.output):
            if out == rgb_tensor:
                node.output[i] = OUTPUT_NAME


def widen_conv_inputs(graph, conv_names):
    """Pad each conv's weight input-channel axis 3 -> 4 with zeros."""
    for conv_name in conv_names:
        conv = node_by_name(graph, conv_name)
        weight_dq = producer_of(graph, conv.input[1])
        w_name = weight_dq.input[0]
        w = nh.to_array(get_init(graph, w_name))
        if w.shape[1] == CHANNELS:
            continue
        pad = np.zeros((w.shape[0], CHANNELS - w.shape[1]) + w.shape[2:], dtype=w.dtype)
        set_init(graph, w_name, np.concatenate([w, pad], axis=1))


def widen_conv_output(graph, conv_name, extra, bias_value):
    """Append `extra` output channels with zero weights and a constant bias."""
    conv = node_by_name(graph, conv_name)

    weight_dq = producer_of(graph, conv.input[1])
    w_name, w_scale_name, w_zp_name = weight_dq.input[0], weight_dq.input[1], weight_dq.input[2]
    w = nh.to_array(get_init(graph, w_name))
    w_scale = nh.to_array(get_init(graph, w_scale_name))
    w_zp = nh.to_array(get_init(graph, w_zp_name))

    set_init(graph, w_name,
             np.concatenate([w, np.zeros((extra,) + w.shape[1:], dtype=w.dtype)], axis=0))
    # Any non-zero scale is valid for an all-zero filter; the mean keeps the
    # per-channel array in the same numeric range as its neighbours.
    new_w_scale = np.full(extra, w_scale.mean(), dtype=w_scale.dtype)
    set_init(graph, w_scale_name, np.concatenate([w_scale, new_w_scale]))
    set_init(graph, w_zp_name, np.concatenate([w_zp, np.zeros(extra, dtype=w_zp.dtype)]))

    # Bias is int32-quantized with scale = input_scale * weight_scale per
    # channel. Recovering input_scale from the existing pair keeps that
    # relationship exact for the channels being appended, which is what QNN
    # validates the convolution against.
    bias_dq = producer_of(graph, conv.input[2])
    b_name, b_scale_name, b_zp_name = bias_dq.input[0], bias_dq.input[1], bias_dq.input[2]
    b = nh.to_array(get_init(graph, b_name))
    b_scale = nh.to_array(get_init(graph, b_scale_name))
    b_zp = nh.to_array(get_init(graph, b_zp_name))

    input_scale = float(np.mean(b_scale / w_scale))
    new_b_scale = (input_scale * new_w_scale).astype(b_scale.dtype)
    quantized_bias = np.round(bias_value / new_b_scale).astype(b.dtype)

    set_init(graph, b_name, np.concatenate([b, quantized_bias]))
    set_init(graph, b_scale_name, np.concatenate([b_scale, new_b_scale]))
    set_init(graph, b_zp_name, np.concatenate([b_zp, np.zeros(extra, dtype=b_zp.dtype)]))


def qdq_boundary(graph):
    """Express uint8 graph I/O as quantization encodings instead of arithmetic.

    The exports wrap the boundary in raw-uint8 arithmetic:

        image(u8) -> Cast(float) -> Div(255) -> Q(1/255,0) -> DQ -> Conv ...
        ... Clip -> Q(1/255,0) -> DQ -> Mul(255) -> Q -> DQ -> Round -> Cast(u8)

    Cast/Div/Mul/Round sit outside the QDQ graph, so QNN cannot take them
    (backendValidateOpConfig 3110); the graph partitions around them and the NPU
    subgraph then returns near-zero -- while the CPU EP stays perfectly correct,
    which is what makes this so easy to miss.

    Both wrappers are redundant: the head activation scale and the tail Clip
    scale are already exactly 1/255 with zero_point 0, so wiring the existing
    QuantizeLinear/DequantizeLinear pair straight to the graph boundary is an
    exact transformation, and it is what the model that always worked on the NPU
    (quicksrnetlarge-q2rtx-w8a8.onnx) does.
    """
    by_name = {n.name: n for n in graph.node}

    def require(name):
        if name not in by_name:
            raise SystemExit(f"expected a node named '{name}' - export layout changed?")
        return by_name[name]

    # Head: the graph input feeds the DequantizeLinear directly, so QNN sees
    # 'image' itself as the quantized tensor. Anything in between -- the channel
    # Slice included -- leaves the convolution's input with no encoding, and QNN
    # then rejects the Conv itself (3110 on Conv2d).
    head_dq = require("/Div_output_0_DequantizeLinear")
    if abs(float(const_value(graph, head_dq.input[1])) - 1.0 / 255.0) > 1e-6:
        raise SystemExit("input activation scale is not 1/255; the wrapper is not redundant")
    head_dq.input[0] = INPUT_NAME
    for name in ("/Cast", "/Div", "/Div_output_0_QuantizeLinear", "/Slice"):
        graph.node.remove(require(name))

    # Tail: the output QuantizeLinear already produces exactly the uint8 the
    # Mul/Round/Cast chain was re-deriving, so let it be the graph output.
    tail_q = require("/Clip_output_0_QuantizeLinear")
    if abs(float(const_value(graph, tail_q.input[1])) - 1.0 / 255.0) > 1e-6:
        raise SystemExit("output Clip scale is not 1/255; the wrapper is not redundant")
    tail_q.output[0] = OUTPUT_NAME
    for name in ("/Clip_output_0_DequantizeLinear", "/Mul", "/Mul_output_0_QuantizeLinear",
                 "/Mul_output_0_DequantizeLinear", "/Round", "/Cast_1"):
        graph.node.remove(require(name))


def fold_alpha_into_network(graph):
    """Replace the raw-uint8 alpha branch with a 4th channel out of the convs."""
    depth_to_space = next(n for n in graph.node
                          if n.op_type == "DepthToSpace" and n.name.startswith("/model"))
    scale = next(a.i for a in depth_to_space.attribute if a.name == "blocksize")

    # The Clip that consumes DepthToSpace bounds the network output before it is
    # scaled to 0..255, so its maximum is the value alpha has to carry.
    clip = next(n for n in graph.node
                if n.op_type == "Clip" and n.input[0] == depth_to_space.output[0])
    alpha_value = float(const_value(graph, clip.input[2]))

    drop_alpha_branch(graph)

    # One extra output plane, s^2 sub-pixels each. Only the anchor convolution
    # carries the bias -- the two are summed before the Clip.
    extra = scale * scale
    widen_conv_output(graph, "/model/anchor/net/Conv", extra, alpha_value)
    widen_conv_output(graph, "/model/conv_last/Conv", extra, 0.0)

    return scale, alpha_value


# --------------------------------------------------------------------------
# 2. move the declared layout to NHWC
# --------------------------------------------------------------------------
def to_nhwc_io(graph, out_spatial):
    for name, is_input in ((INPUT_NAME, True), (OUTPUT_NAME, False)):
        seq = graph.input if is_input else graph.output
        old = find(seq, name)
        spatial = dims(old)[2:4] if is_input else out_spatial
        inner = name + "_nchw"
        nhwc = [1, spatial[0], spatial[1], CHANNELS]

        if is_input:
            rename_consumers(graph, name, inner)
            seq.remove(old)
            seq.insert(0, helper.make_tensor_value_info(name, TensorProto.UINT8, nhwc))
            graph.node.insert(0, helper.make_node(
                "Transpose", [name], [inner], name="nhwc_to_nchw_in", perm=[0, 3, 1, 2]))
        else:
            for node in graph.node:
                for i, out in enumerate(node.output):
                    if out == name:
                        node.output[i] = inner
            seq.remove(old)
            seq.insert(0, helper.make_tensor_value_info(name, TensorProto.UINT8, nhwc))
            graph.node.append(helper.make_node(
                "Transpose", [inner], [name], name="nchw_to_nhwc_out", perm=[0, 2, 3, 1]))


def convert(path_in, path_out):
    model = onnx.load(path_in)
    graph = model.graph

    src_in, src_out = find(graph.input, INPUT_NAME), find(graph.output, OUTPUT_NAME)
    if src_in is None or src_out is None:
        raise SystemExit(f"expected graph I/O named '{INPUT_NAME}'/'{OUTPUT_NAME}'")

    in_dims, out_dims = dims(src_in), dims(src_out)
    for what, vi, d in (("input", src_in, in_dims), ("output", src_out, out_dims)):
        if vi.type.tensor_type.elem_type != TensorProto.UINT8:
            raise SystemExit(f"{what} '{vi.name}' is not uint8")
        if len(d) != 4 or d[0] != 1 or d[1] != CHANNELS:
            raise SystemExit(
                f"{what} '{vi.name}' is {d}, expected [1, {CHANNELS}, H, W] NCHW "
                "- already converted?")

    scale, alpha = fold_alpha_into_network(graph)
    # With the input Slice gone the first convolutions see all four channels, so
    # they need a fourth (zero) input plane. Zero weights contribute nothing, so
    # RGB is unchanged and the alpha the game feeds in is simply ignored.
    widen_conv_inputs(graph, ["/model/anchor/net/Conv", "/model/cnn/cnn.0/Conv"])
    qdq_boundary(graph)
    to_nhwc_io(graph, out_dims[2:4])

    # Every intermediate still describes the pre-widening 3*s^2 channel counts,
    # which ORT reports as shape-merge conflicts and which leaves the declared
    # output shape disagreeing with what the model actually produces. Drop them
    # and let shape inference rebuild from the real graph.
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    onnx.save(model, path_out)

    print(f"wrote {path_out}  ({scale}x, alpha folded in at {alpha})")
    print(f"  input  {INPUT_NAME}  {dims(find(model.graph.input, INPUT_NAME))}")
    print(f"  output {OUTPUT_NAME} {dims(find(model.graph.output, OUTPUT_NAME))}")
    return model


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------
def verify(path_orig, path_conv, height, width):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3  # the originals are noisy about unused initializers

    rng = np.random.default_rng(0)
    nchw = rng.integers(0, 256, size=(1, CHANNELS, height, width), dtype=np.uint8)

    a = ort.InferenceSession(path_orig, options, providers=["CPUExecutionProvider"]).run(
        None, {INPUT_NAME: nchw})[0].transpose(0, 2, 3, 1)
    b = ort.InferenceSession(path_conv, options, providers=["CPUExecutionProvider"]).run(
        None, {INPUT_NAME: np.ascontiguousarray(nchw.transpose(0, 2, 3, 1))})[0]

    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")
    if not np.array_equal(a[..., :3], b[..., :3]):
        worst = np.abs(a[..., :3].astype(int) - b[..., :3].astype(int)).max()
        raise SystemExit(f"RGB differs at {width}x{height} (max abs {worst})")
    if b[..., 3].min() != 255 or b[..., 3].max() != 255:
        raise SystemExit(f"alpha is not constant 255 at {width}x{height} "
                         f"(min {b[..., 3].min()}, max {b[..., 3].max()})")

    print(f"  verified {width}x{height} -> {b.shape[2]}x{b.shape[1]}: "
          "RGB bit-identical, alpha 255")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--verify", action="store_true",
                    help="compare against the original on the CPU EP (needs onnxruntime)")
    args = ap.parse_args()

    convert(args.input, args.output)

    if args.verify:
        # Odd, non-square, not a multiple of 4: the render extent is whatever
        # viewsize lands on, and nothing pads it.
        for h, w in ((64, 64), (373, 640), (541, 962)):
            verify(args.input, args.output, h, w)

    return 0


if __name__ == "__main__":
    sys.exit(main())
