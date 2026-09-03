// Vision tower: the mmproj `qwen3vl_merger` ViT, run on the Qwen runtime's
// stream with the corpus kernels. Included by v2_runtime.cpp after the model
// and runtime structs; see plans/qwen-vision.md for the computation.
//
// The tower's weights are uploaded once at prepare (bf16 matrices straight
// from the mapping, everything else widened to f32); an image is patched on
// the host, pushed through the blocks, merged 2x2 and projected into the
// language model's width, one f32 row per merged token.

using flyweight::v2::VisionBlockWeights;
using flyweight::v2::VisionConfig;
using flyweight::v2::VisionMatrix;
using flyweight::v2::VisionTowerState;

static const Tensor* vision_find_tensor(const FlyweightV2Model& sidecar, const std::string& name) {
    for (const auto& tensor : sidecar.tensors)
        if (tensor.name == name) return &tensor;
    return nullptr;
}

static const Tensor& vision_require_tensor(const FlyweightV2Model& sidecar, const std::string& name) {
    const Tensor* tensor = vision_find_tensor(sidecar, name);
    if (!tensor) throw std::runtime_error("mmproj is missing tensor " + name);
    return *tensor;
}

static std::uint64_t vision_element_count(const Tensor& tensor) {
    std::uint64_t count = 1;
    for (const auto dim : tensor.shape) count *= dim;
    return count;
}

static std::uint64_t vision_device_alloc(VisionTowerState& tower, std::uint64_t bytes) {
    std::uint64_t pointer = 0;
    if (flyweight_gpu_alloc(bytes, &pointer) != 0 || !pointer)
        throw std::runtime_error("vision tower device allocation failed");
    tower.allocations.push_back(pointer);
    tower.weight_bytes += bytes;
    return pointer;
}

// Any tensor widened to f32 on the device.
static std::uint64_t vision_upload_f32(VisionTowerState& tower, const FlyweightV2Model& sidecar,
                                const Tensor& tensor, std::uint64_t expected_elements) {
    const std::uint64_t elements = vision_element_count(tensor);
    if (elements != expected_elements)
        throw std::runtime_error("mmproj tensor " + tensor.name + " has " +
                                 std::to_string(elements) + " elements, expected " +
                                 std::to_string(expected_elements));
    std::vector<float> widened(elements);
    const auto* source = tensor_data(sidecar, tensor);
    for (std::uint64_t i = 0; i < elements; ++i) widened[i] = tensor_value(source, tensor.type, i);
    const std::uint64_t pointer = vision_device_alloc(tower, elements * sizeof(float));
    if (flyweight_gpu_upload_sync(pointer, widened.data(), elements * sizeof(float)) != 0)
        throw std::runtime_error("vision tower upload failed for " + tensor.name);
    return pointer;
}

// A [rows][columns] matrix: bf16 kept as stored, anything else widened.
static VisionMatrix vision_upload_matrix(VisionTowerState& tower, const FlyweightV2Model& sidecar,
                                  const Tensor& tensor, std::uint64_t rows, std::uint64_t columns) {
    if (tensor.shape.size() != 2 || tensor.shape[0] != columns || tensor.shape[1] != rows)
        throw std::runtime_error("mmproj tensor " + tensor.name + " is not [" +
                                 std::to_string(rows) + " x " + std::to_string(columns) + "]");
    VisionMatrix matrix;
    if (tensor.type == 30) {
        matrix.bf16 = true;
        matrix.device = vision_device_alloc(tower, tensor.size);
        if (flyweight_gpu_upload_sync(matrix.device, tensor_data(sidecar, tensor), tensor.size) != 0)
            throw std::runtime_error("vision tower upload failed for " + tensor.name);
    } else {
        matrix.device = vision_upload_f32(tower, sidecar, tensor, rows * columns);
    }
    return matrix;
}

static void qwen_vision_release(FlyweightV2QwenRuntime& runtime) {
    auto& tower = runtime.vision_tower;
    for (const auto pointer : tower.allocations) flyweight_gpu_free(pointer);
    tower.allocations.clear();
    if (tower.workspace) flyweight_gpu_free(tower.workspace);
    tower = VisionTowerState{};
}

static void qwen_vision_prepare(FlyweightV2QwenRuntime& runtime) {
    const FlyweightV2Model* sidecar = runtime.model->vision_sidecar.get();
    if (!sidecar) return;
    const VisionConfig& config = sidecar->vision;
    flyweight::v2::validate_vision_config(config);
    auto& tower = runtime.vision_tower;
    qwen_vision_release(runtime);
    const std::uint64_t width = config.embedding_length, ffn = config.feed_forward_length;
    const std::uint64_t patch_inputs = 3ull * config.patch_size * config.patch_size;
    const std::uint64_t merged = width * config.spatial_merge_size * config.spatial_merge_size;
    const std::uint64_t projection = config.projection_dim;

    // Patch embedding: the conv over the temporal pair is two [E][3][p][p]
    // slices, and a still image feeds the same pixels to both, so the sum of
    // the slices is the matrix that maps one patch to its embedding.
    {
        const Tensor& first = vision_require_tensor(*sidecar, "v.patch_embd.weight");
        const Tensor* second = vision_find_tensor(*sidecar, "v.patch_embd.weight.1");
        const std::uint64_t elements = width * patch_inputs;
        if (vision_element_count(first) != elements ||
            (second && vision_element_count(*second) != elements))
            throw std::runtime_error("mmproj patch embedding does not match the declared geometry");
        std::vector<float> summed(elements);
        const auto* data = tensor_data(*sidecar, first);
        for (std::uint64_t i = 0; i < elements; ++i) summed[i] = tensor_value(data, first.type, i);
        if (second) {
            const auto* other = tensor_data(*sidecar, *second);
            for (std::uint64_t i = 0; i < elements; ++i) summed[i] += tensor_value(other, second->type, i);
        }
        tower.patch.bf16 = false;
        tower.patch.device = vision_device_alloc(tower, elements * sizeof(float));
        if (flyweight_gpu_upload_sync(tower.patch.device, summed.data(), elements * sizeof(float)) != 0)
            throw std::runtime_error("vision tower upload failed for the patch embedding");
        tower.patch_bias = vision_upload_f32(
            tower, *sidecar, vision_require_tensor(*sidecar, "v.patch_embd.bias"), width);
    }
    // The learned position grid stays on the host: it is resampled per image.
    {
        const Tensor& table = vision_require_tensor(*sidecar, "v.position_embd.weight");
        const std::uint64_t side = config.position_grid_side();
        const std::uint64_t elements = side * side * width;
        if (vision_element_count(table) != elements)
            throw std::runtime_error("mmproj position table does not match image_size / patch_size");
        tower.position_table.resize(elements);
        const auto* data = tensor_data(*sidecar, table);
        for (std::uint64_t i = 0; i < elements; ++i)
            tower.position_table[i] = tensor_value(data, table.type, i);
    }
    tower.blocks.resize(config.block_count);
    for (std::uint32_t index = 0; index < config.block_count; ++index) {
        auto& block = tower.blocks[index];
        const std::string prefix = "v.blk." + std::to_string(index) + ".";
        auto vector = [&](const char* name, std::uint64_t elements) {
            return vision_upload_f32(tower, *sidecar, vision_require_tensor(*sidecar, prefix + name), elements);
        };
        auto matrix = [&](const char* name, std::uint64_t rows, std::uint64_t columns) {
            return vision_upload_matrix(tower, *sidecar, vision_require_tensor(*sidecar, prefix + name), rows, columns);
        };
        block.ln1_weight = vector("ln1.weight", width);
        block.ln1_bias = vector("ln1.bias", width);
        block.ln2_weight = vector("ln2.weight", width);
        block.ln2_bias = vector("ln2.bias", width);
        block.qkv = matrix("attn_qkv.weight", 3 * width, width);
        block.qkv_bias = vector("attn_qkv.bias", 3 * width);
        block.out = matrix("attn_out.weight", width, width);
        block.out_bias = vector("attn_out.bias", width);
        block.up = matrix("ffn_up.weight", ffn, width);
        block.up_bias = vector("ffn_up.bias", ffn);
        block.down = matrix("ffn_down.weight", width, ffn);
        block.down_bias = vector("ffn_down.bias", width);
        const bool deepstack = std::find(config.deepstack_layers.begin(),
                                         config.deepstack_layers.end(), index)
                               != config.deepstack_layers.end();
        if (deepstack) {
            const std::string stack = "v.deepstack." + std::to_string(index) + ".";
            auto stack_vector = [&](const char* name, std::uint64_t elements) {
                return vision_upload_f32(tower, *sidecar, vision_require_tensor(*sidecar, stack + name), elements);
            };
            block.deepstack = true;
            block.deepstack_norm_weight = stack_vector("norm.weight", merged);
            block.deepstack_norm_bias = stack_vector("norm.bias", merged);
            block.deepstack_fc1 = vision_upload_matrix(
                tower, *sidecar, vision_require_tensor(*sidecar, stack + "fc1.weight"), merged, merged);
            block.deepstack_fc1_bias = stack_vector("fc1.bias", merged);
            block.deepstack_fc2 = vision_upload_matrix(
                tower, *sidecar, vision_require_tensor(*sidecar, stack + "fc2.weight"), projection, merged);
            block.deepstack_fc2_bias = stack_vector("fc2.bias", projection);
        }
    }
    tower.post_weight = vision_upload_f32(tower, *sidecar, vision_require_tensor(*sidecar, "v.post_ln.weight"), width);
    tower.post_bias = vision_upload_f32(tower, *sidecar, vision_require_tensor(*sidecar, "v.post_ln.bias"), width);
    tower.merger_fc1 = vision_upload_matrix(tower, *sidecar, vision_require_tensor(*sidecar, "mm.0.weight"), merged, merged);
    tower.merger_fc1_bias = vision_upload_f32(tower, *sidecar, vision_require_tensor(*sidecar, "mm.0.bias"), merged);
    tower.merger_fc2 = vision_upload_matrix(tower, *sidecar, vision_require_tensor(*sidecar, "mm.2.weight"), projection, merged);
    tower.merger_fc2_bias = vision_upload_f32(tower, *sidecar, vision_require_tensor(*sidecar, "mm.2.bias"), projection);
    tower.ready = true;
    if (std::getenv("FLYWEIGHT_VISION_TRACE"))
        std::fprintf(stderr, "[vision] tower ready: %u blocks, %.1f MiB on device%s\n",
                     config.block_count, tower.weight_bytes / 1048576.0,
                     config.deepstack_layers.empty() ? "" : ", deepstack");
}

// Rows the language model receives per image: the merged tokens, each
// carrying `projection_dim` values, followed by one more `projection_dim`
// block per deepstack layer.
static std::uint64_t qwen_vision_row_width(const VisionConfig& config) {
    return static_cast<std::uint64_t>(config.projection_dim) * (1 + config.deepstack_layers.size());
}

// Encode one image. `pixels` is HWC f32, normalized, with sides already
// multiples of the merged patch. Writes `tokens * row_width` floats.
static void qwen_vision_encode(FlyweightV2QwenRuntime& runtime, const float* pixels,
                        std::uint32_t image_width, std::uint32_t image_height,
                        float* output, std::uint64_t output_capacity) {
    auto& tower = runtime.vision_tower;
    if (!tower.ready) throw std::runtime_error("no vision tower is attached to this runtime");
    if (!runtime.stream) throw std::runtime_error("native Qwen runtime is not prepared");
    const VisionConfig& config = runtime.model->vision_sidecar->vision;
    const auto started = std::chrono::steady_clock::now();

    std::vector<float> patches, positions;
    std::vector<std::int32_t> rows_y, rows_x;
    flyweight::v2::vision_patchify(config, pixels, image_width, image_height, patches, rows_y, rows_x);
    const std::uint32_t grid_w = image_width / config.patch_size;
    const std::uint32_t grid_h = image_height / config.patch_size;
    flyweight::v2::vision_position_embeddings(config, tower.position_table.data(), grid_w, grid_h, positions);
    const int n = static_cast<int>(rows_y.size());
    const int merge = static_cast<int>(config.spatial_merge_size * config.spatial_merge_size);
    const int tokens = n / merge;
    const std::uint64_t row_width = qwen_vision_row_width(config);
    if (output_capacity < static_cast<std::uint64_t>(tokens) * row_width)
        throw std::runtime_error("vision output buffer is too small");

    const int width = static_cast<int>(config.embedding_length);
    const int ffn = static_cast<int>(config.feed_forward_length);
    const int patch_inputs = static_cast<int>(3 * config.patch_size * config.patch_size);
    const int merged = width * merge;
    const int projection = static_cast<int>(config.projection_dim);
    const int heads = static_cast<int>(config.head_count);
    const int head_dim = static_cast<int>(config.head_dim());
    const int deepstack_count = static_cast<int>(config.deepstack_layers.size());

    // Workspace layout, f32 unless noted.
    auto bytes_of = [](std::uint64_t elements) { return elements * sizeof(float); };
    const std::uint64_t n64 = static_cast<std::uint64_t>(n), t64 = static_cast<std::uint64_t>(tokens);
    std::uint64_t cursor = 0;
    auto reserve = [&](std::uint64_t bytes) {
        const std::uint64_t at = cursor;
        cursor += (bytes + 255) & ~static_cast<std::uint64_t>(255);
        return at;
    };
    const std::uint64_t off_patches = reserve(bytes_of(n64 * patch_inputs));
    const std::uint64_t off_positions = reserve(bytes_of(n64 * width));
    const std::uint64_t off_hidden = reserve(bytes_of(n64 * width));
    const std::uint64_t off_normed = reserve(bytes_of(n64 * width));
    const std::uint64_t off_qkv = reserve(bytes_of(n64 * 3 * width));
    const std::uint64_t off_attn = reserve(bytes_of(n64 * width));
    const std::uint64_t off_mlp = reserve(bytes_of(n64 * ffn));
    const std::uint64_t off_merge = reserve(bytes_of(t64 * merged));
    const std::uint64_t off_output = reserve(bytes_of(t64 * projection));
    const std::uint64_t off_rows_y = reserve(n64 * sizeof(std::int32_t));
    const std::uint64_t off_rows_x = reserve(n64 * sizeof(std::int32_t));
    if (cursor > tower.workspace_bytes) {
        if (tower.workspace) flyweight_gpu_free(tower.workspace);
        tower.workspace = 0;
        tower.workspace_bytes = 0;
        if (flyweight_gpu_alloc(cursor, &tower.workspace) != 0 || !tower.workspace)
            throw std::runtime_error("vision workspace allocation failed (" +
                                     std::to_string(cursor / 1048576) + " MiB)");
        tower.workspace_bytes = cursor;
    }
    const std::uint64_t base = tower.workspace;
    const std::uint64_t d_patches = base + off_patches, d_positions = base + off_positions;
    const std::uint64_t d_hidden = base + off_hidden, d_normed = base + off_normed;
    const std::uint64_t d_qkv = base + off_qkv, d_attn = base + off_attn, d_mlp = base + off_mlp;
    const std::uint64_t d_merge = base + off_merge, d_output = base + off_output;
    const std::uint64_t d_rows_y = base + off_rows_y, d_rows_x = base + off_rows_x;

    auto upload = [&](std::uint64_t destination, const void* source, std::uint64_t bytes) {
        if (flyweight_gpu_upload(destination, source, bytes, runtime.stream) != 0)
            throw std::runtime_error("vision workspace upload failed");
    };
    upload(d_patches, patches.data(), bytes_of(n64 * patch_inputs));
    upload(d_positions, positions.data(), bytes_of(n64 * width));
    upload(d_rows_y, rows_y.data(), n64 * sizeof(std::int32_t));
    upload(d_rows_x, rows_x.data(), n64 * sizeof(std::int32_t));

    auto launch = [&](const char* name, std::uint32_t grid_x, std::uint32_t grid_y, void** arguments) {
        if (flyweight_gpu_launch_named(name, grid_x, grid_y, 256, 0, runtime.stream, arguments) != 0)
            throw std::runtime_error(std::string("vision CUDA kernel failed: ") + name);
    };
    auto gemm = [&](const VisionMatrix& matrix, std::uint64_t bias, std::uint64_t input,
                    std::uint64_t out, int input_size, int output_size, int rows) {
        void* arguments[] = {const_cast<std::uint64_t*>(&matrix.device), &bias, &input, &out,
                             &input_size, &output_size, &rows};
        launch(matrix.bf16 ? "vision_bf16_gemm_rows" : "vision_f32_gemm_rows",
               static_cast<std::uint32_t>((output_size + 63) / 64),
               static_cast<std::uint32_t>((rows + 63) / 64), arguments);
    };
    auto layer_norm = [&](std::uint64_t input, std::uint64_t weight, std::uint64_t bias,
                          std::uint64_t out, int norm_width, int rows) {
        float epsilon = config.layer_norm_epsilon;
        void* arguments[] = {&input, &weight, &bias, &out, &norm_width, &rows, &epsilon};
        launch("vision_layer_norm_rows", static_cast<std::uint32_t>(rows), 1, arguments);
    };
    auto elementwise_grid = [](std::uint64_t elements) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>((elements + 255) / 256, 4096));
    };
    auto add = [&](std::uint64_t x, std::uint64_t y, std::uint64_t elements) {
        int count = static_cast<int>(elements);
        void* arguments[] = {&x, &y, &count};
        launch("vision_add_rows", elementwise_grid(elements), 1, arguments);
    };
    auto gelu = [&](std::uint64_t x, std::uint64_t elements) {
        int count = static_cast<int>(elements);
        void* arguments[] = {&x, &count};
        launch("vision_gelu_rows", elementwise_grid(elements), 1, arguments);
    };
    // The deepstack projection of a block output viewed as merged rows,
    // landing in the output's extra column block for that layer.
    auto deepstack = [&](const VisionBlockWeights& block, int slot) {
        layer_norm(d_hidden, block.deepstack_norm_weight, block.deepstack_norm_bias, d_normed, merged, tokens);
        gemm(block.deepstack_fc1, block.deepstack_fc1_bias, d_normed, d_merge, merged, merged, tokens);
        gelu(d_merge, t64 * merged);
        // fc2 writes a [tokens][projection] block into the output staging
        // buffer, which is then scattered into the wide output rows.
        gemm(block.deepstack_fc2, block.deepstack_fc2_bias, d_merge, d_output, merged, projection, tokens);
        std::vector<float> staged(t64 * projection);
        if (flyweight_gpu_download(staged.data(), d_output, bytes_of(t64 * projection), runtime.stream) != 0 ||
            flyweight_gpu_stream_sync(runtime.stream) != 0)
            throw std::runtime_error("vision deepstack download failed");
        for (int token = 0; token < tokens; ++token)
            std::memcpy(output + static_cast<std::uint64_t>(token) * row_width +
                            static_cast<std::uint64_t>(1 + slot) * projection,
                        staged.data() + static_cast<std::uint64_t>(token) * projection,
                        bytes_of(projection));
    };

    gemm(tower.patch, tower.patch_bias, d_patches, d_hidden, patch_inputs, width, n);
    add(d_hidden, d_positions, n64 * width);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float rope_theta = 10000.0f;
    int deepstack_slot = 0;
    for (std::uint32_t index = 0; index < config.block_count; ++index) {
        const auto& block = tower.blocks[index];
        layer_norm(d_hidden, block.ln1_weight, block.ln1_bias, d_normed, width, n);
        gemm(block.qkv, block.qkv_bias, d_normed, d_qkv, width, 3 * width, n);
        {
            std::uint64_t qkv = d_qkv, y = d_rows_y, x = d_rows_x;
            int heads_arg = heads, head_dim_arg = head_dim, rows = n;
            float theta = rope_theta;
            void* arguments[] = {&qkv, &y, &x, &heads_arg, &head_dim_arg, &rows, &theta};
            launch("vision_rope_rows", static_cast<std::uint32_t>(n), 2, arguments);
        }
        {
            std::uint64_t qkv = d_qkv, out = d_attn;
            int heads_arg = heads, head_dim_arg = head_dim, rows = n;
            float scale_arg = scale;
            void* arguments[] = {&qkv, &out, &heads_arg, &head_dim_arg, &rows, &scale_arg};
            launch("vision_attention_rows", static_cast<std::uint32_t>((n + 31) / 32),
                   static_cast<std::uint32_t>(heads), arguments);
        }
        gemm(block.out, block.out_bias, d_attn, d_normed, width, width, n);
        add(d_hidden, d_normed, n64 * width);
        layer_norm(d_hidden, block.ln2_weight, block.ln2_bias, d_normed, width, n);
        gemm(block.up, block.up_bias, d_normed, d_mlp, width, ffn, n);
        gelu(d_mlp, n64 * ffn);
        gemm(block.down, block.down_bias, d_mlp, d_normed, ffn, width, n);
        add(d_hidden, d_normed, n64 * width);
        if (block.deepstack) deepstack(block, deepstack_slot++);
    }
    layer_norm(d_hidden, tower.post_weight, tower.post_bias, d_normed, width, n);
    // d_normed as [tokens][merged]: four consecutive patch rows per token.
    gemm(tower.merger_fc1, tower.merger_fc1_bias, d_normed, d_merge, merged, merged, tokens);
    gelu(d_merge, t64 * merged);
    gemm(tower.merger_fc2, tower.merger_fc2_bias, d_merge, d_output, merged, projection, tokens);
    std::vector<float> projected(t64 * projection);
    if (flyweight_gpu_download(projected.data(), d_output, bytes_of(t64 * projection), runtime.stream) != 0 ||
        flyweight_gpu_stream_sync(runtime.stream) != 0)
        throw std::runtime_error("vision output download failed");
    for (int token = 0; token < tokens; ++token)
        std::memcpy(output + static_cast<std::uint64_t>(token) * row_width,
                    projected.data() + static_cast<std::uint64_t>(token) * projection,
                    bytes_of(projection));
    (void)deepstack_count;
    if (std::getenv("FLYWEIGHT_VISION_TRACE"))
        std::fprintf(stderr, "[vision] %ux%u -> %d patches, %d tokens in %.1f ms\n",
                     image_width, image_height, n, tokens,
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - started).count());
}

// Lay out a task's images: spans (which only need the geometry) for prefix
// matching and rope, and one device slot per image that is filled the first
// time a prefill touches it. An image inside a reused prefix therefore never
// runs the tower again.
static void qwen_task_encode_images(FlyweightV2QwenRuntime& runtime, QwenEngineTask& task) {
    auto& encoded = task.encoded;
    if (encoded.encoded || task.images.empty()) return;
    if (!runtime.vision_tower.ready)
        throw std::runtime_error("the prompt carries images but no vision tower is attached");
    const VisionConfig& config = runtime.model->vision_sidecar->vision;
    const std::uint64_t hidden = config.projection_dim;
    encoded.planes = static_cast<std::uint32_t>(qwen_vision_row_width(config) / hidden);
    std::sort(task.images.begin(), task.images.end(),
              [](const QwenTaskImage& a, const QwenTaskImage& b) { return a.token_offset < b.token_offset; });
    encoded.spans.clear();
    encoded.offsets.assign(task.images.size(), 0);
    encoded.devices.assign(task.images.size(), 0);
    for (const auto& image : task.images) {
        const auto resize = flyweight::v2::vision_smart_resize(
            config, image.width, image.height, 1, 1u << 20);
        if (resize.width != image.width || resize.height != image.height)
            throw std::runtime_error("image sides must be multiples of the merged patch");
        QwenImageSpan span;
        span.start = image.token_offset;
        span.count = resize.tokens;
        span.grid_h = resize.grid_h / config.spatial_merge_size;
        span.grid_w = resize.grid_w / config.spatial_merge_size;
        span.hash = image.hash;
        if (span.start + span.count > task.prompt.size())
            throw std::runtime_error("image tokens run past the end of the prompt");
        if (!encoded.spans.empty() &&
            span.start < encoded.spans.back().start + encoded.spans.back().count)
            throw std::runtime_error("image token spans overlap");
        encoded.spans.push_back(span);
    }
    encoded.encoded = true;
}

// Run the tower over image `k` of the task and park its rows on the device,
// plane-major: `planes` blocks of [count][hidden].
static void qwen_task_image_upload(FlyweightV2QwenRuntime& runtime, QwenTaskImages& encoded,
                                   QwenTaskImage& image, std::size_t k) {
    if (encoded.devices[k]) return;
    const VisionConfig& config = runtime.model->vision_sidecar->vision;
    const std::uint64_t hidden = config.projection_dim;
    const std::uint64_t row_width = qwen_vision_row_width(config);
    const auto& span = encoded.spans[k];
    std::vector<float> rows(static_cast<std::size_t>(span.count) * row_width);
    qwen_vision_encode(runtime, image.pixels.data(), image.width, image.height,
                       rows.data(), rows.size());
    std::vector<float> staged(rows.size());
    for (std::uint32_t plane = 0; plane < encoded.planes; ++plane)
        for (std::uint32_t token = 0; token < span.count; ++token)
            std::memcpy(staged.data() + (static_cast<std::size_t>(plane) * span.count + token) * hidden,
                        rows.data() + static_cast<std::size_t>(token) * row_width + plane * hidden,
                        hidden * sizeof(float));
    std::uint64_t device = 0;
    if (flyweight_gpu_alloc(staged.size() * sizeof(float), &device) != 0 || !device)
        throw std::runtime_error("image embedding device allocation failed");
    if (flyweight_gpu_upload_sync(device, staged.data(), staged.size() * sizeof(float)) != 0) {
        flyweight_gpu_free(device);
        throw std::runtime_error("image embedding upload failed");
    }
    encoded.devices[k] = device;
    std::vector<float>().swap(image.pixels);
}

static void qwen_task_free_images(QwenEngineTask& task) {
    for (auto device : task.encoded.devices) if (device) flyweight_gpu_free(device);
    task.encoded.devices.clear();
    task.encoded.encoded = false;
    task.encoded.spans.clear();
    task.encoded.offsets.clear();
}

// The rows [first, first + rows) just embedded into `hidden_rows`: replace
// the ones inside the prefilling task's image spans with the encoded rows
// of `plane` (plane 0 is the projected embedding; `add` accumulates a
// deepstack plane instead of overwriting). Encodes an image on first touch.
static void qwen_image_rows_apply(FlyweightV2QwenRuntime& runtime, std::uint64_t first, int rows,
                                  std::uint64_t hidden_rows, int hidden_size,
                                  std::uint32_t plane, bool add) {
    QwenEngineTask* task = runtime.active_task;
    if (!task || !task->encoded.encoded || rows <= 0) return;
    auto& images = task->encoded;
    const std::uint64_t last = first + static_cast<std::uint64_t>(rows);
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(hidden_size) * sizeof(float);
    for (std::size_t k = 0; k < images.spans.size(); ++k) {
        const auto& span = images.spans[k];
        const std::uint64_t begin = std::max(first, span.start);
        const std::uint64_t end = std::min(last, span.start + span.count);
        if (begin >= end) continue;
        qwen_task_image_upload(runtime, images, task->images[k], k);
        const std::uint64_t source = images.devices[k] +
            ((static_cast<std::uint64_t>(plane) * span.count + (begin - span.start)) *
             static_cast<std::uint64_t>(hidden_size)) * sizeof(float);
        const std::uint64_t target = hidden_rows + (begin - first) * row_bytes;
        const std::uint64_t count = end - begin;
        if (add) {
            int elements = static_cast<int>(count * static_cast<std::uint64_t>(hidden_size));
            std::uint64_t x = target, y = source;
            void* arguments[] = {&x, &y, &elements};
            if (flyweight_gpu_launch_named("vision_add_rows",
                    static_cast<std::uint32_t>(std::min<std::uint64_t>((elements + 255) / 256, 4096)),
                    1, 256, 0, runtime.stream, arguments) != 0)
                throw std::runtime_error("image row accumulate failed");
        } else if (flyweight_gpu_copy_device(target, source, count * row_bytes, runtime.stream) != 0) {
            throw std::runtime_error("image row overlay failed");
        }
    }
}

int flyweight_v2_model_attach_vision(FlyweightV2Model* m, const char* path) { return guarded([&]{
    if (!m || !path || !path[0]) throw std::runtime_error("model and mmproj path are required");
    if (m->vision_sidecar) throw std::runtime_error("a vision tower is already attached");
    if (m->config.architecture.find("qwen") != 0)
        throw std::runtime_error("vision towers attach to Qwen checkpoints only");
    FlyweightV2Model* opened = nullptr;
    if (flyweight_v2_model_open(path, &opened) != 0)
        throw std::runtime_error(error.empty() ? "failed to open mmproj" : error);
    std::unique_ptr<FlyweightV2Model> sidecar(opened);
    if (sidecar->architecture != "clip")
        throw std::runtime_error("mmproj architecture is '" + sidecar->architecture + "', not clip");
    flyweight::v2::validate_vision_config(sidecar->vision);
    if (!sidecar->vision.deepstack_layers.empty())
        throw std::runtime_error("mmproj uses deepstack layers, which this runtime does not "
                                 "inject into the decoder yet");
    if (sidecar->vision.projection_dim != m->config.hidden_size)
        throw std::runtime_error("mmproj projects to " + std::to_string(sidecar->vision.projection_dim) +
                                 " but the model is " + std::to_string(m->config.hidden_size) + " wide");
    // Every tensor the tower will read, checked now so a truncated or foreign
    // mmproj fails at attach rather than at the first image.
    vision_require_tensor(*sidecar, "v.patch_embd.weight");
    vision_require_tensor(*sidecar, "v.position_embd.weight");
    vision_require_tensor(*sidecar, "mm.0.weight");
    vision_require_tensor(*sidecar, "mm.2.weight");
    for (std::uint32_t index = 0; index < sidecar->vision.block_count; ++index)
        vision_require_tensor(*sidecar, "v.blk." + std::to_string(index) + ".attn_qkv.weight");
    m->vision_sidecar = std::move(sidecar);
    return 0;
}); }

int flyweight_v2_vision_info(const FlyweightV2Model* m, FlyweightV2VisionInfo* out) { return guarded([&]{
    if (!m || !out) throw std::runtime_error("invalid vision info arguments");
    std::memset(out, 0, sizeof(*out));
    if (!m->vision_sidecar) return 0;
    const auto& config = m->vision_sidecar->vision;
    out->attached = 1;
    out->patch_size = config.patch_size;
    out->spatial_merge_size = config.spatial_merge_size;
    out->embedding_length = config.embedding_length;
    out->projection_dim = config.projection_dim;
    out->block_count = config.block_count;
    out->deepstack_layers = static_cast<uint32_t>(config.deepstack_layers.size());
    out->row_width = qwen_vision_row_width(config);
    for (int c = 0; c < 3; ++c) {
        out->image_mean[c] = config.image_mean[c];
        out->image_std[c] = config.image_std[c];
    }
    return 0;
}); }

int flyweight_v2_vision_resize(const FlyweightV2Model* m, uint32_t width, uint32_t height,
                               uint32_t min_tokens, uint32_t max_tokens,
                               FlyweightV2VisionResize* out) { return guarded([&]{
    if (!m || !out) throw std::runtime_error("invalid vision resize arguments");
    if (!m->vision_sidecar) throw std::runtime_error("no vision tower is attached");
    const auto resize = flyweight::v2::vision_smart_resize(
        m->vision_sidecar->vision, width, height, min_tokens ? min_tokens : 1,
        max_tokens ? max_tokens : 4096);
    out->width = resize.width;
    out->height = resize.height;
    out->grid_w = resize.grid_w / m->vision_sidecar->vision.spatial_merge_size;
    out->grid_h = resize.grid_h / m->vision_sidecar->vision.spatial_merge_size;
    out->tokens = resize.tokens;
    return 0;
}); }

int flyweight_v2_qwen_vision_encode(FlyweightV2QwenRuntime* runtime, const float* pixels,
                                    uint32_t width, uint32_t height, float* output,
                                    uint64_t capacity) { return guarded([&]{
    if (!runtime || !pixels || !output) throw std::runtime_error("invalid vision encode arguments");
    qwen_vision_encode(*runtime, pixels, width, height, output, capacity);
    return 0;
}); }

