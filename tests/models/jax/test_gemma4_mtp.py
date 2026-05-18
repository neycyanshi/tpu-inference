# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import get_model_loader

from tpu_inference.distributed.jax_parallel_state import \
    init_pp_distributed_environment
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.pp_utils import PPMissingLayer
from tpu_inference.models.jax.gemma4_mtp import (Gemma4MTPDecoderLayer,
                                                 Gemma4MTPForCausalLM)


class TestGemma4MTPForCausalLM:

    @pytest.mark.parametrize("model_name", [
        "google/gemma-4-31B-it",
    ])
    @pytest.mark.parametrize("pp_rank,pp_world_size", [(0, 1), (0, 4), (1, 4),
                                                       (3, 4)])
    @pytest.mark.parametrize(
        "load_format", ["skip_layers_model_loader_for_test", "jax_dummy"])
    @pytest.mark.parametrize("use_ordered_embeddings", [True, False])
    def test_model_loading(
            self,
            model_name,
            pp_rank,
            pp_world_size,
            load_format,
            use_ordered_embeddings,
            # following are defined in conftest.py
            rng,
            mesh,
            mock_vllm_config):
        """Tests loading weights and running forward pass of the MTP model following test_gemma4.py"""
        kv_cache_type = "auto"
        vllm_config = mock_vllm_config(model_name, kv_cache_type)

        # Lightweight config for target/verifier layers
        vllm_config.model_config.hf_config.text_config.num_hidden_layers = 4
        vllm_config.load_config.load_format = load_format
        vllm_config.load_config.num_layers_to_load_for_test = 4
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.enable_expert_parallel = False

        # For HF loader testing, we redirect the model to point to the real assistant draft checkpoint
        if load_format == "skip_layers_model_loader_for_test":
            vllm_config.model_config.model = "google/gemma-4-31B-it-assistant"

        # Construct Speculative Draft Config matching real draft checkpoint dimensions
        vllm_config.speculative_config = MagicMock()
        draft_model_config = MagicMock()

        draft_hf_config = MagicMock()
        draft_hf_config.text_config.hidden_size = 1024
        draft_hf_config.text_config.vocab_size = vllm_config.model_config.get_vocab_size(
        )
        draft_hf_config.text_config.num_hidden_layers = 4
        draft_hf_config.text_config.rms_norm_eps = 1e-6
        draft_hf_config.text_config.layer_types = ["full_attention"] * 4
        draft_hf_config.text_config.rope_theta = 10000.0
        draft_hf_config.text_config.rope_scaling = None
        draft_hf_config.text_config.head_dim = 256
        draft_hf_config.text_config.global_head_dim = 256
        draft_hf_config.text_config.num_attention_heads = 32
        draft_hf_config.text_config.num_key_value_heads = 2
        draft_hf_config.text_config.attention_bias = False
        draft_hf_config.text_config.intermediate_size = 8192
        draft_hf_config.text_config.final_logit_softcapping = 30.0
        draft_hf_config.backbone_hidden_size = vllm_config.model_config.get_hidden_size(
        )
        draft_hf_config.tie_word_embeddings = True
        draft_hf_config.use_ordered_embeddings = use_ordered_embeddings
        draft_hf_config.num_centroids = 32
        draft_hf_config.centroid_intermediate_top_k = 4

        draft_model_config.hf_config = draft_hf_config
        draft_model_config.get_hidden_size = lambda: 1024
        vllm_config.speculative_config.draft_model_config = draft_model_config

        # Initialize Pipeline Parallel group
        init_pp_distributed_environment(
            ip="",
            rank=pp_rank,
            world_size=pp_world_size,
            device=jax.devices()[0],
            need_pp=False,
        )

        model_config = vllm_config.model_config
        kv_dtype = jnp.bfloat16

        with jax.set_mesh(mesh):
            model = Gemma4MTPForCausalLM(vllm_config, rng, mesh)

        # Load weights (monitoring or simple call)
        with jax.set_mesh(mesh):
            loader = get_model_loader(vllm_config.load_config)
            with set_current_vllm_config(vllm_config):
                loader.load_weights(model, model_config)

        # Validate layer counts and partitioning
        assert model.model is not None
        assert len(model.model.layers) == 4

        # Fetch the active MTP layer index on this pipeline parallel rank
        start_layer_idx = model.model.start_layer
        end_layer_idx = model.model.end_layer

        if start_layer_idx < end_layer_idx:
            # Verify that the active layer is loaded
            layer_0: Gemma4MTPDecoderLayer = model.model.layers[
                start_layer_idx]
            assert not isinstance(layer_0, PPMissingLayer)

            num_key_value_heads = layer_0.self_attn.num_kv_heads
            qk_head_dim = layer_0.self_attn.head_dim_original

            # Run forward pass on active layer
            seq_len = 2
            input_tensor = jnp.ones(
                (seq_len, draft_hf_config.text_config.hidden_size),
                dtype=jnp.bfloat16)

            block_size = 16
            num_blocks = 8
            cache_shape = get_kv_cache_shape(num_blocks, block_size,
                                             num_key_value_heads, qk_head_dim,
                                             kv_dtype)

            # Populate centroids ordering if enabled to avoid sparse projection crashes
            if use_ordered_embeddings and model.masked_embedding is not None:
                model.masked_embedding.token_ordering.value = jnp.arange(
                    draft_hf_config.text_config.vocab_size, dtype=jnp.int32)

            with jax.set_mesh(mesh):
                _, jax_output, _ = layer_0(
                    kv_cache=jnp.zeros(cache_shape, dtype=kv_dtype),
                    x=input_tensor,
                    attention_metadata=AttentionMetadata(
                        input_positions=jnp.arange(seq_len),
                        block_tables=jnp.array(list(range(1))),
                        seq_lens=jnp.array([seq_len]),
                        query_start_loc=jnp.array([0, seq_len]),
                        request_distribution=jnp.array([0, 0, 1]),
                    ),
                )
            assert jax_output is not None
        else:
            # Verify that all layers are missing on this rank (PPMissingLayer)
            for idx in range(4):
                assert isinstance(model.model.layers[idx], PPMissingLayer)
