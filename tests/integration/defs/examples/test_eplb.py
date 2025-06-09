import os
import tempfile
from pathlib import Path

import pytest
import yaml

from tensorrt_llm._torch import LLM
from tensorrt_llm._torch.pyexecutor.config import MoeLoadBalancerConfig
from tensorrt_llm.llmapi import KvCacheConfig

from ..accuracy.accuracy_core import GSM8K, MMLU
from ..conftest import llm_models_root


class TestEPLB:
    """Expert Parallelism Load Balancer (EPLB) end-to-end tests"""

    MODEL_NAME = "deepseek-ai/DeepSeek-V3-Lite"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_device_not_contain(["H100", "H200"])
    def test_eplb_e2e_workflow(self):
        """
        End-to-end EPLB workflow test:
        1. Collect expert statistics
        2. Generate EPLB configuration
        3. Run inference with EPLB configuration
        """
        # Setup temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            expert_statistic_path = Path(temp_dir) / "expert_statistic"
            expert_statistic_path.mkdir()
            config_path = Path(temp_dir) / "moe_load_balancer.yaml"

            # Stage 1: Collect expert statistics
            self._collect_expert_statistics(expert_statistic_path)

            # Stage 2: Generate EPLB configuration
            self._generate_eplb_config(expert_statistic_path, config_path)

            # Stage 3: Run inference with EPLB configuration
            self._run_inference_with_eplb(config_path)

    def _collect_expert_statistics(self, expert_statistic_path: Path):
        """Collect expert statistics"""
        # Set environment variables to enable expert statistics collection
        os.environ["EXPERT_STATISTIC_PATH"] = str(expert_statistic_path)
        os.environ["EXPERT_STATISTIC_ITER_RANGE"] = "5-15"

        try:
            kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
            pytorch_backend_options = dict(use_cuda_graph=True,
                                           enable_attention_dp=True)

            # Run inference to collect statistics
            llm = LLM(self.MODEL_PATH,
                      tensor_parallel_size=4,
                      moe_expert_parallel_size=4,
                      kv_cache_config=kv_cache_config,
                      **pytorch_backend_options)

            with llm:
                # Use GSM8K task to collect statistics
                task = GSM8K(self.MODEL_NAME)
                # Run only a small number of samples to collect statistics
                task.evaluate(llm, extra_evaluator_kwargs={"num_samples": 20})

            # Verify that statistic files are generated
            assert expert_statistic_path.exists()
            statistic_files = list(
                expert_statistic_path.glob("rank*.safetensors"))
            assert len(statistic_files
                       ) > 0, "Expert statistic files should be generated"

            # Verify that meta info file exists
            meta_info_file = expert_statistic_path / "meta_info.json"
            assert meta_info_file.exists(
            ), "meta_info.json file should be generated"

        finally:
            # Clean up environment variables
            if "EXPERT_STATISTIC_PATH" in os.environ:
                del os.environ["EXPERT_STATISTIC_PATH"]
            if "EXPERT_STATISTIC_ITER_RANGE" in os.environ:
                del os.environ["EXPERT_STATISTIC_ITER_RANGE"]

    def _generate_eplb_config(self, expert_statistic_path: Path,
                              config_path: Path):
        """Generate EPLB configuration"""
        import subprocess

        # Path to the generate_eplb_config.py script
        script_path = Path(
            __file__
        ).parent.parent.parent.parent.parent / "examples" / "ep_load_balancer" / "generate_eplb_config.py"

        # Run the generate_eplb_config.py script
        cmd = [
            "python",
            str(script_path), "--expert_statistic_path",
            str(expert_statistic_path), "--output_path",
            str(config_path), "--ep_size", "4", "--num_slots", "80",
            "--layer_updates_per_iter", "0"
        ]

        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                cwd=str(expert_statistic_path.parent))

        if result.returncode != 0:
            raise RuntimeError(
                f"generate_eplb_config.py failed with return code {result.returncode}. "
                f"stdout: {result.stdout}, stderr: {result.stderr}")

        assert config_path.exists(
        ), "EPLB configuration file should be generated"

    def _run_inference_with_eplb(self, config_path: Path):
        """Run inference with EPLB configuration"""
        # Read EPLB configuration
        with open(config_path, "r") as f:
            eplb_config_dict = yaml.safe_load(f)

        eplb_config = MoeLoadBalancerConfig(**eplb_config_dict)

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.9)
        pytorch_backend_options = dict(use_cuda_graph=True,
                                       enable_attention_dp=True,
                                       moe_load_balancer=eplb_config)

        # Run inference with EPLB configuration
        llm = LLM(self.MODEL_PATH,
                  tensor_parallel_size=4,
                  moe_expert_parallel_size=4,
                  kv_cache_config=kv_cache_config,
                  **pytorch_backend_options)

        with llm:
            # Verify EPLB configuration is loaded correctly
            assert llm.args.moe_load_balancer is not None
            assert isinstance(llm.args.moe_load_balancer, MoeLoadBalancerConfig)
            assert llm.args.moe_load_balancer.num_slots == eplb_config.num_slots

            # Run evaluation tasks
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)

            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
