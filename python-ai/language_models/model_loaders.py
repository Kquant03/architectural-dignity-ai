"""
Model Loaders Module
Handles loading, configuration, and optimization of language models
with support for multiple providers and hardware-aware optimization.
"""

import os
import gc
import json
import torch
import psutil
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from abc import ABC, abstractmethod

# Model provider clients
import anthropic
import openai
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer
)


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading and optimization"""
    provider: str  # anthropic, openai, local
    model_name: str
    api_key: Optional[str] = None
    
    # Local model settings
    model_path: Optional[str] = None
    device: str = "cuda"
    dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention: bool = True
    
    # Memory optimization
    max_memory: Optional[Dict[int, str]] = None
    offload_folder: Optional[str] = None
    offload_state_dict: bool = False
    
    # Generation settings
    max_length: int = 4096
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Streaming settings
    stream: bool = True
    stream_interval: int = 2
    
    # Hardware optimization
    use_triton: bool = False
    use_xformers: bool = False
    gradient_checkpointing: bool = False
    
    # Consciousness integration
    consciousness_aware: bool = True
    emotional_modulation: bool = True


@dataclass
class HardwareProfile:
    """Hardware capabilities profile"""
    device_name: str
    total_memory: int  # in MB
    available_memory: int
    compute_capability: Tuple[int, int]
    multi_gpu: bool = False
    num_gpus: int = 1
    
    @classmethod
    def detect(cls) -> 'HardwareProfile':
        """Detect current hardware capabilities"""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            available_memory = torch.cuda.memory_reserved(0) // (1024 * 1024)
            compute_capability = torch.cuda.get_device_capability(0)
            num_gpus = torch.cuda.device_count()
            
            return cls(
                device_name=device_name,
                total_memory=total_memory,
                available_memory=available_memory,
                compute_capability=compute_capability,
                multi_gpu=num_gpus > 1,
                num_gpus=num_gpus
            )
        else:
            # CPU fallback
            return cls(
                device_name="CPU",
                total_memory=psutil.virtual_memory().total // (1024 * 1024),
                available_memory=psutil.virtual_memory().available // (1024 * 1024),
                compute_capability=(0, 0),
                multi_gpu=False,
                num_gpus=0
            )


class ModelLoader(ABC):
    """Abstract base class for model loaders"""
    
    @abstractmethod
    async def load_model(self, config: ModelConfig) -> Any:
        """Load model with given configuration"""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def stream_generate(self, prompt: str, **kwargs):
        """Stream generation of text"""
        pass


class APIModelLoader(ModelLoader):
    """Loader for API-based models (Anthropic, OpenAI)"""
    
    def __init__(self):
        self.client = None
        self.config = None
    
    async def load_model(self, config: ModelConfig) -> Any:
        """Initialize API client"""
        self.config = config
        
        if config.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=config.api_key)
        elif config.provider == "openai":
            self.client = openai.OpenAI(api_key=config.api_key)
        else:
            raise ValueError(f"Unknown API provider: {config.provider}")
        
        logger.info(f"Initialized {config.provider} API client for {config.model_name}")
        return self.client
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using API"""
        if self.config.provider == "anthropic":
            response = await self._generate_anthropic(prompt, **kwargs)
        elif self.config.provider == "openai":
            response = await self._generate_openai(prompt, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")
        
        return response
    
    async def stream_generate(self, prompt: str, **kwargs):
        """Stream generation using API"""
        if self.config.provider == "anthropic":
            async for chunk in self._stream_anthropic(prompt, **kwargs):
                yield chunk
        elif self.config.provider == "openai":
            async for chunk in self._stream_openai(prompt, **kwargs):
                yield chunk
    
    async def _generate_anthropic(self, prompt: str, **kwargs) -> str:
        """Generate using Anthropic API"""
        messages = kwargs.get("messages", [{"role": "user", "content": prompt}])
        
        response = self.client.messages.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_length),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p)
        )
        
        return response.content[0].text
    
    async def _generate_openai(self, prompt: str, **kwargs) -> str:
        """Generate using OpenAI API"""
        messages = kwargs.get("messages", [{"role": "user", "content": prompt}])
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_length),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p)
        )
        
        return response.choices[0].message.content
    
    async def _stream_anthropic(self, prompt: str, **kwargs):
        """Stream using Anthropic API"""
        messages = kwargs.get("messages", [{"role": "user", "content": prompt}])
        
        stream = self.client.messages.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_length),
            temperature=kwargs.get("temperature", self.config.temperature),
            stream=True
        )
        
        for chunk in stream:
            if chunk.type == "content_block_delta":
                yield chunk.delta.text
    
    async def _stream_openai(self, prompt: str, **kwargs):
        """Stream using OpenAI API"""
        messages = kwargs.get("messages", [{"role": "user", "content": prompt}])
        
        stream = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_length),
            temperature=kwargs.get("temperature", self.config.temperature),
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LocalModelLoader(ModelLoader):
    """Loader for local transformer models with optimization"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.device = None
        self.streamer = None
    
    async def load_model(self, config: ModelConfig) -> Any:
        """Load local model with optimizations"""
        self.config = config
        self.device = torch.device(config.device)
        
        # Configure quantization if requested
        quantization_config = self._get_quantization_config(config)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path or config.model_name,
            trust_remote_code=True
        )
        
        # Configure model loading arguments
        model_kwargs = {
            "torch_dtype": self._get_torch_dtype(config.dtype),
            "device_map": "auto" if config.max_memory else config.device,
            "trust_remote_code": True,
            "offload_folder": config.offload_folder,
            "offload_state_dict": config.offload_state_dict
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        if config.max_memory:
            model_kwargs["max_memory"] = config.max_memory
        
        # Load model
        logger.info(f"Loading model {config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path or config.model_name,
            **model_kwargs
        )
        
        # Apply optimizations
        await self._apply_optimizations(config)
        
        # Setup streamer if needed
        if config.stream:
            self.streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
        
        # Log memory usage
        self._log_memory_usage()
        
        return self.model
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using local model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generation_config = {
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_length),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    async def stream_generate(self, prompt: str, **kwargs):
        """Stream generation using local model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generation_config = {
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_length),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.repetition_penalty),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Custom streaming implementation
        generated_tokens = []
        past_key_values = None
        
        for _ in range(generation_config["max_new_tokens"]):
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if generation_config["temperature"] > 0:
                logits = logits / generation_config["temperature"]
            
            # Apply top-k and top-p filtering
            filtered_logits = self._top_k_top_p_filtering(
                logits,
                top_k=generation_config["top_k"],
                top_p=generation_config["top_p"]
            )
            
            # Sample
            probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode and yield
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text
            
            # Update inputs for next iteration
            inputs["input_ids"] = next_token
            past_key_values = outputs.past_key_values
            
            # Check for EOS
            if next_token[0].item() == self.tokenizer.eos_token_id:
                break
    
    def _get_quantization_config(self, config: ModelConfig) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration"""
        if config.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=self._get_torch_dtype(config.dtype)
            )
        elif config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._get_torch_dtype(config.dtype),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        return None
    
    def _get_torch_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert string dtype to torch dtype"""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8
        }
        return dtype_map.get(dtype_str, torch.float16)
    
    async def _apply_optimizations(self, config: ModelConfig):
        """Apply various optimizations to the model"""
        
        # Flash Attention
        if config.use_flash_attention and self._check_flash_attention_available():
            logger.info("Enabling Flash Attention")
            self._enable_flash_attention()
        
        # Gradient checkpointing
        if config.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable()
        
        # Triton optimizations
        if config.use_triton and self._check_triton_available():
            logger.info("Enabling Triton optimizations")
            self._enable_triton_optimizations()
        
        # xFormers
        if config.use_xformers and self._check_xformers_available():
            logger.info("Enabling xFormers")
            self._enable_xformers()
        
        # Compile model with torch.compile if available
        if hasattr(torch, 'compile') and not (config.load_in_8bit or config.load_in_4bit):
            logger.info("Compiling model with torch.compile")
            self.model = torch.compile(self.model, mode="reduce-overhead")
    
    def _check_flash_attention_available(self) -> bool:
        """Check if Flash Attention is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _enable_flash_attention(self):
        """Enable Flash Attention in the model"""
        # This would require model-specific implementation
        # For now, we'll set a flag that can be used by the model
        if hasattr(self.model.config, 'use_flash_attention'):
            self.model.config.use_flash_attention = True
    
    def _check_triton_available(self) -> bool:
        """Check if Triton is available"""
        try:
            import triton
            return True
        except ImportError:
            return False
    
    def _enable_triton_optimizations(self):
        """Enable Triton optimizations"""
        # Triton-specific optimizations would go here
        pass
    
    def _check_xformers_available(self) -> bool:
        """Check if xFormers is available"""
        try:
            import xformers
            return True
        except ImportError:
            return False
    
    def _enable_xformers(self):
        """Enable xFormers optimizations"""
        if hasattr(self.model, 'enable_xformers_memory_efficient_attention'):
            self.model.enable_xformers_memory_efficient_attention()
    
    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 0.0,
        filter_value: float = -float('Inf')
    ) -> torch.Tensor:
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering"""
        top_k = min(top_k, logits.size(-1))  # Safety check
        
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        
        return logits
    
    def _log_memory_usage(self):
        """Log current memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        # Log CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / 1024**3
        logger.info(f"CPU Memory: {cpu_memory:.2f}GB")


class ModelOptimizer:
    """Optimizes model configuration based on hardware capabilities"""
    
    @staticmethod
    def optimize_for_hardware(
        config: ModelConfig,
        hardware: HardwareProfile
    ) -> ModelConfig:
        """Optimize model configuration for detected hardware"""
        
        optimized = config
        
        # RTX 3090 specific optimizations
        if "3090" in hardware.device_name:
            logger.info("Detected RTX 3090, applying optimizations")
            
            # 24GB VRAM optimizations
            if hardware.total_memory >= 24000:
                optimized.use_flash_attention = True
                optimized.dtype = "float16"  # FP16 for Ampere
                
                # Model size recommendations
                if hardware.available_memory > 20000:
                    logger.info("Sufficient memory for 13B model")
                elif hardware.available_memory > 15000:
                    logger.info("Sufficient memory for 7B model")
                    optimized.load_in_8bit = True
                else:
                    logger.info("Limited memory, using 4-bit quantization")
                    optimized.load_in_4bit = True
            
            # Enable Ampere-specific features
            if hardware.compute_capability >= (8, 6):
                optimized.use_triton = True
        
        # Multi-GPU setup
        if hardware.multi_gpu:
            logger.info(f"Detected {hardware.num_gpus} GPUs, configuring for multi-GPU")
            optimized.max_memory = {i: f"{hardware.total_memory}MB" for i in range(hardware.num_gpus)}
        
        # CPU offloading for limited VRAM
        if hardware.available_memory < 8000:
            logger.info("Limited VRAM, enabling CPU offloading")
            optimized.offload_folder = "./offload"
            optimized.offload_state_dict = True
        
        return optimized


class UnifiedModelLoader:
    """Unified loader that handles both API and local models"""
    
    def __init__(self):
        self.loaders: Dict[str, ModelLoader] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.hardware_profile = HardwareProfile.detect()
    
    async def load_model(
        self,
        model_id: str,
        config: ModelConfig,
        optimize_for_hardware: bool = True
    ) -> ModelLoader:
        """Load a model with given configuration"""
        
        # Optimize configuration if requested
        if optimize_for_hardware:
            config = ModelOptimizer.optimize_for_hardware(config, self.hardware_profile)
        
        # Select appropriate loader
        if config.provider in ["anthropic", "openai"]:
            loader = APIModelLoader()
        else:
            loader = LocalModelLoader()
        
        # Load model
        await loader.load_model(config)
        
        # Store for later use
        self.loaders[model_id] = loader
        self.configs[model_id] = config
        
        logger.info(f"Successfully loaded model {model_id}")
        return loader
    
    def get_loader(self, model_id: str) -> Optional[ModelLoader]:
        """Get a previously loaded model"""
        return self.loaders.get(model_id)
    
    async def unload_model(self, model_id: str):
        """Unload a model and free resources"""
        if model_id in self.loaders:
            loader = self.loaders[model_id]
            
            # Clean up local models
            if isinstance(loader, LocalModelLoader):
                if loader.model:
                    del loader.model
                if loader.tokenizer:
                    del loader.tokenizer
                
                # Force garbage collection
                gc.collect()
                torch.cuda.empty_cache()
            
            del self.loaders[model_id]
            del self.configs[model_id]
            
            logger.info(f"Unloaded model {model_id}")
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded model IDs"""
        return list(self.loaders.keys())
    
    def get_hardware_profile(self) -> HardwareProfile:
        """Get detected hardware profile"""
        return self.hardware_profile
    
    async def benchmark_model(self, model_id: str, prompt: str = "Hello, world!") -> Dict[str, Any]:
        """Benchmark a loaded model"""
        loader = self.get_loader(model_id)
        if not loader:
            raise ValueError(f"Model {model_id} not loaded")
        
        import time
        
        # Warmup
        await loader.generate(prompt, max_tokens=10)
        
        # Benchmark generation
        start_time = time.time()
        response = await loader.generate(prompt, max_tokens=100)
        generation_time = time.time() - start_time
        
        # Calculate metrics
        tokens_generated = len(response.split())
        tokens_per_second = tokens_generated / generation_time
        
        # Memory usage
        if isinstance(loader, LocalModelLoader):
            memory_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        else:
            memory_used = 0
        
        return {
            "model_id": model_id,
            "generation_time": generation_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second,
            "memory_used_gb": memory_used,
            "response_preview": response[:100] + "..." if len(response) > 100 else response
        }


# Convenience functions

async def create_consciousness_aware_loader(
    model_name: str = "claude-3-opus-20240229",
    api_key: Optional[str] = None
) -> UnifiedModelLoader:
    """Create a loader configured for consciousness-aware generation"""
    
    loader = UnifiedModelLoader()
    
    # Default to Anthropic API
    config = ModelConfig(
        provider="anthropic",
        model_name=model_name,
        api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        consciousness_aware=True,
        emotional_modulation=True,
        stream=True
    )
    
    await loader.load_model("consciousness_model", config)
    
    return loader


async def create_local_optimized_loader(
    model_path: str,
    model_name: str = "local_model"
) -> UnifiedModelLoader:
    """Create a loader optimized for local model execution"""
    
    loader = UnifiedModelLoader()
    
    config = ModelConfig(
        provider="local",
        model_name=model_name,
        model_path=model_path,
        consciousness_aware=True,
        emotional_modulation=True,
        stream=True
    )
    
    await loader.load_model(model_name, config, optimize_for_hardware=True)
    
    return loader