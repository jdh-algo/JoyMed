import os
from typing import Any

class LLMRegistry:
    _models = {}
    
    @classmethod
    def register(cls, name):
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name):
        if name not in cls._models:
            raise ValueError(f"Model {name} not found in registry, supported models: {cls._models}")
        return cls._models[name]

@LLMRegistry.register("Qwen2-VL")
class Qwen2VL:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.Qwen2_VL.Qwen2_VL_vllm import Qwen2VL
        else:
            from models.Qwen2_VL.Qwen2_VL_hf import Qwen2VL
        return Qwen2VL(model_path, args)

@LLMRegistry.register("Qwen2.5-VL") 
class Qwen2_5_VL:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.Qwen2_5_VL.Qwen2_5_VL_vllm import Qwen2_5_VL
        else:
            from models.Qwen2_5_VL.Qwen2_5_VL_hf import Qwen2_5_VL
        return Qwen2_5_VL(model_path, args)

@LLMRegistry.register("Qwen3-VL") 
class Qwen3_VL:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.Qwen3.Qwen3_VL import Qwen3VL
        return Qwen3VL(model_path, args)

@LLMRegistry.register("JoyMed") 
class JoyMed:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.JoyMed.JoyMed import JoyMed
        return JoyMed(model_path, args)
        
# @LLMRegistry.register("LingShu")
# class LingShu:
#     def __new__(cls, model_path: str, args: Any) -> Any:
#         if os.environ.get("use_vllm", "True") == "True":
#             from models.LingShu.LingShu_vllm import LingShu
#         else:
#             from models.LingShu.LingShu_hf import LingShu
#         return LingShu(model_path, args)


@LLMRegistry.register("LingShu")
class LingShu:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.Qwen2_5_VL.Qwen2_5_VL_vllm import Qwen2_5_VL
        else:
            from models.Qwen2_5_VL.Qwen2_5_VL_hf import Qwen2_5_VL
        return Qwen2_5_VL(model_path, args)

@LLMRegistry.register("Citrus_v")
class Citrus_v:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.Citrus_v.Citrus_v import Citrus_v
        return Citrus_v(model_path, args)

@LLMRegistry.register("VitLlama")
class VitLlama:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.VitLlama.VitLlama import VitLlama
        return VitLlama(model_path, args)

@LLMRegistry.register("VitQwen")
class VitQwen:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.VitQwen.VitQwen import VitQwen
        return VitQwen(model_path, args)

@LLMRegistry.register("BiMediX2")
class BiMediX2:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.BiMediX2.BiMediX2_hf import BiMediX2
        return BiMediX2(model_path, args)

@LLMRegistry.register("LLava_Med")
class LLavaMed:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.LLava_Med.LLava_Med_vllm import LLavaMed
        else:
            from models.LLava_Med.LLava_Med_hf import LLavaMed
        return LLavaMed(model_path, args)

@LLMRegistry.register("LLava_OV") 
class LLava_OV:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.LLava_OV.LLava_OV import LLava_OV
        return LLava_OV(model_path, args)
        
@LLMRegistry.register("HuatuoGPT")
class HuatuoGPT:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.HuatuoGPT.HuatuoGPT_vllm import HuatuoGPT
        else:
            from models.HuatuoGPT.HuatuoGPT_hf import HuatuoGPT
        return HuatuoGPT(model_path, args)

@LLMRegistry.register("MedPLIB")
class MedPLIB:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.MedPLIB.MedPLIB import MedPLIB
        return MedPLIB(model_path, args)
        
@LLMRegistry.register("InternVL")
class InternVL:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if "InternVL3_5" in model_path:
            from models.InternVL.InternVL3_5 import InternVL3_5 as InternVL
        else:
            if os.environ.get("use_vllm", "True") == "True":
                from models.InternVL.InternVL_vllm import InternVL
            else:
                from models.InternVL.InternVL_hf import InternVL
        return InternVL(model_path, args)

@LLMRegistry.register("Llama-3.2")
class LlamaVision:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.Llama_3.Llama_3_2_vision_instruct_vllm import LlamaVision
        return LlamaVision(model_path, args)

@LLMRegistry.register("LLava")
class Llava:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if os.environ.get("use_vllm", "True") == "True":
            from models.LLava.LLava_vllm import Llava
        else:
            from models.LLava.LLava_hf import Llava
        return Llava(model_path, args)


@LLMRegistry.register("Janus")
class Janus:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.Janus.Janus import Janus
        return Janus(model_path, args)


@LLMRegistry.register("HealthGPT")
class HealthGPT:
    def __new__(cls, model_path: str, args: Any) -> Any:
        if "Phi-3" in model_path:
            from models.HealthGPT.HealthGPT_phi3 import HealthGPT
        elif "Phi-4" in model_path:
            from models.HealthGPT.HealthGPT_phi4 import HealthGPT
        else:
            from models.HealthGPT.HealthGPT import HealthGPT
        return HealthGPT(model_path, args)

@LLMRegistry.register("BiomedGPT")
class BiomedGPT:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.BiomedGPT.BiomedGPT import BiomedGPT
        return BiomedGPT(model_path, args)

@LLMRegistry.register("Baichuan")
class Baichuan:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.Baichuan.baichuan_m2 import Baichuan
        return Baichuan(model_path, args)
        
@LLMRegistry.register("TestModel")
class TestModel:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.TestModel.TestModel import TestModel
        return TestModel(model_path)

@LLMRegistry.register("Vllm_Text")
class VllmText:
    def __new__(cls, model_path: str, args: Any) -> Any:
        preprocessor_config_path = os.path.join(model_path, "preprocessor_config.json")
        if os.path.exists(preprocessor_config_path):
            from models.vllm_text.vllm_processor import Vllm_Text
        else:
            from models.vllm_text.vllm_tokenizer import Vllm_Text
        return Vllm_Text(model_path, args)

@LLMRegistry.register("MedGemma")
class MedGemma:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.MedGemma.MedGemma import MedGemma
        return MedGemma(model_path, args)

@LLMRegistry.register("MedGemma_1_5")
class MedGemma_1_5:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.MedGemma.MedGemma_1_5 import MedGemma_1_5
        return MedGemma_1_5(model_path, args)
        
@LLMRegistry.register("Med_Flamingo")
class MedFlamingo:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.Med_Flamingo.Med_Flamingo_hf import Med_Flamingo
        return Med_Flamingo(model_path, args)

@LLMRegistry.register("MedDr")
class MedDr:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.MedDr.MedDr import MedDr
        return MedDr(model_path, args)

@LLMRegistry.register("Hulu")
class Hulu:
    def __new__(cls, model_path: str, args: Any) -> Any:
        from models.Hulu.Hulu import Hulu
        return Hulu(model_path, args)

@LLMRegistry.register("API")
class API:
    def __new__(cls, api_url: str, args: Any) -> Any:
        from models.API.API import API
        return API(api_url, args)


def init_llm(args):
    model_class = LLMRegistry.get_model(args.model_name)
    if args.model_name == "API":
        return model_class(args.api_url, args)
    else:
        return model_class(args.model_path, args)
    # try:
    #     model_class = LLMRegistry.get_model(args.model_name)
    #     if args.model_name == "API":
    #         return model_class(args.api_url, args)
    #     else:
    #         return model_class(args.model_path, args)
    # except ValueError as e:
    #     raise ValueError(f"{args.model_name} not supported") from e

