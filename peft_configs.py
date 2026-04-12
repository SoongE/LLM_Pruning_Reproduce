# DoRA
from peft import AdaLoraConfig, TaskType, BOFTConfig, FourierFTConfig
from peft import LoraConfig
from peft import VeraConfig

TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def get_adapter_config(method, total_step):
    method = method.lower()
    assert method in ['lora', 'dora', 'adalora', 'vera', 'boft', 'fourier']
    if method == 'lora':
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=TARGET_MODULES,
            bias="none",
            inference_mode=False,
        )
    elif method == 'dora':
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            use_dora=True,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=TARGET_MODULES,
            bias="none",
            inference_mode=False,
        )
    elif method == 'adalora':  # AdaLoRA
        config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_r=32,
            init_r=64,
            tinit=200,
            tfinal=1000,
            deltaT=10,
            total_step=total_step,
            lora_alpha=64,
            target_modules=TARGET_MODULES,
        )
    elif method == 'vera':
        config = VeraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=128,
            vera_dropout=0.05,
            target_modules=TARGET_MODULES,
        )
    elif method == 'boft':
        config = BOFTConfig(
            task_type=TaskType.CAUSAL_LM,
            boft_block_size=2,  # Butterfly factor block size
            boft_n_butterfly_factor=2,  # Butterfly factorization
            boft_dropout=0.05,
            bias="none",
            target_modules=TARGET_MODULES,
        )
    elif method =='fourier':
        config = FourierFTConfig(
            task_type=TaskType.CAUSAL_LM,
            n_frequency=2000,  # 학습할 주파수 성분 수. 적을수록 파라미터 ↓
            scaling=150.0,
            target_modules=TARGET_MODULES,
        )
    else:
        raise ValueError(f"Unknown adapter method: {method}")

    return config
