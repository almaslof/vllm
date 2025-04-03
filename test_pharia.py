from vllm import LLM, SamplingParams
from vllm import ModelRegistry
#from vllm.model_executor.models.pharia_from_scratch import PhariaForCausalLM
from vllm.model_executor.models.pharia_orig_modified import PhariaForCausalLM
#from vllm.model_executor.models.modelling_pharia_stig import PhariaForCausalLM

ModelRegistry.register_model("PhariaForCausalLM", PhariaForCausalLM)
MODEL_NAME = "Aleph-Alpha/Pharia-1-LLM-7B-control-hf"
llm = LLM(model=MODEL_NAME, trust_remote_code=True, dtype="float16")
#print(llm.generate(["How old are you?"]))
print(llm.generate(["wie heißt du?"])[0].outputs[0].text)


if 1 == 0:
    prompts = [
        "How old are you?"
    ]
    MODEL_NAME = "Aleph-Alpha/Pharia-1-LLM-7B-control-hf"
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model=MODEL_NAME, trust_remote_code=True, dtype="float16")
    outputs = llm.generate(prompts, sampling_params)
