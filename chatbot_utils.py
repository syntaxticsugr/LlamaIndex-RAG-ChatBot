from llama_index.core import PromptTemplate, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
import torch



def getResponse(query_engine, prompt, history):
    response = query_engine.query(prompt).response

    history.append((prompt, response))

    return "", history



def initialize_bot(temperature, maxtokens, topk, vec_db):
    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
    - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
    - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
    - StableLM will refuse to participate in anything that could harm a human.
    """

    query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=maxtokens,
        generate_kwargs={"temperature": temperature, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="StabilityAI/stablelm-tuned-alpha-3b",
        model_name="StabilityAI/stablelm-tuned-alpha-3b",
        device_map="auto",
        stopping_ids=[50278, 50279, 50277, 1, 0],
        tokenizer_kwargs={"max_length": 4096},
        # uncomment this if using CUDA to reduce memory usage
        model_kwargs={"torch_dtype": torch.float16}
    )

    query_engine = vec_db.as_query_engine(llm=llm)

    return query_engine, "### ChatBot Initialized."



def initialize_db(file_obj_list):
    file_path_list = [x.name for x in file_obj_list if x is not None]

    docs = SimpleDirectoryReader(input_files=file_path_list).load_data()

    parser = SentenceSplitter(chunk_size=200, chunk_overlap=10)

    nodes = parser.get_nodes_from_documents(documents=docs)

    embedding = HuggingFaceEmbedding()

    index = VectorStoreIndex(nodes=nodes, embed_model=embedding)

    return index, "### Database Created."
