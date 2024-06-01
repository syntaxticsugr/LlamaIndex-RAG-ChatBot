from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM



def initialize_bot(temperature, maxtokens, topk, vec_db):
    llm = HuggingFaceLLM(
        model_name="NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        generate_kwargs={"temperature": temperature, "top_k": topk},
        max_new_tokens=maxtokens
    )

    query_engine = vec_db.as_query_engine(llm=llm)

    return query_engine, "### ChatBot Initialized."



def create_db(nodes):
    embedding = HuggingFaceEmbedding()
    index = VectorStoreIndex(nodes=nodes, embed_model=embedding)
    return index



def load_docs(list_file_path):
    docs = SimpleDirectoryReader(input_files=list_file_path).load_data()
    text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=10)
    nodes = text_splitter.get_nodes_from_documents(documents=docs)
    return nodes



def initialize_db(file_obj_list):
    file_path_list = [x.name for x in file_obj_list if x is not None]
    nodes = load_docs(file_path_list)
    vector_db = create_db(nodes)
    return vector_db, "### Database Created."
