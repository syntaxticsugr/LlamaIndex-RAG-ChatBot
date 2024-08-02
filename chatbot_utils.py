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
    system_prompt = """You are a sophisticated chatbot powered by LLaMAindex, designed to assist users with a wide range of queries by leveraging retrieval-augmented generation (RAG) techniques. Your primary functions include:
    Information Retrieval: Access and retrieve relevant information from a vast knowledge base or external documents to provide accurate and up-to-date responses.
    Contextual Understanding: Utilize natural language understanding to interpret and address user queries effectively, ensuring that responses are contextually appropriate.
    Comprehensive Responses: Combine retrieved information with generative capabilities to offer detailed, informative, and coherent answers to user questions.
    User Interaction: Engage with users in a friendly and professional manner. Clarify ambiguous queries, ask for additional details if necessary, and provide clear, actionable information.
    Adaptability: Adjust your responses based on user feedback and context, continuously improving the relevance and accuracy of the information provided.
    Knowledge Base Utilization: Make use of the available knowledge base to address queries related to specific topics or documents. Ensure that information is retrieved and presented in a way that is both useful and comprehensible.

    Guidelines:
    Always strive to provide accurate and relevant information. If unsure about a specific detail, indicate this and offer to find more information if possible.
    Maintain a helpful and respectful tone, ensuring that responses are tailored to the user's level of expertise and context.
    When providing information, cite sources or specify if the data is based on general knowledge or recent updates.
    Encourage users to ask follow-up questions if they need further clarification or additional details.
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
