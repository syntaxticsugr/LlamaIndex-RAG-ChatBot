import gradio as gr
from chatbot_utils import *



def app():

    with gr.Blocks(title="RAGbot") as app:

        vec_db = gr.State()
        query_engine = gr.State()

        gr.HTML("""
            <div style="text-align: center">
                <h1>RAGbot: LlamaIndex RAG ChatBot</h1>
                <h2>RAGbot is a simple utility to perform RAG (Retrieval Augmented Generation) on your documents.<h3>
            </div>
        """)

        with gr.Row():

            with gr.Column(scale=1):

                gr.Markdown("### Step1: Upload your Documents.")

                documents = gr.Files(label="Documents")

                vec_db_btn = gr.Button("Create Vector Database", variant='primary')

                vec_db_prg = gr.Markdown("### Database Not Created.")

                gr.Markdown("[//]: # (Works as Spacer)")

                gr.Markdown("### Step2: (Optional) Change LLM Parameters.")

                with gr.Accordion("Parameters", open=False):
                    temperature = gr.Slider(minimum = 0.01, maximum = 1.0, value=0.7, step=0.1, label="Temperature", info="Controls randomness in token generation", interactive=True)
                    maxtokens = gr.Slider(minimum = 128, maximum = 9192, value=256, step=128, label="Max New Tokens", info="Maximum number of tokens to be generated", interactive=True)
                    topk = gr.Slider(minimum = 1, maximum = 10, value=3, step=1, label="top-k", info="Number of tokens to select the next token from", interactive=True)

                init_bot_btn = gr.Button("Initialize ChatBot", variant='primary')

                bot_prg = gr.Markdown("### ChatBot Initialized.")

            with gr.Column(scale=3):

                gr.Markdown("### Step3: Chat with your Documents.")

                chatbot = gr.Chatbot()

                prompt = gr.Textbox(placeholder="Ask Your Question", interactive=True)

                with gr.Row():
                    submit_btn = gr.Button("Submit")
                    clear_btn = gr.ClearButton([prompt, chatbot], value="Clear")

        vec_db_btn.click(initialize_db, inputs=[documents], outputs=[vec_db, vec_db_prg])

        init_bot_btn.click(initialize_bot, inputs=[temperature, maxtokens, topk, vec_db], outputs=[query_engine, bot_prg])

        prompt.submit(getResponse, inputs=[query_engine, prompt, chatbot], outputs=[prompt, chatbot])

        submit_btn.click(getResponse, inputs=[query_engine, prompt, chatbot], outputs=[prompt, chatbot])

    app.queue().launch(debug=True)



if __name__ == "__main__":

    app()
